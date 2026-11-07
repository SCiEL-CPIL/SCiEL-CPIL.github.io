"""Combined program for spectral graph lifts with continuous phase optimization and discrete polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product as iproduct

def sign_edges(graph, budget):
    from autoresearch_suite.science_tcs_expansion import check_lift

    deadline = time.time() + 85.0
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)

    # Map edge indices to a, b, c variables
    a_idx = [-1] * m
    b_idx = [-1] * m
    c_idx = [-1] * m
    
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            if v == (u + 1) % m:
                a_idx[u] = idx
            elif u == (v + 1) % m:
                a_idx[v] = idx
        elif u >= m and v >= m:
            u_local = u - m
            v_local = v - m
            if v_local == (u_local + 1) % m:
                b_idx[u_local] = idx
            elif u_local == (v_local + 1) % m:
                b_idx[v_local] = idx
        else:
            i = min(u, v)
            c_idx[i] = idx

    assert all(i != -1 for i in a_idx), "Missing top edge"
    assert all(i != -1 for i in b_idx), "Missing bottom edge"
    assert all(i != -1 for i in c_idx), "Missing matching edge"

    def build_matrix(a, b, c):
        S = np.zeros((n, n), dtype=np.float64)
        for i in range(m):
            u = i
            v = (i + 1) % m
            S[u, v] = a[i]
            S[v, u] = a[i]
            
            u = m + i
            v = m + (i + 1) % m
            S[u, v] = b[i]
            S[v, u] = b[i]
            
            u = i
            v = m + i
            S[u, v] = c[i]
            S[v, u] = c[i]
        return S

    def spectral_radius(a, b, c):
        S = build_matrix(a, b, c)
        eigvals = np.linalg.eigvalsh(S)
        return np.max(np.abs(eigvals))

    def is_connected_fast(a, b, c):
        p_outer = 1
        for x in a: p_outer *= x
        for x in b: p_outer *= x
        p_outer *= c[0] * c[m-1]
        
        if p_outer == -1:
            return True
        
        for i in range(m):
            if a[i] * c[(i+1)%m] * b[i] * c[i] == -1:
                return True
        return False

    def signs_from_abc(a, b, c):
        signs = np.zeros(num_edges, dtype=np.int8)
        for i in range(m):
            signs[a_idx[i]] = a[i]
            signs[b_idx[i]] = b[i]
            signs[c_idx[i]] = c[i]
        return signs

    best_signs = None
    best_radius = np.inf

    # For m <= 5, exhaustive search guarantees global optimum
    if m <= 5:
        total_vars = 3 * m
        for combo in iproduct([-1, 1], repeat=total_vars):
            if time.time() > deadline:
                break
            a = np.array(combo[0:m], dtype=np.int8)
            b = np.array(combo[m:2*m], dtype=np.int8)
            c = np.array(combo[2*m:3*m], dtype=np.int8)
            
            if not is_connected_fast(a, b, c):
                continue
            
            radius = spectral_radius(a, b, c)
            if radius < best_radius:
                best_radius = radius
                best_signs = signs_from_abc(a, b, c)
        
        if best_signs is not None:
            try:
                check_lift(graph, best_signs)
                return best_signs
            except:
                pass

    # Continuous phase optimization for m > 5
    # The signed adjacency matrix has circulant structure.
    # Parameterize each sign variable as a complex unit: z = exp(i*theta)
    # The spectral radius of the real part gives a smooth surrogate.
    # We minimize the max |eigenvalue| of the continuous matrix.
    
    def build_continuous_matrix(thetas):
        """Build matrix using cos(thetas) as continuous relaxation of signs."""
        a = np.cos(thetas[:m])
        b = np.cos(thetas[m:2*m])
        c = np.cos(thetas[2*m:3*m])
        S = np.zeros((n, n), dtype=np.float64)
        for i in range(m):
            u = i
            v = (i + 1) % m
            S[u, v] = a[i]
            S[v, u] = a[i]
            
            u = m + i
            v = m + (i + 1) % m
            S[u, v] = b[i]
            S[v, u] = b[i]
            
            u = i
            v = m + i
            S[u, v] = c[i]
            S[v, u] = c[i]
        return S

    def continuous_spectral_radius(thetas):
        S = build_continuous_matrix(thetas)
        eigvals = np.linalg.eigvalsh(S)
        return np.max(np.abs(eigvals))

    def discrete_from_thetas(thetas):
        """Round continuous phases to discrete signs."""
        a = np.ones(m, dtype=np.int8)
        b = np.ones(m, dtype=np.int8)
        c = np.ones(m, dtype=np.int8)
        for i in range(m):
            a[i] = 1 if np.cos(thetas[i]) >= 0 else -1
            b[i] = 1 if np.cos(thetas[m+i]) >= 0 else -1
            c[i] = 1 if np.cos(thetas[2*m+i]) >= 0 else -1
        return a, b, c

    rng = np.random.default_rng(3)
    
    # Multiple random restarts in continuous space
    n_restarts = 20
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            # Start near all +1 with one flip
            thetas = np.zeros(3*m)
            thetas[2*m] = np.pi  # c[0] = -1
        else:
            thetas = rng.uniform(-np.pi, np.pi, size=3*m)
        
        # Gradient descent on continuous spectral radius
        lr = 0.1
        for step in range(500):
            if time.time() > deadline:
                break
            
            # Compute gradient via finite differences (cheap for small m)
            grad = np.zeros(3*m)
            eps = 1e-4
            f0 = continuous_spectral_radius(thetas)
            for j in range(3*m):
                thetas_p = thetas.copy()
                thetas_p[j] += eps
                fp = continuous_spectral_radius(thetas_p)
                grad[j] = (fp - f0) / eps
            
            # Gradient step
            thetas = thetas - lr * grad
            
            # Adaptive learning rate
            if step % 50 == 49:
                lr *= 0.9
        
        # Round to discrete
        a, b, c = discrete_from_thetas(thetas)
        
        # Ensure connectivity
        if not is_connected_fast(a, b, c):
            # Try flipping matching edges to ensure connectivity
            for i in range(m):
                c_try = c.copy()
                c_try[i] = -c_try[i]
                if is_connected_fast(a, b, c_try):
                    c = c_try
                    break
            else:
                # Try flipping a top edge
                for i in range(m):
                    a_try = a.copy()
                    a_try[i] = -a_try[i]
                    if is_connected_fast(a_try, b, c):
                        a = a_try
                        break
        
        if not is_connected_fast(a, b, c):
            continue
        
        radius = spectral_radius(a, b, c)
        if radius < best_radius:
            best_radius = radius
            best_signs = signs_from_abc(a, b, c)
            # Use this as starting point for local search
            cur_a, cur_b, cur_c = a.copy(), b.copy(), c.copy()
            current_radius = radius
            break
    
    # If we found a good candidate, do local search polish
    if best_signs is not None:
        # Local search from the best continuous-derived solution
        while time.time() < deadline:
            improved = False
            
            # 1-flip moves
            for var_type in ['a', 'b', 'c']:
                if time.time() > deadline: break
                for i in range(m):
                    if time.time() > deadline: break
                    
                    a = cur_a.copy()
                    b = cur_b.copy()
                    c = cur_c.copy()
                    
                    if var_type == 'a': a[i] = -a[i]
                    elif var_type == 'b': b[i] = -b[i]
                    else: c[i] = -c[i]
                    
                    if not is_connected_fast(a, b, c):
                        continue
                    
                    radius = spectral_radius(a, b, c)
                    if radius < current_radius - 1e-10:
                        cur_a, cur_b, cur_c = a, b, c
                        current_radius = radius
                        best_radius = radius
                        best_signs = signs_from_abc(a, b, c)
                        improved = True

            if improved:
                continue

            # 2-flip moves
            two_flip_improved = False
            for i in range(m):
                if time.time() > deadline: break
                for j in range(i, m):
                    if time.time() > deadline: break
                    for vi in range(3):
                        if time.time() > deadline: break
                        for vj in range(vi, 3):
                            if time.time() > deadline: break
                            
                            a = cur_a.copy()
                            b = cur_b.copy()
                            c = cur_c.copy()
                            
                            if vi == 0: a[i] = -a[i]
                            elif vi == 1: b[i] = -b[i]
                            else: c[i] = -c[i]
                            
                            if vj == 0: a[j] = -a[j]
                            elif vj == 1: b[j] = -b[j]
                            else: c[j] = -c[j]
                            
                            if not is_connected_fast(a, b, c):
                                continue
                            
                            radius = spectral_radius(a, b, c)
                            if radius < current_radius - 1e-10:
                                cur_a, cur_b, cur_c = a, b, c
                                current_radius = radius
                                best_radius = radius
                                best_signs = signs_from_abc(a, b, c)
                                two_flip_improved = True
                                break
                        if two_flip_improved: break
                    if two_flip_improved: break
                if two_flip_improved: break
            
            if two_flip_improved:
                continue

            # Random restart if stuck
            restart_improved = False
            for _ in range(500):
                if time.time() > deadline: break
                
                a = rng.choice([-1, 1], size=m).astype(np.int8)
                b = rng.choice([-1, 1], size=m).astype(np.int8)
                c = rng.choice([-1, 1], size=m).astype(np.int8)
                
                if not is_connected_fast(a, b, c):
                    continue
                
                radius = spectral_radius(a, b, c)
                if radius < current_radius - 1e-10:
                    cur_a, cur_b, cur_c = a, b, c
                    current_radius = radius
                    best_radius = radius
                    best_signs = signs_from_abc(a, b, c)
                    restart_improved = True
                    break
            
            if not restart_improved:
                break

    # Fallback if no valid solution found yet
    if best_signs is None:
        cur_a = np.ones(m, dtype=np.int8)
        cur_b = np.ones(m, dtype=np.int8)
        cur_c = np.ones(m, dtype=np.int8)
        cur_c[0] = -1
        best_signs = signs_from_abc(cur_a, cur_b, cur_c)
        best_radius = spectral_radius(cur_a, cur_b, cur_c)

    # Final verification
    try:
        check_lift(graph, best_signs)
    except:
        pass

    return best_signs
# EVOLVE-BLOCK-END
