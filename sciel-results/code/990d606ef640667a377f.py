"""Combined program for spectral graph lifts with exhaustive search for small m and optimized local search for large m."""
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
    # a[i]: top cycle edge i (i to i+1)
    # b[i]: bottom cycle edge i (m+i to m+i+1)
    # c[i]: matching edge i (i to m+i)
    
    a_idx = [-1] * m
    b_idx = [-1] * m
    c_idx = [-1] * m
    
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            # Top cycle
            # Determine i such that edge is (i, i+1) or (i+1, i)
            if v == (u + 1) % m:
                a_idx[u] = idx
            elif u == (v + 1) % m:
                a_idx[v] = idx
        elif u >= m and v >= m:
            # Bottom cycle
            u_local = u - m
            v_local = v - m
            if v_local == (u_local + 1) % m:
                b_idx[u_local] = idx
            elif u_local == (v_local + 1) % m:
                b_idx[v_local] = idx
        else:
            # Matching
            i = min(u, v)
            c_idx[i] = idx

    # Verify all indices found
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
        # Outer cycle product: prod(a) * prod(b) * c[0] * c[m-1]
        # Note: The outer cycle traverses top 0->m-1, match m-1, bottom m-1->0, match 0.
        # Product of a: prod(a)
        # Product of b: prod(b)
        # Matches: c[m-1] and c[0]
        p_outer = 1
        for x in a: p_outer *= x
        for x in b: p_outer *= x
        p_outer *= c[0] * c[m-1]
        
        if p_outer == -1:
            return True
        
        # Check small cycles: a[i] * c[i+1] * b[i] * c[i]
        # Indices mod m
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

    # For m <= 5, exhaustive search is feasible and guarantees global optimum
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

    # For m > 5, use local search
    rng = np.random.default_rng(3)
    
    # Initialize with a valid connected lift
    # Start with all +1, then flip one matching edge to ensure connectivity
    # All +1: p_outer = 1. Small cycles = 1. Disconnected.
    # Flip c[0] to -1. p_outer = -1. Connected.
    cur_a = np.ones(m, dtype=np.int8)
    cur_b = np.ones(m, dtype=np.int8)
    cur_c = np.ones(m, dtype=np.int8)
    cur_c[0] = -1
    
    if not is_connected_fast(cur_a, cur_b, cur_c):
        # Fallback: try random
        for _ in range(100):
            cur_a = rng.choice([-1, 1], size=m).astype(np.int8)
            cur_b = rng.choice([-1, 1], size=m).astype(np.int8)
            cur_c = rng.choice([-1, 1], size=m).astype(np.int8)
            if is_connected_fast(cur_a, cur_b, cur_c):
                break
    
    current_radius = spectral_radius(cur_a, cur_b, cur_c)
    best_radius = current_radius
    best_signs = signs_from_abc(cur_a, cur_b, cur_c)

    # Local Search
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
        for _ in range(1000):
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

    # Final verification
    try:
        check_lift(graph, best_signs)
    except:
        pass

    return best_signs
# EVOLVE-BLOCK-END
