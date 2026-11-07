"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def sign_edges(graph, budget):
    deadline = time.time() + 82.0  # Safety margin before 90s kill
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    # Precompute edge endpoints as arrays for speed
    eu = np.array([u for u, v in edges], dtype=np.int32)
    ev = np.array([v for u, v in edges], dtype=np.int32)
    
    # Build vertex-to-edge mapping (each vertex in prism has degree 3)
    vert_edges = [[] for _ in range(n)]
    for i in range(num_edges):
        vert_edges[eu[i]].append(i)
        vert_edges[ev[i]].append(i)
    
    def build_signed_matrix(signs):
        A = np.zeros((n, n), dtype=np.float64)
        np.add.at(A, (eu, ev), signs.astype(np.float64))
        np.add.at(A, (ev, eu), signs.astype(np.float64))
        return A
    
    def spectral_radius(A):
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def is_connected_lift(signs):
        # Fast connectivity check using BFS on the 2-lift
        visited = np.zeros(2 * n, dtype=bool)
        queue = [0]  # (vertex 0, layer 0)
        visited[0] = True
        
        twist = (signs == -1).astype(np.int8)
        
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            v = node // 2
            layer = node % 2
            for ei in vert_edges[v]:
                u = eu[ei] if ev[ei] == v else ev[ei]
                t = twist[ei]
                nl = layer ^ t
                nn = u * 2 + nl
                if not visited[nn]:
                    visited[nn] = True
                    queue.append(nn)
        
        return all(visited)
    
    def is_valid(signs):
        return is_connected_lift(signs)
    
    # Continuous phase-based optimization
    def spectral_radius_continuous(thetas):
        A = np.zeros((n, n), dtype=np.float64)
        c = np.cos(thetas)
        np.add.at(A, (eu, ev), c)
        np.add.at(A, (ev, eu), c)
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def gradient_spectral_radius(thetas):
        eps = 1e-6
        grad = np.zeros_like(thetas)
        base_r = spectral_radius_continuous(thetas)
        for i in range(num_edges):
            theta_plus = thetas.copy()
            theta_plus[i] += eps
            r_plus = spectral_radius_continuous(theta_plus)
            grad[i] = (r_plus - base_r) / eps
        return grad
    
    def sign_from_phases(thetas):
        signs = np.where(np.cos(thetas) >= 0, 1, -1).astype(np.int8)
        return signs
    
    rng = np.random.default_rng(42)
    
    best_signs = None
    best_radius = float('inf')
    
    # Phase 1: Continuous optimization from multiple random starts
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        thetas = rng.uniform(0, 2 * np.pi, size=num_edges)
        
        lr = 0.1
        for step in range(150):
            if time.time() > deadline:
                break
            
            r = spectral_radius_continuous(thetas)
            grad = gradient_spectral_radius(thetas)
            
            thetas = thetas - lr * grad
            thetas = thetas % (2 * np.pi)
        
        signs = sign_from_phases(thetas)
        
        if is_valid(signs):
            A = build_signed_matrix(signs)
            r = spectral_radius(A)
            if r < best_radius:
                best_radius = r
                best_signs = signs.copy()
    
    # Phase 2: Local search with single and double flips
    if best_signs is not None:
        signs = best_signs.copy()
        A = build_signed_matrix(signs)
        cur_radius = spectral_radius(A)
        
        def flip_and_eval(signs, A, idx):
            u, v = int(eu[idx]), int(ev[idx])
            old_s = float(signs[idx])
            signs[idx] = -signs[idx]
            new_s = float(signs[idx])
            
            A[u, v] = new_s
            A[v, u] = new_s
            
            if not is_valid(signs):
                signs[idx] = -signs[idx]
                A[u, v] = old_s
                A[v, u] = old_s
                return None
            
            r = spectral_radius(A)
            return r
        
        improved = True
        stall_count = 0
        while improved and time.time() < deadline:
            improved = False
            order = rng.permutation(num_edges)
            
            # Single flips
            for idx in order:
                if time.time() > deadline:
                    break
                new_r = flip_and_eval(signs, A, idx)
                if new_r is not None and new_r < cur_radius:
                    cur_radius = new_r
                    improved = True
                    if cur_radius < best_radius:
                        best_radius = cur_radius
                        best_signs = signs.copy()
            
            # Double flips for escape from local minima
            if not improved and time.time() < deadline:
                for _ in range(5):
                    if time.time() > deadline:
                        break
                    i1 = rng.integers(num_edges)
                    i2 = rng.integers(num_edges)
                    if i1 == i2:
                        continue
                    
                    u1, v1 = int(eu[i1]), int(ev[i1])
                    u2, v2 = int(eu[i2]), int(ev[i2])
                    old_s1 = float(signs[i1])
                    old_s2 = float(signs[i2])
                    
                    signs[i1] = -signs[i1]
                    signs[i2] = -signs[i2]
                    new_s1 = float(signs[i1])
                    new_s2 = float(signs[i2])
                    
                    A[u1, v1] = new_s1
                    A[v1, u1] = new_s1
                    A[u2, v2] = new_s2
                    A[v2, u2] = new_s2
                    
                    if is_valid(signs):
                        new_r = spectral_radius(A)
                        if new_r < cur_radius:
                            cur_radius = new_r
                            improved = True
                            if cur_radius < best_radius:
                                best_radius = cur_radius
                                best_signs = signs.copy()
                        else:
                            signs[i1] = -signs[i1]
                            signs[i2] = -signs[i2]
                            A[u1, v1] = old_s1
                            A[v1, u1] = old_s1
                            A[u2, v2] = old_s2
                            A[v2, u2] = old_s2
                    else:
                        signs[i1] = -signs[i1]
                        signs[i2] = -signs[i2]
                        A[u1, v1] = old_s1
                        A[v1, u1] = old_s1
                        A[u2, v2] = old_s2
                        A[v2, u2] = old_s2
            
            if not improved:
                stall_count += 1
            else:
                stall_count = 0
    
    # Phase 3: Fallback random search if needed
    if best_signs is None:
        for _ in range(500):
            if time.time() > deadline:
                break
            signs = rng.choice([-1, 1], size=num_edges)
            if is_valid(signs):
                A = build_signed_matrix(signs)
                r = spectral_radius(A)
                if r < best_radius:
                    best_radius = r
                    best_signs = signs.copy()
    
    if best_signs is None:
        signs = np.ones(num_edges, dtype=int)
        if not is_valid(signs):
            signs = -np.ones(num_edges, dtype=int)
            if not is_valid(signs):
                for _ in range(2000):
                    signs = rng.choice([-1, 1], size=num_edges)
                    if is_valid(signs):
                        break
        best_signs = signs
    
    return best_signs
# EVOLVE-BLOCK-END
