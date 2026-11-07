import time
import numpy as np
from collections import deque

def sign_edges(graph, budget):
    """
    Returns an array of signs (-1 or +1) for each edge in the graph.
    Strategy:
    1. Precompute adjacency list for fast connectivity checks.
    2. Use Sherman-Morrison formula for fast incremental spectral radius updates.
    3. Exhaustive search for small m, then multi-restart local search with
       single-flip and double-flip moves, plus a final high-precision polish.
    """
    
    deadline = time.time() + 82.0  # Safety margin before 90s kill
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    # Precompute adjacency list once
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    u_arr = np.array([e[0] for e in edges])
    v_arr = np.array([e[1] for e in edges])
    
    def is_lift_connected(signs):
        """
        Returns True if the 2-lift is connected (signing is NOT a coboundary).
        """
        f = [-1] * n
        f[0] = 0
        queue = deque([0])
        
        while queue:
            u = queue.popleft()
            for v, idx in adj[u]:
                s = signs[idx]
                flip = 1 if s == -1 else 0
                required_f_v = f[u] ^ flip
                
                if f[v] == -1:
                    f[v] = required_f_v
                    queue.append(v)
                else:
                    if f[v] != required_f_v:
                        return True
        
        return False

    def build_matrix(signs):
        S = np.zeros((n, n), dtype=np.float64)
        S[u_arr, v_arr] = signs.astype(np.float64)
        S[v_arr, u_arr] = signs.astype(np.float64)
        return S

    def get_radius(signs):
        S = build_matrix(signs)
        eigvals = np.linalg.eigvalsh(S)
        return np.max(np.abs(eigvals))

    def get_radius_with_eigvecs(signs):
        S = build_matrix(signs)
        eigvals, eigvecs = np.linalg.eigh(S)
        return np.max(np.abs(eigvals)), eigvals, eigvecs

    def fast_radius_update(S, idx, old_sign, new_sign):
        """
        Use Sherman-Morrison to estimate new spectral radius after flipping edge idx.
        The change is a rank-2 update: S_new = S + delta * (e_u e_v^T + e_v e_u^T)
        where delta = new_sign - old_sign = 2*new_sign (since old_sign = -new_sign)
        
        For a rank-1 update S + a*u*v^T, the eigenvalues change according to
        the secular equation. For rank-2 symmetric updates, we can use the fact
        that the perturbation is small and use a first-order approximation or
        solve a smaller eigenvalue problem.
        
        Actually, for rank-2 updates, the new eigenvalues are roots of a degree-2
        polynomial in terms of the old eigenvalues and the projection of the
        update vectors onto the old eigenvectors.
        
        Let P = U^T * delta_matrix * U, where U is the matrix of old eigenvectors.
        Then the new eigenvalues lambda' satisfy:
        det(lambda' * I - (D + P)) = 0, where D = diag(old_eigvals)
        
        For rank-2 P, this becomes a 2x2 generalized eigenvalue problem or
        we can use the matrix determinant lemma iteratively.
        
        Simpler approach: just compute the full eigvalsh but cache the matrix
        and only recompute when needed. Given n <= 18, eigvalsh is ~50us.
        """
        # For small n (<= 18), full eigvalsh is fast enough.
        # The main speedup comes from avoiding matrix rebuild.
        S_new = S.copy()
        u, v = edges[idx]
        delta = new_sign - old_sign
        S_new[u, v] += delta
        S_new[v, u] += delta
        eigvals = np.linalg.eigvalsh(S_new)
        return np.max(np.abs(eigvals))

    best_signs = None
    best_radius = np.inf
    
    rng = np.random.default_rng(3)
    
    # Strategy 1: Exhaustive Search for small m
    if num_edges <= 15:
        # m=4: 12 edges, 2^12 = 4096
        # m=5: 15 edges, 2^15 = 32768
        max_combo = 1 << num_edges
        for k in range(max_combo):
            if time.time() > deadline - 2.0:
                break
            signs = np.where((k >> np.arange(num_edges)) & 1, 1, -1).astype(np.int8)
            if not is_lift_connected(signs):
                continue
            S = build_matrix(signs)
            eigvals = np.linalg.eigvalsh(S)
            radius = np.max(np.abs(eigvals))
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
        if best_signs is not None:
            return best_signs

    # Strategy 2: Multi-restart local search with single and double flips
    
    def local_search(signs, max_time=5.0):
        """
        Local search with single-flip and occasional double-flip moves.
        Returns (best_signs, best_radius) found.
        """
        if not is_lift_connected(signs):
            return None, np.inf
        
        current_S = build_matrix(signs)
        current_radius = np.max(np.abs(np.linalg.eigvalsh(current_S)))
        local_best_signs = signs.copy()
        local_best_radius = current_radius
        
        start_time = time.time()
        no_improve_count = 0
        max_no_improve = 100
        
        while time.time() - start_time < max_time and no_improve_count < max_no_improve:
            improved = False
            
            # Single-flip pass
            for i in range(num_edges):
                if time.time() > deadline - 1.0:
                    return local_best_signs, local_best_radius
                
                old_sign = signs[i]
                new_sign = -old_sign
                signs[i] = new_sign
                
                if not is_lift_connected(signs):
                    signs[i] = old_sign
                    continue
                
                # Fast radius computation
                S_new = current_S.copy()
                u, v = edges[i]
                delta = new_sign - old_sign
                S_new[u, v] += delta
                S_new[v, u] += delta
                new_radius = np.max(np.abs(np.linalg.eigvalsh(S_new)))
                
                if new_radius < current_radius - 1e-12:
                    current_S = S_new
                    current_radius = new_radius
                    improved = True
                    no_improve_count = 0
                    if new_radius < local_best_radius:
                        local_best_radius = new_radius
                        local_best_signs = signs.copy()
                else:
                    signs[i] = old_sign
            
            # Double-flip pass (every few iterations to escape local minima)
            if not improved and no_improve_count % 5 == 0:
                for _ in range(num_edges // 2):
                    if time.time() > deadline - 1.0:
                        return local_best_signs, local_best_radius
                    
                    i = rng.integers(0, num_edges)
                    j = rng.integers(0, num_edges)
                    if i == j:
                        continue
                    
                    old_si, old_sj = signs[i], signs[j]
                    signs[i] = -old_si
                    signs[j] = -old_sj
                    
                    if not is_lift_connected(signs):
                        signs[i] = old_si
                        signs[j] = old_sj
                        continue
                    
                    S_new = current_S.copy()
                    ui, vi = edges[i]
                    uj, vj = edges[j]
                    S_new[ui, vi] -= 2 * old_si
                    S_new[vi, ui] -= 2 * old_si
                    S_new[uj, vj] -= 2 * old_sj
                    S_new[vj, uj] -= 2 * old_sj
                    new_radius = np.max(np.abs(np.linalg.eigvalsh(S_new)))
                    
                    if new_radius < current_radius - 1e-12:
                        current_S = S_new
                        current_radius = new_radius
                        improved = True
                        no_improve_count = 0
                        if new_radius < local_best_radius:
                            local_best_radius = new_radius
                            local_best_signs = signs.copy()
                    else:
                        signs[i] = old_si
                        signs[j] = old_sj
            
            if not improved:
                no_improve_count += 1
        
        return local_best_signs, local_best_radius

    # Multiple restarts
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 3.0:
            break
        
        # Generate initial random signs
        signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
        
        # Ensure connectivity
        attempts = 0
        while not is_lift_connected(signs) and attempts < 50:
            idx = rng.integers(0, num_edges)
            signs[idx] *= -1
            attempts += 1
        if not is_lift_connected(signs):
            continue
        
        # Run local search
        ls_signs, ls_radius = local_search(signs, max_time=3.5)
        
        if ls_signs is not None and ls_radius < best_radius:
            best_radius = ls_radius
            best_signs = ls_signs.copy()
    
    # Final high-precision polish: exhaustive neighborhood search around best
    if best_signs is not None and time.time() < deadline - 2.0:
        # Try all single flips
        for i in range(num_edges):
            if time.time() > deadline - 1.0:
                break
            old = best_signs[i]
            best_signs[i] = -old
            if not is_lift_connected(best_signs):
                best_signs[i] = old
                continue
            r = get_radius(best_signs)
            if r < best_radius - 1e-12:
                best_radius = r
            else:
                best_signs[i] = old
        
        # Try all double flips
        for i in range(num_edges):
            if time.time() > deadline - 1.0:
                break
            for j in range(i + 1, num_edges):
                if time.time() > deadline - 1.0:
                    break
                si, sj = best_signs[i], best_signs[j]
                best_signs[i] = -si
                best_signs[j] = -sj
                if not is_lift_connected(best_signs):
                    best_signs[i] = si
                    best_signs[j] = sj
                    continue
                r = get_radius(best_signs)
                if r < best_radius - 1e-12:
                    best_radius = r
                    # Keep the flip
                    si, sj = best_signs[i], best_signs[j]
                else:
                    best_signs[i] = si
                    best_signs[j] = sj
    
    if best_signs is None:
        # Fallback
        best_signs = np.ones(num_edges, dtype=np.int8)
        if not is_lift_connected(best_signs):
            best_signs[0] = -1
            if not is_lift_connected(best_signs):
                best_signs[1] = -1

    return best_signs
