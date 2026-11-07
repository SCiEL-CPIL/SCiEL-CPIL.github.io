def sign_edges(graph, budget):
    import numpy as np
    import time
    
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin
    
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    # Build adjacency structure
    adj = {i: [] for i in range(n)}
    edge_idx = {}
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
        edge_idx[(u, v)] = idx
        edge_idx[(v, u)] = idx
    
    def is_coboundary(signs):
        """Check if signs is a coboundary: sign(u,v) = f(u)*f(v) for some f: V -> {±1}"""
        f = {}
        f[0] = 1
        stack = [0]
        while stack:
            u = stack.pop()
            for v, idx in adj[u]:
                s = signs[idx]
                if v not in f:
                    f[v] = f[u] * s
                    stack.append(v)
                else:
                    if f[v] != f[u] * s:
                        return False
        return True
    
    def spectral_radius(signs):
        """Compute spectral radius of signed adjacency matrix"""
        A = np.zeros((n, n))
        for idx, (u, v) in enumerate(edges):
            A[u, v] = signs[idx]
            A[v, u] = signs[idx]
        eigvals = np.linalg.eigvalsh(A)
        return max(abs(eigvals[0]), abs(eigvals[-1]))
    
    # Initial random search
    rng = np.random.default_rng(3)
    
    best_signs = None
    best_radius = float('inf')
    
    # Try random sign assignments
    for _ in range(200):
        if time.time() > deadline:
            break
        signs = rng.choice([-1, 1], size=num_edges)
        if is_coboundary(signs):
            continue
        r = spectral_radius(signs)
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()
    
    if best_signs is None:
        # Fallback: all +1 except one -1 to ensure not coboundary
        signs = np.ones(num_edges, dtype=np.int8)
        signs[0] = -1
        if not is_coboundary(signs):
            best_signs = signs
            best_radius = spectral_radius(signs)
        else:
            signs[1] = -1
            best_signs = signs
            best_radius = spectral_radius(signs)
    
    # Local search: flip signs to reduce spectral radius
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(num_edges):
            if time.time() > deadline:
                break
            # Try flipping this sign
            old_sign = best_signs[idx]
            best_signs[idx] = -old_sign
            
            if is_coboundary(best_signs):
                best_signs[idx] = old_sign
                continue
            
            r = spectral_radius(best_signs)
            if r < best_radius:
                best_radius = r
                improved = True
            else:
                best_signs[idx] = old_sign
    
    # Second round of local search for polish
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(num_edges):
            if time.time() > deadline:
                break
            old_sign = best_signs[idx]
            best_signs[idx] = -old_sign
            
            if is_coboundary(best_signs):
                best_signs[idx] = old_sign
                continue
            
            r = spectral_radius(best_signs)
            if r < best_radius - 1e-12:
                best_radius = r
                improved = True
            else:
                best_signs[idx] = old_sign
    
    return best_signs.astype(np.int8)
