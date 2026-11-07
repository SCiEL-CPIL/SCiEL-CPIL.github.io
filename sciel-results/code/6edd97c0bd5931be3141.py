def sign_edges(graph, budget):
    import numpy as np
    from collections import deque

    n = graph['n']
    edges = list(graph['edges'])
    E = len(edges)
    m = n // 2

    # Build adjacency list with edge indices
    adj = {v: [] for v in range(n)}
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))

    def is_unbalanced(signs):
        """
        Returns True if the signed graph is unbalanced (lift is connected).
        Returns False if it is balanced (lift is disconnected).
        """
        tau = np.zeros(n, dtype=np.int8)
        tau[0] = 1
        q = deque([0])
        while q:
            u = q.popleft()
            for (v, idx) in adj[u]:
                expected = tau[u] * signs[idx]
                if tau[v] == 0:
                    tau[v] = expected
                    q.append(v)
                else:
                    if tau[v] != expected:
                        return True
        return False

    def get_radius(signs):
        A = np.zeros((n, n))
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            A[u, v] = s
            A[v, u] = s
        vals = np.linalg.eigvalsh(A)
        return np.max(np.abs(vals))

    rng = np.random.default_rng(3)
    
    best_signs = None
    best_radius = float('inf')
    
    # Find an initial valid solution
    for _ in range(100):
        signs = rng.choice([-1, 1], size=E).astype(np.int8)
        if is_unbalanced(signs):
            r = get_radius(signs)
            if r < best_radius:
                best_radius = r
                best_signs = signs.copy()
            break

    if best_signs is None:
        # Fallback: construct a known unbalanced sign
        best_signs = np.ones(E, dtype=np.int8)
        best_signs[0] = -1
        best_radius = get_radius(best_signs)

    # Local Search (Hill Climbing)
    max_iters = 2000
    restarts = 10
    
    for _ in range(restarts):
        current_signs = best_signs.copy()
        current_radius = best_radius
        
        for it in range(max_iters):
            idx = rng.integers(0, E)
            old_sign = current_signs[idx]
            current_signs[idx] = -old_sign
            
            if is_unbalanced(current_signs):
                r = get_radius(current_signs)
                if r < current_radius:
                    current_radius = r
                    if r < best_radius:
                        best_radius = r
                        best_signs = current_signs.copy()
                else:
                    current_signs[idx] = old_sign
            else:
                current_signs[idx] = old_sign
                
        # Random jump to escape local minima
        if rng.random() < 0.2:
            for _ in range(50):
                rand_signs = rng.choice([-1, 1], size=E).astype(np.int8)
                if is_unbalanced(rand_signs):
                    rr = get_radius(rand_signs)
                    if rr < best_radius:
                        best_radius = rr
                        best_signs = rand_signs.copy()
                    break

    return best_signs.astype(np.int8)
