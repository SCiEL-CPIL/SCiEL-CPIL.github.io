def sign_edges(graph, budget):
    import numpy as np
    import random
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

    def calc_radius(signs):
        # Check connectivity (unbalanced)
        tau = np.zeros(n, dtype=np.int8)
        tau[0] = 1
        balanced = True
        q = deque([0])
        while q and balanced:
            u = q.popleft()
            for (v, idx) in adj[u]:
                expected = tau[u] * signs[idx]
                if tau[v] == 0:
                    tau[v] = expected
                    q.append(v)
                else:
                    if tau[v] != expected:
                        balanced = False
                        break
        if balanced:
            return float('inf') # Invalid (disconnected)

        # Build signed adjacency
        A = np.zeros((n, n))
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            A[u, v] = s
            A[v, u] = s
        vals = np.linalg.eigvalsh(A)
        return np.max(np.abs(vals))

    best_radius = float('inf')
    best_signs = None
    
    # ILS: Random restarts with local search
    num_restarts = 50
    max_iter = 1000
    
    for r in range(num_restarts):
        # Random initialization
        signs = np.random.choice([-1, 1], size=E, dtype=np.int8)
        current_radius = calc_radius(signs)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_signs = signs.copy()
            
        # Local search: flip one edge at a time
        improved = True
        while improved:
            improved = False
            # Shuffle edge order to avoid bias
            edge_order = list(range(E))
            random.shuffle(edge_order)
            for idx in edge_order:
                old_sign = signs[idx]
                signs[idx] = -old_sign
                new_radius = calc_radius(signs)
                if new_radius < current_radius:
                    current_radius = new_radius
                    improved = True
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_signs = signs.copy()
                else:
                    signs[idx] = old_sign
                    
        # Perturbation for next restart? Or just simple restart.
        # Simple restart is fine for small graphs.

    if best_signs is None:
        # Fallback
        best_signs = np.ones(E, dtype=np.int8)
        best_signs[0] = -1
        
    return best_signs.astype(np.int8)
