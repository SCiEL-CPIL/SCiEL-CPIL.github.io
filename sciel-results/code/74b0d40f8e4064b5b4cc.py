import numpy as np
import time

def sign_edges(graph, budget):
    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n = graph['n']
    
    u_arr = np.array([e[0] for e in edges])
    v_arr = np.array([e[1] for e in edges])
    
    # Pre-calculate edge indices for top, bottom, matching
    top_edges = []
    bottom_edges = []
    match_edges = []
    
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_edges.append(i)
        elif u >= m and v >= m:
            bottom_edges.append(i)
        else:
            match_edges.append(i)
            
    # Sort edges within cycles to ensure consistent cycle product calculation
    # Top cycle: (0,1), (1,2), ..., (m-1, 0)
    # Bottom cycle: (m, m+1), ...
    # We need to order them cyclically for product calculation if we were doing that, 
    # but for BFS validity check, order doesn't matter.
    
    def fast_valid(signs):
        # Check if the lift is connected.
        # Lift is disconnected iff there exists y: V -> {0,1} such that y[u] xor y[v] == 0 if sign==1, 1 if sign==-1.
        # This is a 2-coloring constraint with twists.
        # Equivalent to: Product of signs on every cycle is +1.
        # If any cycle has product -1, it's connected.
        # We can check satisfiability of y[u] xor y[v] = sigma[u,v] where sigma is 0 for +1, 1 for -1.
        
        sigma = (1 - signs) // 2 # 0 for +1, 1 for -1
        
        parent = list(range(n))
        rank = [0] * n
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y, val):
            px, py = find(x), find(y)
            if px == py:
                # Check consistency
                # We need a way to store the "offset" or potential.
                # Standard Union-Find for weighted graphs or just BFS is easier.
                # Let's use BFS for simplicity and speed given small N.
                pass
        
        # BFS approach
        # color[v] is the relative sign/parity from 0 to v.
        # color[u] xor color[v] should equal sigma[u,v]
        color = [-1] * n
        color[0] = 0
        queue = [0]
        
        # Adjacency list for base graph
        # We need to map edge index to (u, v, sigma)
        # Since we have u_arr, v_arr, we can just iterate edges in BFS?
        # Better to build adjacency list once.
        
        # Re-build adjacency for speed
        adj = [[] for _ in range(n)]
        for i in range(n_edges):
            u, v = u_arr[i], v_arr[i]
            s = sigma[i]
            adj[u].append((v, s))
            adj[v].append((u, s))
            
        for u in queue:
            for v, s in adj[u]:
                expected_color = color[u] ^ s
                if color[v] == -1:
                    color[v] = expected_color
                    queue.append(v)
                else:
                    if color[v] != expected_color:
                        return True # Contradiction found -> System Inconsistent -> Connected Lift
        
        return False # No contradiction -> System Consistent -> Disconnected Lift

    def spectral_radius(signs):
        A = np.zeros((n, n))
        for i in range(n_edges):
            s = signs[i]
            A[u_arr[i], v_arr[i]] = s
            A[v_arr[i], u_arr[i]] = s
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))

    best_signs = None
    best_radius = float('inf')
    
    start_time = time.time()
    time_limit = 55 # Leave 5s for overhead
    
    rng = np.random.default_rng(42)
    
    # Strategy: Multiple restarts of Greedy Hill Climbing
    # Since N is small, we can do many restarts.
    
    def hill_climb(start_signs):
        current_signs = start_signs.copy()
        if not fast_valid(current_signs):
            # Try to fix validity by flipping edges randomly
            for _ in range(100):
                idx = rng.integers(0, n_edges)
                current_signs[idx] = -current_signs[idx]
                if fast_valid(current_signs):
                    break
            if not fast_valid(current_signs):
                return None, float('inf')
                
        current_radius = spectral_radius(current_signs)
        
        improved = True
        while improved:
            improved = False
            # Try flipping each edge
            for i in range(n_edges):
                if time.time() - start_time > time_limit:
                    return current_signs, current_radius
                original_sign = current_signs[i]
                current_signs[i] = -original_sign
                if fast_valid(current_signs):
                    new_radius = spectral_radius(current_signs)
                    if new_radius < current_radius:
                        current_radius = new_radius
                        improved = True
                    else:
                        current_signs[i] = original_sign
                else:
                    current_signs[i] = original_sign
        return current_signs, current_radius

    num_restarts = 200
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Generate random start
        signs = rng.choice([-1, 1], size=n_edges)
        
        # Optional: Bias towards specific patterns?
        # For now, pure random is robust.
        
        res_signs, res_radius = hill_climb(signs)
        
        if res_signs is not None and res_radius < best_radius:
            best_radius = res_radius
            best_signs = res_signs.copy()
            
    # Fallback if no valid signs found (unlikely)
    if best_signs is None:
        # Try all -1, all +1, etc.
        for pattern in [np.ones(n_edges), -np.ones(n_edges)]:
            if fast_valid(pattern):
                r = spectral_radius(pattern)
                if r < best_radius:
                    best_radius = r
                    best_signs = pattern.copy()
                    
    if best_signs is None:
        best_signs = np.ones(n_edges, dtype=int)

    return best_signs
