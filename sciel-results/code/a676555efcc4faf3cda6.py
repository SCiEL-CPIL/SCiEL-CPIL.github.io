"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def sign_edges(graph, budget):
    import numpy as np
    import time
    from collections import deque

    start_time = time.time()
    DEADLINE = 85  # Safety margin before 90s kill

    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n_vertices = 2 * m

    # Identify edge types
    # Top cycle: (i, (i+1)%m) for i in 0..m-1
    # Bottom cycle: (m+i, m+(i+1)%m) for i in 0..m-1
    # Matching: (i, m+i) for i in 0..m-1
    
    # We need to map edge index to type and position
    # But the input 'edges' is an ordered list. We assume standard ordering or we parse it.
    # The problem says: "graph = {'n': 2m, 'edges': ordered list of (u, v)}: two m-cycles... joined by matching"
    # It doesn't strictly guarantee the order in the list, but usually it's:
    # 0..m-1: Top cycle
    # m..2m-1: Bottom cycle
    # 2m..3m-1: Matching
    # Let's parse robustly.
    
    edge_info = []
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            # Top cycle
            if v == (u + 1) % m:
                edge_info.append(('top', u, idx))
            else:
                # Could be (u+1)%m == u? No, m>=4.
                # Handle reversed
                if u == (v + 1) % m:
                    edge_info.append(('top', v, idx))
                else:
                    edge_info.append(('top', u, idx)) # Fallback
        elif u >= m and v >= m:
            # Bottom cycle
            u_local = u - m
            v_local = v - m
            if v_local == (u_local + 1) % m:
                edge_info.append(('bottom', u_local, idx))
            else:
                if u_local == (v_local + 1) % m:
                    edge_info.append(('bottom', v_local, idx))
                else:
                    edge_info.append(('bottom', u_local, idx))
        else:
            # Matching
            # u < m, v >= m
            # v = u + m
            edge_info.append(('match', u, idx))

    # Initial signs: Analytic pattern
    # Goal: Minimize spectral radius of A_s, ensure unbalanced.
    # Strategy: 
    # If m is even: Match signs alternate +1, -1, +1, -1...
    # If m is odd: Match signs all +1 except one -1.
    # Top and Bottom all +1.
    
    initial_signs = np.ones(n_edges, dtype=int)
    
    match_indices = [info[2] for info in edge_info if info[0] == 'match']
    match_positions = [info[1] for info in edge_info if info[0] == 'match']
    
    if m % 2 == 0:
        for i, idx in enumerate(match_indices):
            if i % 2 == 1:
                initial_signs[idx] = -1
    else:
        # Set the last match edge to -1
        if match_indices:
            initial_signs[match_indices[-1]] = -1

    def is_balanced(signs):
        """
        Check if the signed graph is balanced.
        Balanced iff we can assign vertex signs sigma_v in {1, -1} such that
        signs[edge(u,v)] == sigma_u * sigma_v.
        This is equivalent to checking if the lift is disconnected.
        We do a BFS on the signed graph.
        """
        # Build adjacency list
        adj = [[] for _ in range(n_vertices)]
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            adj[u].append((v, s))
            adj[v].append((u, s))
            
        sigma = [0] * n_vertices
        for i in range(n_vertices):
            if sigma[i] == 0:
                sigma[i] = 1
                queue = deque([i])
                while queue:
                    u = queue.popleft()
                    for v, s in adj[u]:
                        expected_sigma_v = sigma[u] * s
                        if sigma[v] == 0:
                            sigma[v] = expected_sigma_v
                            queue.append(v)
                        elif sigma[v] != expected_sigma_v:
                            return False
        return True

    def spectral_radius(signs):
        """
        Compute spectral radius of the signed adjacency matrix.
        """
        A = np.zeros((n_vertices, n_vertices))
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            A[u, v] = s
            A[v, u] = s
        # eigvalsh for symmetric matrix
        try:
            vals = np.linalg.eigvalsh(A)
            return np.max(np.abs(vals))
        except:
            return float('inf')

    def evaluate(signs):
        if is_balanced(signs):
            return float('inf')
        return spectral_radius(signs)

    # Initial evaluation
    best_signs = initial_signs.copy()
    best_radius = evaluate(best_signs)
    
    if best_radius == float('inf'):
        # Fallback: flip one more edge to break balance
        # Try flipping each match edge
        for idx in match_indices:
            trial = best_signs.copy()
            trial[idx] *= -1
            r = evaluate(trial)
            if r < best_radius:
                best_signs = trial
                best_radius = r
                break
        if best_radius == float('inf'):
            # Try random
            rng = np.random.default_rng(3)
            for _ in range(1000):
                trial = rng.choice([-1, 1], size=n_edges)
                r = evaluate(trial)
                if r < best_radius:
                    best_signs = trial
                    best_radius = r
                    break

    # Local Search
    # We will iterate: for each edge, try flipping. If improvement, accept.
    # To allow escaping local minima, we can add simulated annealing later if time permits.
    # But greedy hill climbing from a good start is often sufficient for these small graphs.
    
    improved = True
    while improved and time.time() < DEADLINE - 5:
        improved = False
        for idx in range(n_edges):
            if time.time() > DEADLINE - 5:
                break
            trial = best_signs.copy()
            trial[idx] *= -1
            r = evaluate(trial)
            if r < best_radius:
                best_signs = trial
                best_radius = r
                improved = True

    # If time remains, do a few random restarts with local search
    if time.time() < DEADLINE - 10:
        rng = np.random.default_rng(42)
        for _ in range(5):
            if time.time() > DEADLINE - 10:
                break
            trial_signs = rng.choice([-1, 1], size=n_edges)
            # Local search from this random start
            curr_radius = evaluate(trial_signs)
            if curr_radius == float('inf'):
                continue
            for _ in range(50): # limited iterations
                if time.time() > DEADLINE - 10:
                    break
                improved = False
                for idx in range(n_edges):
                    t = trial_signs.copy()
                    t[idx] *= -1
                    r = evaluate(t)
                    if r < curr_radius:
                        trial_signs = t
                        curr_radius = r
                        improved = True
                        break
                if not improved:
                    break
            if curr_radius < best_radius:
                best_signs = trial_signs
                best_radius = curr_radius

    return best_signs
# EVOLVE-BLOCK-END
