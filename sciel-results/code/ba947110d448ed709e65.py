import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # 1. Compute Degrees
    degrees = adj.sum(axis=1).astype(float)
    
    # 2. Compute Triangle Counts
    # Triangles containing v = 0.5 * (A^3)[v,v]
    # A^3 can be computed via matrix multiplication
    # For N=200, this is very fast.
    A2 = adj @ adj
    # Triangles per vertex
    tri_counts = 0.5 * np.diag(A2 @ adj).astype(float)
    
    # 3. Compute Coreness
    def compute_coreness(adj):
        n = adj.shape[0]
        current_adj = adj.copy()
        current_deg = adj.sum(axis=1).astype(int)
        core = np.zeros(n, dtype=float)
        remaining = list(range(n))
        
        for _ in range(n):
            if not remaining:
                break
            # Find vertex with minimum degree among remaining
            min_v = min(remaining, key=lambda x: current_deg[x])
            min_d = current_deg[min_v]
            core[min_v] = min_d
            
            remaining.remove(min_v)
            # Update neighbors
            neighbors = np.where(current_adj[min_v])[0]
            for u in neighbors:
                current_deg[u] -= 1
                current_adj[min_v, u] = False
                current_adj[u, min_v] = False
        return core

    coreness = compute_coreness(adj)
    
    # 4. Define Heuristics
    # H1: Seed (-degree)
    h1 = -degrees
    
    # H2: -Coreness
    h2 = -coreness
    
    # H3: -degree - triangles
    h3 = -degrees - tri_counts
    
    # H4: -degree + neighbor_degree_sum (penalize high neighbor degrees? or reward?)
    # Let's try: -degree - 0.1 * (sum of neighbor degrees)
    # High neighbor degrees mean v is connected to "important" nodes.
    # If we pick v, we block them.
    # If we don't pick v, they might be picked.
    # This is ambiguous. Let's try a few variations.
    neighbor_deg_sum = adj @ degrees
    h4 = -degrees - 0.1 * neighbor_deg_sum
    
    h5 = -degrees + 0.1 * neighbor_deg_sum
    
    # H6: -Coreness - triangles
    h6 = -coreness - tri_counts
    
    heuristics = [h1, h2, h3, h4, h5, h6]
    
    # 5. Simulate Greedy for each heuristic
    best_score_val = -1
    best_scores = h1
    
    def simulate_greedy(scores):
        # Sort indices by descending score
        order = np.argsort(-scores)
        selected = 0
        blocked = np.zeros(n, dtype=bool)
        
        for v in order:
            if not blocked[v]:
                selected += 1
                blocked[v] = True
                # Block neighbors
                neighbors = np.where(adj[v])[0]
                blocked[neighbors] = True
        return selected

    for h in heuristics:
        if time.time() > deadline:
            break
        val = simulate_greedy(h)
        if val > best_score_val:
            best_score_val = val
            best_scores = h
    
    return best_scores
