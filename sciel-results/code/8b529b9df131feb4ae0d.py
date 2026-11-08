"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit
    
    # Convert to float matrix for efficient matrix multiplication
    adj = np.asarray(adjacency, dtype=np.float64)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    adj_bool = adj > 0.5
    degrees = adj.sum(axis=1)
    
    # Precompute neighbor indices for each vertex for fast access
    neighbor_indices = [np.nonzero(adj_bool[i])[0] for i in range(n)]
    
    def greedy_size_from_order(order):
        """Compute greedy MIS size given a priority order (array of vertex indices)."""
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                count += 1
                selected[v] = True
                if len(neighbor_indices[v]) > 0:
                    selected[neighbor_indices[v]] = True
        return count

    def order_from_scores(s):
        """Get vertex order by descending score."""
        # Add small random noise to break ties deterministically-ish but consistently within this call
        # Actually, for evaluation, we just need the order. 
        # Using argsort on -scores. 
        return np.argsort(-s, kind='stable')

    # Generate candidate score vectors from different structural heuristics
    candidates = []
    
    # 1. Low Degree First (Classic Greedy)
    scores_deg = -degrees.copy()
    candidates.append(scores_deg)
    
    # 2. PageRank-like Centrality (High centrality first)
    # Damping factor 0.85
    try:
        if n > 1:
            # Avoid division by zero
            deg_safe = np.where(degrees == 0, 1, degrees)
            M = adj / deg_safe[:, np.newaxis]
            # Add uniform component
            M = 0.15 / n + 0.85 * M
            # Power iteration
            p = np.ones(n) / n
            for _ in range(100):
                new_p = M.T @ p
                # Normalize
                new_p = new_p / np.sum(new_p)
                if np.allclose(p, new_p):
                    break
                p = new_p
            scores_pr = p.copy()
            candidates.append(scores_pr)
    except Exception:
        pass

    # 3. Coreness (K-core peeling order)
    # Vertices with higher core number are "deeper" in the structure.
    # Usually, peeling order (low core first) is good for MIS? 
    # Or high core first? 
    # Let's try both: Low Core First and High Core First.
    try:
        deg_current = degrees.copy()
        core_numbers = np.zeros(n, dtype=int)
        in_queue = np.ones(n, dtype=bool)
        q = list(range(n))
        
        # Simplified K-core approximation using degree peeling
        # This is O(N^2) worst case but N=200 is small.
        remaining = set(range(n))
        deg_rem = degrees.copy()
        
        while remaining:
            # Find min degree in remaining
            min_deg = min(deg_rem[v] for v in remaining)
            # Remove all with this degree
            to_remove = [v for v in remaining if deg_rem[v] == min_deg]
            for v in to_remove:
                core_numbers[v] = min_deg
                remaining.remove(v)
                for u in neighbor_indices[v]:
                    if u in remaining:
                        deg_rem[u] -= 1
        
        # Heuristic A: Low Core First (similar to low degree)
        scores_core_low = -core_numbers.astype(float)
        candidates.append(scores_core_low)
        
        # Heuristic B: High Core First (center of graph)
        scores_core_high = core_numbers.astype(float)
        candidates.append(scores_core_high)
    except Exception:
        pass

    # 4. Betweenness-like / Local Independence
    # Score based on: 1 / (1 + sum of degrees of neighbors)
    # This penalizes vertices connected to high-degree hubs.
    neighbor_deg_sum = np.array([np.sum(degrees[neighbor_indices[i]]) if len(neighbor_indices[i])>0 else 0 for i in range(n)])
    scores_indep = 1.0 / (1.0 + neighbor_deg_sum)
    candidates.append(scores_indep)
    
    # 5. Combined: (Degree + Core)
    if len(candidates) > 3:
        scores_comb = candidates[0] + candidates[3] # Low deg + Low core
        candidates.append(scores_comb)

    # Evaluate each candidate
    best_size = -1
    best_scores = candidates[0]
    
    for scores in candidates:
        if time.time() > deadline:
            break
        order = order_from_scores(scores)
        size = greedy_size_from_order(order)
        if size > best_size:
            best_size = size
            best_scores = scores
            
    # If we have time left, do a quick local improvement on the best candidate's order
    # This is a "polish" step, structurally different from full SA.
    if time.time() < deadline - 10:
        current_order = order_from_scores(best_scores)
        current_size = greedy_size_from_order(current_order)
        
        # Simple hill climbing with restarts for a few seconds
        max_iters = 2000
        iters = 0
        while iters < max_iters and time.time() < deadline - 5:
            # Random swap
            i, j = random.sample(range(n), 2)
            current_order[i], current_order[j] = current_order[j], current_order[i]
            new_size = greedy_size_from_order(current_order)
            
            if new_size > current_size:
                current_size = new_size
                # Update best scores to reflect this order
                best_scores = np.zeros(n)
                for rank, v in enumerate(current_order):
                    best_scores[v] = n - rank
            else:
                # Revert
                current_order[i], current_order[j] = current_order[j], current_order[i]
            
            iters += 1
            
    return best_scores.astype(float)

# EVOLVE-BLOCK-END
