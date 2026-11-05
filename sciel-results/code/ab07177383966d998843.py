"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

DEADLINE_MARGIN = 2.0

def partition(adjacency):
    start_time = time.time()
    deadline = start_time + 120.0 - DEADLINE_MARGIN
    
    adj = np.asarray(adjacency, dtype=np.float64)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)
    
    # Precompute degrees for fast gain calculation
    # degrees[v] = number of neighbors
    degrees = adj.sum(axis=1)
    
    # Helper to calculate cut size
    # Using the property: cut_size = 0.5 * (sum(degrees) - sum(2*same_side_edges))
    # Or more simply: cut = 0.5 * sum_{u,v} adj[u,v] * (1 - (2*label[u]-1)*(2*label[v]-1))
    # Let's use a vectorized approach:
    # Let s = 2*labels - 1 (vector of +1/-1)
    # s_adj_s = s @ adj @ s
    # cut = 0.5 * (sum(adj) - s_adj_s)
    # Wait, sum(adj) is 2 * E.
    # s_adj_s = sum_{u,v} s_u adj_uv s_v = sum_{u,v in same} adj_uv - sum_{u,v in diff} adj_uv
    # Let S_same = sum_{u,v in same} adj_uv, S_diff = sum_{u,v in diff} adj_uv
    # cut = 0.5 * S_diff
    # Total edges E = 0.5 * (S_same + S_diff) = 0.5 * sum(adj)
    # s_adj_s = S_same - S_diff
    # S_diff = sum(adj) - s_adj_s ? No.
    # S_same + S_diff = sum(adj)
    # S_same - S_diff = s_adj_s
    # 2 * S_diff = sum(adj) - s_adj_s
    # cut = 0.5 * S_diff = 0.25 * (sum(adj) - s_adj_s)
    
    total_edges_sum = adj.sum() # This is 2 * E
    
    def calc_cut(labels):
        s = 2.0 * labels - 1.0
        # s @ adj @ s
        # Since adj is symmetric, s @ adj gives vector, then dot with s
        s_adj = s @ adj
        s_adj_s = np.dot(s, s_adj)
        return 0.25 * (total_edges_sum - s_adj_s)

    best_labels = None
    best_cut = -1
    
    # Simulated Annealing parameters
    # Start temperature high enough to accept bad moves, cool down exponentially
    # We will run multiple restarts.
    
    # Estimate a good initial temperature
    # Max possible gain per flip is roughly degree.
    # Let's use a temperature that allows significant exploration initially.
    T0 = 10.0
    cooling_rate = 0.99
    
    # Main loop: run SA instances until deadline
    while time.time() < deadline:
        # Random initialization
        # Sometimes alternating is good, sometimes random. Let's stick to random for diversity.
        labels = np.random.randint(0, 2, n).astype(np.float64)
        
        current_cut = calc_cut(labels)
        
        # If this random start is already better than our global best, update
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
            
        T = T0
        improved_this_run = False
        
        # Inner SA loop
        while time.time() < deadline:
            # Pick a random vertex to flip
            v = np.random.randint(n)
            
            # Calculate gain of flipping v
            # Current label l = labels[v]
            # Neighbors with label l: same_side
            # Neighbors with label 1-l: other_side
            # Gain = same_side - other_side
            # If Gain > 0, cut increases.
            
            # Efficient gain calc:
            # same_side = adj[v, labels == labels[v]].sum()
            # other_side = degrees[v] - same_side
            # Gain = same_side - (degrees[v] - same_side) = 2 * same_side - degrees[v]
            
            l = labels[v]
            # Vectorized sum of neighbors with same label
            # This is O(n) per flip.
            # We can precompute row sums? No, labels change.
            # For n=100, O(n) is 100 ops. Very fast.
            same_side = adj[v, labels == l].sum()
            gain = 2.0 * same_side - degrees[v]
            
            # Acceptance probability
            if gain > 0:
                accept = True
            else:
                # Accept with probability exp(-gain / T)
                # gain is negative here
                prob = np.exp(gain / T)
                accept = (np.random.rand() < prob)
            
            if accept:
                labels[v] = 1.0 - labels[v]
                current_cut += gain
                improved_this_run = True
                
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
            
            # Cool down
            T *= cooling_rate
            
            # If T becomes very small, we are in local search mode.
            # To ensure we explore different basins, we break out of this SA run
            # if we haven't improved the global best for a while or if T is very low.
            # Actually, the outer loop handles restarts.
            # Let's limit the inner loop iterations per restart to ensure diversity.
            # Or just let it run until T is tiny.
            if T < 1e-6:
                break
                
            # Check time inside inner loop to avoid overshooting deadline significantly
            if time.time() > deadline:
                break
            
    # If no best found (shouldn't happen), return random
    if best_labels is None:
        return np.random.randint(0, 2, n).astype(int)
        
    return best_labels.astype(int)
# EVOLVE-BLOCK-END
