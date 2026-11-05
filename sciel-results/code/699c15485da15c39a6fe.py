import numpy as np
import time

def priority(adjacency):
    """
    Calculate priority scores for a greedy Maximum Independent Set decoder.
    
    Strategy:
    1. Initialize scores using a "survival probability" heuristic:
       score ~ 1 / (degree + 1 + beta * avg_neighbor_degree)
       This models vertices that are structurally isolated from dense clusters
       as having higher survival probability in a random sequential elimination.
    2. Refine these scores using local search (pair swaps) to directly maximize
       the size of the independent set produced by the greedy decoder.
       - Use a fast simulation function.
       - Implement a more efficient neighborhood search or larger step swaps?
       - Actually, the current local search is just random swaps.
       - Let's try to incorporate the "structured swap" idea from the current program
         but make it more effective by biasing towards vertices that are close in rank.
         Or, better yet, let's look at the inspiration 1 which is simpler but scored 1.1353.
         Why did it score higher? It used a different initial heuristic:
         Score = -deg - 0.5 * avg_neighbor_deg
         And it ran for 10 seconds.
         The current program runs for ~115 seconds.
         The difference might be in the initial heuristic or the local search dynamics.
         
         Let's combine:
         - Use the initial heuristic from Inspiration 1 (which scored higher):
           scores = -deg - 0.5 * avg_neighbor_deg
           Then normalize to [0,1] or just use as is? 
           The local search swaps scores. If scores are large negative numbers, 
           adding noise might be tricky. Normalization helps.
           
         - Use the local search machinery from the current program (more restarts, 
           longer time budget) but ensure the simulation is as fast as possible.
           
         - Key improvement idea: The random swap local search can get stuck. 
           Let's try a "rank-based" swap: pick two vertices that are close in 
           the current ranking. This is a finer-grained move.
           Or, even better, let's try to optimize the simulation speed.
           For N=200, simulation is very fast. The bottleneck is the number of 
           simulations we can do.
           
         Let's stick to the Inspiration 1 heuristic as the starting point, 
         but use the longer time budget and multi-restart strategy of the current program.
         Also, let's refine the local search to include "rank-adjacent" swaps 
         which might be more effective for fine-tuning the order.
    """
    start_time = time.time()
    DEADLINE = 118.0  # Safety margin before 120s limit
    
    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]
    
    if N == 0:
        return np.array([])
    
    # Ensure symmetry and remove self-loops
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    
    # Convert to boolean for faster neighbor lookups
    A_bool = A > 0.5
    
    # Precompute neighbor lists for fast simulation
    neighbors_list = [np.where(A_bool[i])[0] for i in range(N)]
    
    # --- 1. Initial Heuristic: From Inspiration 1 (Scored 1.1353) ---
    deg = A.sum(axis=1)
    
    # Sum of degrees of neighbors
    neighbor_deg_sum = A.dot(deg)
    
    # Average degree of neighbors
    safe_deg = deg + 1e-6
    avg_neighbor_deg = neighbor_deg_sum / safe_deg
    
    # Heuristic: Low degree is good. Being near hubs is bad (penalty).
    # Score = -deg - 0.5 * avg_neighbor_deg
    # We want higher scores to be picked first.
    # So we invert: Score = -(deg + 0.5 * avg_neighbor_deg)
    # Then normalize to [0, 1] to make noise scaling easier.
    
    raw_scores = -(deg + 0.5 * avg_neighbor_deg)
    
    # Normalize to [0, 1]
    if raw_scores.max() > raw_scores.min():
        scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
    else:
        scores = np.ones_like(scores)
        
    def simulate_fast(s):
        """
        Fast greedy simulation using vectorized operations.
        Returns the size of the independent set.
        """
        order = np.argsort(-s, kind='stable')
        active = np.ones(N, dtype=bool)
        count = 0
        
        # Precompute neighbor masks for faster vectorized updates
        # We'll use a bitset approach: for each vertex, have a bitmask of neighbors
        # But for N=200, we can use a 2D boolean array or just loop efficiently
        
        for v in order:
            if active[v]:
                count += 1
                active[v] = False
                nbrs = neighbors_list[v]
                if len(nbrs) > 0:
                    active[nbrs] = False
        return count
    
    def simulate_fast_v2(s):
        """
        Optimized greedy simulation using precomputed neighbor indices.
        """
        order = np.argsort(-s, kind='stable')
        active = np.ones(N, dtype=bool)
        count = 0
        
        for v in order:
            if active[v]:
                count += 1
                active[v] = False
                nbrs = neighbors_list[v]
                if len(nbrs) > 0:
                    active[nbrs] = False
        return count
    
    # --- 2. Baseline ---
    best_val = simulate_fast(scores)
    best_scores = scores.copy()
    
    # --- 3. Local Search ---
    rng = np.random.default_rng(42)
    
    # We will perform multiple restarts.
    # For each restart, we do a local search.
    # To improve on the current program, let's try a mix of:
    # 1. Random swaps (global exploration)
    # 2. Rank-adjacent swaps (local exploitation)
    
    num_restarts = 10  # More restarts, shorter time each?
    # Total time is ~115s. 10 restarts = ~11s each.
    
    for restart in range(num_restarts):
        if time.time() - start_time > DEADLINE:
            break
        
        if restart == 0:
            current_scores = best_scores.copy()
            current_val = best_val
        else:
            # Perturb the best score found so far
            # Add Gaussian noise
            noise_scale = 0.1 + 0.05 * restart
            noise = rng.normal(0, noise_scale, N)
            current_scores = best_scores + noise
            current_val = simulate_fast(current_scores)
            
            # If the perturbed score is significantly worse, it might be in a bad basin.
            # But let's give the local search a chance.
        
        # Local search loop with mixed move types
        no_improve = 0
        max_no_improve = 3000  # Lower to allow more restarts
        
        # Precompute current ranking for rank-adjacent swaps
        current_rank = np.argsort(-current_scores, kind='stable')
        
        while time.time() - start_time < DEADLINE and no_improve < max_no_improve:
            # Choose move type: 70% random, 30% rank-adjacent
            move_type = rng.random()
            
            if move_type < 0.7:
                # Random swap
                i, j = rng.integers(0, N, size=2)
                if i == j:
                    continue
            else:
                # Rank-adjacent swap: pick a random position in the ranking,
                # swap with a neighbor in the ranking (within ±3 positions)
                pos = rng.integers(0, N - 1)
                offset = rng.integers(1, 4)  # 1, 2, or 3
                pos2 = pos + offset if pos + offset < N else pos - offset
                if pos2 < 0:
                    pos2 = pos + 1
                i = current_rank[pos]
                j = current_rank[pos2]
                if i == j:
                    continue
            
            # Swap
            current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
            new_val = simulate_fast(current_scores)
            
            if new_val > current_val:
                current_val = new_val
                no_improve = 0
                if current_val > best_val:
                    best_val = current_val
                    best_scores = current_scores.copy()
                # Update ranking after improvement
                current_rank = np.argsort(-current_scores, kind='stable')
            else:
                # Revert
                current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
                no_improve += 1
        
        # If we haven't improved the global best in this restart, 
        # and we are in a later restart, we might break early to save time 
        # for other restarts? No, we break on time anyway.
        
        # However, if current_val is much worse than best_val, 
        # we might be wasting time. 
        # But the noise restarts are meant to escape local optima.
        
    # Final polish pass: try to improve with more targeted swaps
    if time.time() - start_time < DEADLINE - 5.0:
        polish_start = time.time()
        polish_deadline = DEADLINE - 2.0  # Leave 2s safety margin
        current_scores = best_scores.copy()
        current_val = best_val
        current_rank = np.argsort(-current_scores, kind='stable')
        no_improve = 0
        max_no_improve_polish = 2000
        
        while time.time() - start_time < polish_deadline and no_improve < max_no_improve_polish:
            # In polish, use only rank-adjacent swaps for fine-tuning
            pos = rng.integers(0, N - 1)
            offset = rng.integers(1, 3)  # 1 or 2
            pos2 = pos + offset if pos + offset < N else pos - offset
            if pos2 < 0:
                pos2 = pos + 1
            i = current_rank[pos]
            j = current_rank[pos2]
            if i == j:
                continue
            
            current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
            new_val = simulate_fast(current_scores)
            
            if new_val > current_val:
                current_val = new_val
                no_improve = 0
                if current_val > best_val:
                    best_val = current_val
                    best_scores = current_scores.copy()
                current_rank = np.argsort(-current_scores, kind='stable')
            else:
                current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
                no_improve += 1
    
    return best_scores
