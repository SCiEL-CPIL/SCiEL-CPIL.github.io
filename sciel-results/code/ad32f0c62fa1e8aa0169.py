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
         Or, even better, let's look at the inspiration 1 which is simpler but scored 1.1353.
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
        Fast greedy simulation.
        Returns the size of the independent set.
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
        
        # Local search loop
        # We'll use a time-sliced approach: run local search until time is up for this restart.
        # Or use a fixed number of iterations? Time is safer.
        
        # To make local search more effective, let's alternate between 
        # random swaps and rank-adjacent swaps.
        
        no_improve = 0
        max_no_improve = 5000  # Reset on improvement
        
        # Pre-compute current order for rank-adjacent swaps?
        # It changes, so we'd need to update it. 
        # Let's just use random swaps for simplicity first, as in Inspiration 1.
        # But Inspiration 1 only ran for 10s. We have 115s.
        # Maybe the issue is that random swaps are inefficient?
        
        # Let's try a "greedy" local search:
        # Pick a random vertex i.
        # Try swapping it with every other vertex j.
        # Keep the best swap.
        # This is O(N^2) per step, which is 40,000 simulations.
        # Each simulation is ~200 ops. 8M ops per step.
        # 115s / 8M ops is not enough for many steps.
        # So random swaps (O(1) per step) are necessary.
        
        # Let's stick to random swaps but with a twist:
        # If we have no improvement for a while, increase the "step size" or change strategy.
        
        while time.time() - start_time < DEADLINE and no_improve < max_no_improve:
            i, j = rng.integers(0, N, size=2)
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
            else:
                # Revert
                current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
                no_improve += 1
                
                # If stuck, maybe do a larger perturbation?
                # Or just continue.
        
        # If we haven't improved the global best in this restart, 
        # and we are in a later restart, we might break early to save time 
        # for other restarts? No, we break on time anyway.
        
        # However, if current_val is much worse than best_val, 
        # we might be wasting time. 
        # But the noise restarts are meant to escape local optima.
        
    return best_scores
