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
    3. Use multiple restarts with noise to escape local optima.
    
    The key insight is that the greedy decoder's output is highly sensitive to
    small changes in the priority ordering. Local search allows us to navigate
    this landscape effectively, starting from a structurally informed baseline.
    """
    start_time = time.time()
    DEADLINE = 118.0  # Safety margin before 120s limit (evaluator kills at 150s, but let's be safe)
    
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
    
    # --- 1. Initial Heuristic: Survival Probability Model ---
    # We model the score as an inverse weighted degree.
    # Vertices with low degree and low neighbor degree sum are "safer" to pick.
    # This approximates the probability of a vertex surviving a random sequential
    # elimination process where neighbors block each other.
    
    deg = A.sum(axis=1)
    
    # Sum of degrees of neighbors
    neighbor_deg_sum = A.dot(deg)
    
    # Average degree of neighbors (local clustering/density proxy)
    safe_deg = deg + 1e-6
    avg_neighbor_deg = neighbor_deg_sum / safe_deg
    
    # The "survival score" is inversely proportional to the effective degree.
    # effective_degree = deg + beta * avg_neighbor_deg
    # We want high scores for low effective degree.
    
    # Let's tune beta. A beta of 1.0 balances own degree and neighbor density.
    # We can also add a small constant to avoid division by zero and to shift the scale.
    beta = 1.0
    effective_deg = deg + beta * avg_neighbor_deg + 1.0
    
    # Inverse for score
    scores = 1.0 / effective_deg
    
    # Normalize scores to [0, 1] for better numerical behavior in local search
    # This helps when adding noise later.
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        scores = np.ones_like(scores) # All same degree, any order is equivalent
    
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
                    # Deactivate neighbors
                    # Using a loop is sometimes faster than vectorized assignment for small lists
                    # but for N=200, vectorized is fine.
                    active[nbrs] = False
        return count
    
    # --- 2. Baseline ---
    best_val = simulate_fast(scores)
    best_scores = scores.copy()
    
    # --- 3. Local Search ---
    rng = np.random.default_rng(42)
    
    # Precompute score order for structured swaps
    score_order = np.argsort(-scores)
    
    # Try multiple restarts
    # We have ~115s budget. Simulation is O(N^2) worst case ~ 40k ops.
    # We can do many swaps.
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() - start_time > DEADLINE:
            break
        
        # For restart 0, use original. For others, perturb from best found.
        if restart == 0:
            current_scores = best_scores.copy()
            current_val = best_val
        else:
            # Add noise to escape local optima
            # Scale noise relative to score range
            noise_scale = 0.05 + 0.02 * restart
            noise = rng.normal(0, noise_scale, N)
            current_scores = best_scores + noise
            current_val = simulate_fast(current_scores)
            
            # If the restart is bad, skip to next? Or just let local search fix it?
            # If it's significantly worse, maybe it's in a bad basin.
            # But local search might find its way out. Let's run it.
        
        # Local search loop
        no_improve = 0
        max_no_improve = 1500 + 300 * restart
        
        while time.time() - start_time < DEADLINE and no_improve < max_no_improve:
            # Choose swap candidates
            # Mix of structured (close in rank) and random
            if rng.random() < 0.5:
                base_idx = rng.integers(0, N - 1)
                # Pick a nearby index in the current score order
                # We need to keep score_order updated or just use indices?
                # Using current score_order is better for structured swaps.
                # But score_order changes as we swap.
                # Re-sorting every step is O(N log N). N=200, this is fast.
                # However, we only need to re-sort if we accept a swap?
                # For simplicity and speed, let's just pick random i, j.
                # Structured swaps are less critical if we have enough random swaps.
                # Let's stick to random swaps for speed and simplicity, 
                # but bias towards vertices with similar scores?
                # Actually, random swaps over N=200 are very effective.
                
                i, j = rng.integers(0, N, size=2)
            else:
                # Occasionally try a structured swap based on current rank
                # Get current rank order (expensive if done every time, so only here)
                # Actually, let's just do random. The previous code had structured, 
                # but random is often sufficient for this size.
                i, j = rng.integers(0, N, size=2)
            
            if i == j:
                continue
            
            # Swap scores
            current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
            new_val = simulate_fast(current_scores)
            
            if new_val > current_val:
                current_val = new_val
                no_improve = 0
                if current_val > best_val:
                    best_val = current_val
                    best_scores = current_scores.copy()
            else:
                # Revert swap
                current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
                no_improve += 1
        
        # Early stopping if no improvement in this restart
        if current_val <= best_val and restart > 1:
             # If we haven't beaten the global best in this restart, 
             # and it's a later restart, maybe break?
             # But noise might have been small. Let's just break if we've tried a few.
             break
    
    return best_scores
