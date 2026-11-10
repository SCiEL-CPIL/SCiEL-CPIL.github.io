"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multiple-constraint knapsack.
    
    Strategy: Greedy-Set Membership Heuristic (Feasibility-Aware Simulation).
    
    1. Problem Framing:
       The standard ACO heuristic (prize/weight) ignores that items compete for 
       capacity across 5 constraints. An item with a great ratio might be "unfit" 
       because it consumes capacity in a constraint that is already tight for 
       other high-value items. The Lagrangian dual approach approximates this 
       with continuous prices, but misses the discrete "blocking" effect.
       
    2. Method:
       - We perform a fast, deterministic simulation of the greedy packing process 
         that the ACO ants will roughly follow (since pheromones start uniform and 
         heuristics dominate early).
       - We run multiple "greedy passes" with slightly different sorting criteria 
         (e.g., prize/weight, prize/max_weight, random tie-breaking) to estimate 
         the frequency of each item appearing in a valid, high-value pack.
       - Items that frequently appear in these valid packs are "safe" and "valuable". 
         Items that rarely appear (because they block others or don't fit) are "risky".
       
    3. Heuristic Construction:
       - Score_i = (Frequency_i)^alpha * (Prize_i)^beta
       - This ensures that only items that are both valuable AND feasible in the 
         context of the whole set get high scores.
       - This guides the ACO to avoid "greedy traps" where picking a locally optimal 
         item prevents a globally optimal set.
    """
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Handle shapes: prizes (5, 100) -> (100,), weights (5, 100)
    if prizes.ndim == 2:
        prizes = prizes.sum(axis=0)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    
    n_items = weights.shape[1]
    n_constraints = weights.shape[0]
    capacity = 25.0
    
    # Time budget guard
    start_time = time.time()
    time_limit = 10.0  # Greedy simulation is very fast
    
    # Initialize frequency counter
    freq = np.zeros(n_items)
    
    # Number of greedy simulations to run
    # 100 simulations is very fast (100 items * 100 sims * 5 constraints = 50k ops)
    n_sims = 100
    
    # Precompute some metrics for sorting
    # 1. Total prize per item (already have prizes)
    # 2. Max weight per item (across constraints) - proxy for "size"
    max_w = weights.max(axis=0)
    # 3. Min weight per item - proxy for "efficiency" in tight constraints
    min_w = weights.min(axis=0)
    
    # Define a few sorting keys to diversify the greedy selections
    # Key 1: Prize / Max Weight (standard efficiency)
    # Key 2: Prize / Min Weight (optimistic efficiency)
    # Key 3: Prize (pure value)
    # Key 4: Randomized tie-breaking on Prize/Max Weight
    
    # To keep it fast, we'll just run a few variations of the greedy algorithm.
    # Variation A: Sort by Prize / (1 + Max_W)
    # Variation B: Sort by Prize / (1 + Min_W)
    # Variation C: Sort by Prize
    # Variation D: Randomized perturbations of Prize / (1 + Max_W)
    
    # We'll do 25 sims of each type to get a robust estimate.
    
    def run_greedy(sort_key, random_seed=None):
        """
        Runs a single greedy packing simulation.
        Returns a binary mask of items selected.
        """
        # Create a copy of remaining capacities
        rem_cap = np.full(n_constraints, capacity)
        selected = np.zeros(n_items, dtype=bool)
        
        # Determine order
        if random_seed is not None:
            # Add small noise to break ties and explore
            noise = np.random.default_rng(random_seed).uniform(0, 0.1, n_items)
            keys = sort_key + noise
        else:
            keys = sort_key
            
        # Sort indices by key descending
        order = np.argsort(-keys)
        
        # Iterate through items in sorted order
        for i in order:
            if selected[i]:
                continue
            # Check if item i fits in all constraints
            # weights[:, i] has shape (n_constraints,)
            if np.all(weights[:, i] <= rem_cap):
                # Select it
                selected[i] = True
                rem_cap -= weights[:, i]
            else:
                # Skip it
                pass
                
        return selected

    # Key 1: Prize / (1 + Max_W)
    key1 = prizes / (1.0 + max_w)
    # Key 2: Prize / (1 + Min_W)
    key2 = prizes / (1.0 + min_w)
    # Key 3: Prize
    key3 = prizes
    
    # Run simulations
    # 25 sims of Key1, 25 of Key2, 25 of Key3, 25 Randomized Key1
    for s in range(25):
        mask = run_greedy(key1, random_seed=None)
        freq += mask
        
    for s in range(25):
        mask = run_greedy(key2, random_seed=None)
        freq += mask
        
    for s in range(25):
        mask = run_greedy(key3, random_seed=None)
        freq += mask
        
    for s in range(25):
        mask = run_greedy(key1, random_seed=s)
        freq += mask

    # Check time
    if time.time() - start_time > time_limit:
        # If we ran out of time (unlikely), just use what we have
        pass

    # Normalize frequency to [0, 1]
    max_freq = np.max(freq)
    if max_freq > 0:
        freq_norm = freq / max_freq
    else:
        freq_norm = np.ones(n_items)
        
    # Combine with prize
    # We want to reward items that are frequently selected AND have high prize.
    # Simple product: Score = freq_norm * prize
    # To sharpen, we can apply a power to the frequency.
    # Items that appear in 80% of sims are much better than those in 40%.
    # Let's use freq_norm^2 to emphasize consistency.
    
    scores = (freq_norm ** 2.0) * prizes
    
    # Ensure positive
    # If an item is never selected (freq=0), score is 0.
    # Map non-positive to 1e-9.
    scores_final = np.where(scores > 0, scores, 1e-9)
    
    # Ensure finite
    scores_final = np.where(np.isfinite(scores_final), scores_final, 1e-9)
    
    return scores_final
# EVOLVE-BLOCK-END
