"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multiple-constraint knapsack.
    
    Strategy: Empirical Marginal Utility via Multi-Start Greedy Simulation.
    1. Run multiple fast greedy simulations with different sorting keys to find good feasible subsets.
    2. Count how frequently each item appears in these high-value subsets.
    3. Combine this empirical frequency (utility) with a base prize/weight ratio.
    4. This creates a heuristic that reflects the "true" value of items in the context of the constraints,
       rather than just a static local ratio.
    """
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    if prizes.ndim == 2:
        prizes = prizes.sum(axis=0)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    
    n_items = weights.shape[1]
    n_constraints = weights.shape[0]
    capacity = 25.0
    
    # Deadline guard
    start_time = time.time()
    deadline = 300.0 - 5.0  # Safety margin
    
    # Base score: prize / max_weight
    max_weight = weights.max(axis=0)
    base_score = prizes / (max_weight + 1e-9)
    
    # Frequency counter
    freq = np.zeros(n_items)
    
    # Define several sorting keys for greedy simulation
    # 1. Prize density (prize / max_weight)
    # 2. Prize / min_weight (optimistic)
    # 3. Prize / mean_weight
    # 4. Prize only
    # 5. Inverse max_weight (small items)
    
    min_weight = weights.min(axis=0)
    mean_weight = weights.mean(axis=0)
    
    keys = [
        base_score,
        prizes / (min_weight + 1e-9),
        prizes / (mean_weight + 1e-9),
        prizes,
        1.0 / (max_weight + 1e-9)
    ]
    
    # Number of simulations per key
    # We can afford many fast simulations
    n_sims_per_key = 50
    
    for key in keys:
        # Sort indices by key descending
        sorted_indices = np.argsort(-key)
        
        for _ in range(n_sims_per_key):
            if time.time() > deadline:
                break
                
            remaining_cap = np.full(n_constraints, capacity)
            selected = np.zeros(n_items, dtype=bool)
            
            for idx in sorted_indices:
                # Check if item fits
                if np.all(weights[:, idx] <= remaining_cap):
                    selected[idx] = True
                    remaining_cap -= weights[:, idx]
            
            freq[selected] += 1
            
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break
    
    # Normalize frequency to [0, 1]
    max_freq = np.max(freq) if np.max(freq) > 0 else 1.0
    normalized_freq = freq / max_freq
    
    # Combine base score and empirical frequency
    # The heuristic should guide the ACO to items that are both individually good and part of good sets
    # Use a geometric mean or weighted sum
    # Let's try: score = base_score * (1 + normalized_freq)
    # Or: score = base_score * normalized_freq^0.5
    
    # ACO pheromone starts at 1. Heuristic is multiplied.
    # If we make the heuristic too sharp, the colony might get stuck.
    # If too flat, it explores randomly.
    # The empirical freq captures the "collaborative" value.
    
    # Let's blend: 50% base, 50% empirical
    # But we need to ensure positive values.
    
    # Normalize base_score to [0, 1] for blending
    max_base = np.max(base_score) if np.max(base_score) > 0 else 1.0
    min_base = np.min(base_score)
    range_base = max_base - min_base
    if range_base > 1e-12:
        norm_base = (base_score - min_base) / range_base
    else:
        norm_base = np.ones_like(base_score)
    
    # Blend
    blend_alpha = 0.5
    blended = (1 - blend_alpha) * norm_base + blend_alpha * normalized_freq
    
    # Power amplification to sharpen the top items slightly
    # The ACO mechanism already provides some focus via pheromone,
    # but a sharper heuristic helps early convergence.
    power = 1.5
    amplified = np.power(blended, power)
    
    # Ensure positive
    amplified = np.where(amplified > 1e-9, amplified, 1e-9)
    
    return amplified
# EVOLVE-BLOCK-END
