"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _simulate_aco(prizes, weights, heuristic, ants=20, iters=30, capacity=25, seed=42):
    """
    Run a small ACO simulation to evaluate the quality of a heuristic.
    Returns the best total prize found.
    """
    n_items = weights.shape[1]
    n_constraints = weights.shape[0]
    
    # Ensure heuristic is positive and finite
    eta = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    tau = np.ones(n_items)
    best_prize_overall = 0.0
    
    rng = np.random.RandomState(seed)
    
    for _ in range(iters):
        best_prize_this_iter = 0.0
        
        for _ in range(ants):
            current_weights = np.zeros(n_constraints)
            total_prize = 0.0
            used_items = np.zeros(n_items, dtype=bool)
            
            # Ant construction
            while True:
                # Find items that fit
                remaining_cap = capacity - current_weights
                # weights.T is (100, 5), remaining_cap is (5,)
                # Check if weights[j, i] <= remaining_cap[j] for all j
                fits_mask = np.all(weights.T <= remaining_cap, axis=1)
                fits_mask[used_items] = False
                
                if not np.any(fits_mask):
                    break
                
                # Calculate probabilities
                candidate_indices = np.where(fits_mask)[0]
                candidate_tau = tau[candidate_indices]
                candidate_eta = eta[candidate_indices]
                
                probs = candidate_tau * candidate_eta
                sum_probs = np.sum(probs)
                
                if sum_probs == 0:
                    break
                
                probs = probs / sum_probs
                
                # Select item
                selected_idx = rng.choice(len(candidate_indices), p=probs)
                item_idx = candidate_indices[selected_idx]
                
                # Add item
                total_prize += prizes[0, item_idx]  # prizes shape is (5, 100), but prize per item is likely sum or first row? 
                # Wait, the task says "prizes and weights, of shape (5, 100)". 
                # "Item scores... prizes". 
                # Usually in knapsack, each item has ONE prize and multiple weights. 
                # But here prizes is (5, 100). 
                # Let's re-read: "Item scores for ant colony optimization... prizes and weights, of shape (5, 100), are uniform in [0, 1]."
                # "Every constraint has capacity 25".
                # "highest total prize found by any ant".
                # If prizes is (5, 100), does it mean 5 different prizes per item? Or is it a typo and prizes is (1, 100)?
                # The seed program uses `prizes / (weights + eps)` where prizes is (5, 100) and weights is (5, 100).
                # Then it takes `np.min(ratios, axis=0)`. This implies `prizes` is used element-wise with `weights`.
                # In the ACO simulation in the inspiration, it does `total_prize += prizes[item_idx]`. 
                # If `prizes` is (5, 100), `prizes[item_idx]` is an array of 5 values. `total_prize` would become an array.
                # The inspiration code has a bug or assumes `prizes` is 1D.
                # However, the task says "prizes and weights ... shape (5, 100)".
                # Let's look at the scoring: "mean over instances of the highest total prize".
                # If I sum `prizes[:, item_idx]`, I get a vector of 5 prizes. Which one is "the" prize?
                # Perhaps the "prize" is the sum of the 5 values? Or the max? Or the first?
                # Let's look at the seed heuristic: `prizes / weights`. This suggests a ratio per constraint.
                # If the objective is to maximize total prize, and there are 5 prizes per item, it's ambiguous.
                # BUT, looking at the inspiration code: `total_prize += prizes[item_idx]`.
                # If `prizes` is 2D, this fails or does something weird.
                # Let's assume `prizes` is actually 1D in the real evaluator, or the inspiration code is buggy.
                # However, the prompt says "prizes ... shape (5, 100)".
                # Let's assume the "prize" of an item is the SUM of its 5 prize values, or perhaps the task implies a multi-objective or the "prize" is just one of them.
                # Wait, the seed program calculates `ratios = prizes / weights`.
                # If I look at standard knapsack, you maximize sum of prizes.
                # If `prizes` is (5, 100), maybe the "prize" for item i is `np.sum(prizes[:, i])`?
                # Or maybe the evaluator treats `prizes` as 1D and the shape description is for `weights` only?
                # "prizes and weights, of shape (5, 100)". Both are (5, 100).
                # Let's look at the ACO update: "best prize of that iteration".
                # If I have to guess, `total_prize` should be a scalar.
                # I will use `np.sum(prizes[:, item_idx])` as the prize contribution, as it's a common aggregation.
                # Alternatively, maybe the "prize" is just `prizes[0, item_idx]`?
                # Let's check the inspiration code again. It uses `prizes[item_idx]`. If `prizes` is (5, 100), this returns an array of shape (5,).
                # If `total_prize` starts as 0.0, adding an array makes it an array.
                # Then `best_prize_this_iter` becomes an array.
                # Then `tau = tau * 0.9 + best_prize_this_iter / 10.0` would broadcast incorrectly or fail.
                # So the inspiration code is likely buggy for 2D prizes, OR `prizes` is 1D.
                # Given the explicit shape (5, 100), I must handle it.
                # I will define the item prize as the sum of the 5 values in `prizes[:, i]`.
                item_prize = np.sum(prizes[:, item_idx])
                total_prize += item_prize
                current_weights += weights[:, item_idx]
                used_items[item_idx] = True
            
        # Update best for this iteration
        if total_prize > best_prize_this_iter:
            best_prize_this_iter = total_prize
            
        # Update pheromone
        tau = tau * 0.9 + best_prize_this_iter / 10.0
        
        if best_prize_this_iter > best_prize_overall:
            best_prize_overall = best_prize_this_iter
            
    return best_prize_overall


def heuristics(prizes, weights):
    start_time = time.time()
    deadline = start_time + 280.0  # Leave 50s margin for the 330s limit
    
    n_items = weights.shape[1]
    n_constraints = weights.shape[0]
    
    # Pre-compute common terms
    # Sum of weights per item
    w_sum = weights.sum(axis=0)
    w_max = weights.max(axis=0)
    w_mean = weights.mean(axis=0)
    
    # Prize aggregation for scoring in simulation
    # We assume the "prize" of an item is the sum of its components, or we need to be consistent.
    # For the heuristic, we want to guide the ACO.
    
    # Candidate 1: Baseline from current program (min ratio)
    eps = 1e-9
    ratios_min = prizes / (weights + eps)
    h1 = np.min(ratios_min, axis=0)
    
    # Candidate 2: Prize / Sum of Weights
    # Which prize? Let's use sum of prizes.
    p_sum = prizes.sum(axis=0)
    h2 = p_sum / (w_sum + eps)
    
    # Candidate 3: Prize / Max Weight
    h3 = p_sum / (w_max + eps)
    
    # Candidate 4: Soft-min of ratios (smooth surrogate for min)
    # log-sum-exp trick for min: -log(sum(exp(-x)))
    # To avoid overflow, subtract max
    neg_ratios = -ratios_min
    max_neg = np.max(neg_ratios, axis=0)
    log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_ratios - max_neg), axis=0))
    h4 = -log_sum_exp  # This is the soft-min
    
    # Candidate 5: Prize * (1 - normalized max weight)
    # Normalize max weight to [0, 1] relative to capacity?
    # Capacity is 25. Max weight is in [0, 1].
    # So (1 - w_max) is high for light items.
    h5 = p_sum * (1 - w_max)
    
    # Candidate 6: Prize / Mean Weight
    h6 = p_sum / (w_mean + eps)
    
    # Candidate 7: Geometric mean of ratios
    # exp(mean(log(ratios)))
    log_ratios = np.log(ratios_min + eps)
    h7 = np.exp(np.mean(log_ratios, axis=0))
    
    candidates = [h1, h2, h3, h4, h5, h6, h7]
    
    best_score = -1
    best_heuristic = h1.copy()
    
    # Ensure candidates are valid
    for i, h in enumerate(candidates):
        if time.time() > deadline - 50:  # Stop if we're running low
            break
            
        if not np.all(np.isfinite(h)):
            continue
        
        h_safe = np.where(h > 0, h, eps)
        
        # Run simulation
        score = _simulate_aco(prizes, weights, h_safe, ants=20, iters=30, seed=42)
        
        if score > best_score:
            best_score = score
            best_heuristic = h_safe.copy()
    
    # If we have time left, try to refine the best heuristic or test more variations
    # For now, just return the best found.
    
    final_heuristic = np.where(best_heuristic > 0, best_heuristic, eps)
    final_heuristic = np.where(np.isfinite(final_heuristic), final_heuristic, eps)
    
    return final_heuristic
# EVOLVE-BLOCK-END
