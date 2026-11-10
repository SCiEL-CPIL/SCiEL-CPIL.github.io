"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multiple-constraint knapsack.
    
    Strategy: Monte Carlo Greedy Survival with Lagrangian Guidance.
    
    1. Lagrangian Dual: Compute shadow prices to estimate the 'profitability'
       of each item (Reduced Cost).
    2. Monte Carlo Greedy Simulation: Simulate the packing process multiple
       times. In each simulation, items are selected probabilistically based
       on their reduced cost. This captures the 'blocking' effect: items that
       frequently appear in high-value feasible solutions are deemed more
       attractive.
    3. Combination: The final heuristic is the geometric mean of the normalized
       reduced cost and the normalized Monte Carlo contribution score. This
       ensures items are both theoretically profitable and practically useful.
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
    
    start_time = time.time()
    time_limit = 290.0  # Safety margin

    # --- Part 1: Lagrangian Dual Approximation ---
    # Subgradient method to find lambda
    lambda_c = np.zeros(n_constraints)
    best_lambda = np.zeros(n_constraints)
    best_rc = np.zeros(n_items)
    best_prize = -1.0
    
    # Initialize step size
    step_size = 1.0
    n_lag_iters = 50
    
    for it in range(n_lag_iters):
        if time.time() - start_time > time_limit:
            break
            
        # Calculate reduced costs
        shadow_cost = lambda_c @ weights
        rc = prizes - shadow_cost
        
        # Greedy selection based on RC to evaluate lambda
        order = np.argsort(-rc)
        rem_cap = np.full(n_constraints, capacity)
        selected = np.zeros(n_items, dtype=bool)
        total_prize = 0.0
        
        for i in order:
            if rc[i] <= 0:
                break
            if np.all(weights[:, i] <= rem_cap):
                selected[i] = True
                rem_cap -= weights[:, i]
                total_prize += prizes[i]
        
        if total_prize > best_prize:
            best_prize = total_prize
            best_rc = rc.copy()
            best_lambda = lambda_c.copy()
        
        # Update lambda
        used_cap = weights @ selected
        slack = capacity - used_cap
        # Simple decay for step size
        step_size = 1.0 / (1.0 + 0.1 * it)
        lambda_c = np.maximum(0, lambda_c + step_size * slack)

    # Ensure we have a valid RC vector
    if best_prize == -1.0:
        best_rc = prizes.copy()
    
    # Normalize RC to [0, 1]
    rc_min = best_rc.min()
    rc_max = best_rc.max()
    if rc_max > rc_min:
        rc_norm = (best_rc - rc_min) / (rc_max - rc_min)
    else:
        rc_norm = np.ones(n_items)
    rc_norm = np.maximum(rc_norm, 0.0)

    # --- Part 2: Monte Carlo Greedy Simulation ---
    # Simulate packing process to identify items that lead to high-value solutions
    mc_scores = np.zeros(n_items, dtype=float)
    n_mc_runs = 100  # Number of simulations
    
    # Pre-compute which items are "potentially selectable" (weight < capacity in all constraints)
    # If an item's weight in any constraint exceeds capacity, it can never be selected.
    feasible_mask = np.all(weights < capacity, axis=0)
    
    if not np.any(feasible_mask):
        # No items can fit, return uniform scores
        return np.ones(n_items) / n_items

    # List of feasible item indices
    feasible_indices = np.where(feasible_mask)[0]
    
    # We will run MC simulations. 
    # To speed up, we can pre-sort items by RC once, but the probabilistic selection
    # requires dynamic probabilities. 
    # Since N=100, a simple loop is fine.
    
    for _ in range(n_mc_runs):
        if time.time() - start_time > time_limit:
            break
            
        rem_cap = np.full(n_constraints, capacity)
        current_rc = best_rc.copy() # Use RC as attractiveness
        selected_in_run = []
        total_prize_run = 0.0
        available_items = feasible_indices.copy()
        
        while len(available_items) > 0:
            # Check which available items fit
            fitting_items = []
            fitting_rcs = []
            
            for idx in available_items:
                if np.all(weights[:, idx] <= rem_cap):
                    fitting_items.append(idx)
                    fitting_rcs.append(current_rc[idx])
            
            if not fitting_items:
                break
                
            fitting_items = np.array(fitting_items)
            fitting_rcs = np.array(fitting_rcs)
            
            # Probabilistic selection based on RC
            # Clip RC to positive to avoid negative probabilities
            probs = np.maximum(fitting_rcs, 0.0)
            prob_sum = probs.sum()
            
            if prob_sum == 0:
                # If all RC are 0 or negative, pick randomly among fitting items
                idx_to_pick = fitting_items[np.random.randint(len(fitting_items))]
            else:
                probs = probs / prob_sum
                idx_to_pick = np.random.choice(fitting_items, p=probs)
            
            # Update state
            rem_cap -= weights[:, idx_to_pick]
            total_prize_run += prizes[idx_to_pick]
            selected_in_run.append(idx_to_pick)
            
            # Remove picked item from available list
            available_items = available_items[available_items != idx_to_pick]
            
        # Update MC scores: Add the total prize of the solution to each selected item's score
        if selected_in_run:
            # Distribute the total prize equally? Or just add the total prize?
            # Adding the total prize rewards items that are part of *good* solutions.
            # If an item is in a bad solution, it gets less reward.
            # This effectively weights the "frequency" of selection by the "quality" of the solution.
            for idx in selected_in_run:
                mc_scores[idx] += total_prize_run

    # Normalize MC scores
    mc_max = mc_scores.max()
    if mc_max > 0:
        mc_norm = mc_scores / mc_max
    else:
        mc_norm = np.ones(n_items)
    mc_norm = np.maximum(mc_norm, 0.0)

    # --- Part 3: Combine Signals ---
    # Geometric Mean: Requires both signals to be high
    # Add epsilon to avoid log(0) or 0^0 issues
    epsilon = 1e-9
    combined = np.sqrt((rc_norm + epsilon) * (mc_norm + epsilon))
    
    # Normalize combined score to [0, 1]
    comb_max = combined.max()
    if comb_max > 0:
        scores = combined / comb_max
    else:
        scores = np.ones(n_items)
        
    # Ensure positivity
    scores = np.maximum(scores, 1e-9)
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
