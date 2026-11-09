"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _run_aco_simulation(prizes, weights, heuristic, n_ants=20, n_iterations=30, capacity=25, seed=42):
    """
    Run a simplified ACO simulation to evaluate how well a heuristic performs.
    Returns the best total prize found.
    """
    num_constraints, n_items = prizes.shape
    rng = np.random.default_rng(seed)
    
    # Pheromone starts at 1
    pheromone = np.ones(n_items, dtype=np.float64)
    
    # Ensure heuristic is positive
    heuristic_safe = np.maximum(heuristic, 1e-9)
    
    best_overall = 0.0
    
    for iteration in range(n_iterations):
        iteration_best = 0.0
        
        for ant in range(n_ants):
            # Track remaining capacity per constraint
            remaining = np.full(num_constraints, capacity, dtype=np.float64)
            total_prize = 0.0
            selected = np.zeros(n_items, dtype=bool)
            
            while True:
                # Find items that still fit in all constraints
                # An item fits if weights[c, item] <= remaining[c] for all c
                fits = np.all(weights <= remaining[:, None], axis=0)
                fits[selected] = False  # Can't select already-selected items
                
                if not np.any(fits):
                    break
                
                # Compute selection probabilities
                # probability proportional to pheromone[item] * heuristic[item]
                probs = pheromone * heuristic_safe
                probs[~fits] = 0.0
                
                prob_sum = probs.sum()
                if prob_sum <= 0:
                    break
                
                probs = probs / prob_sum
                
                # Select an item
                item_idx = rng.choice(n_items, p=probs)
                
                # Add it
                selected[item_idx] = True
                total_prize += prizes[:, item_idx].sum()
                remaining -= weights[:, item_idx]
            
        # After each ant, check if this ant's solution is better
        if total_prize > iteration_best:
            iteration_best = total_prize
        
        if total_prize > best_overall:
            best_overall = total_prize
        
        # Pheromone update: multiply by 0.9, add (best prize of iteration) / 10
        pheromone = pheromone * 0.9 + iteration_best / 10.0
    
    return best_overall


def _generate_candidate_heuristics(prizes, weights):
    """
    Generate multiple diverse candidate heuristics.
    Returns a list of heuristic arrays.
    """
    num_constraints, n_items = prizes.shape
    total_prize = prizes.sum(axis=0)
    sum_weights = weights.sum(axis=0)
    max_weights = weights.max(axis=0)
    min_weights = weights.min(axis=0)
    eps = 1e-9
    
    candidates = []
    
    # Candidate 1: Prize / sum_weights
    h1 = total_prize / (sum_weights + eps)
    candidates.append(h1)
    
    # Candidate 2: Prize / max_weights (binding constraint)
    h2 = total_prize / (max_weights + eps)
    candidates.append(h2)
    
    # Candidate 3: Prize / min_weights (most generous constraint)
    h3 = total_prize / (min_weights + eps)
    candidates.append(h3)
    
    # Candidate 4: Pareto dominance filter on prize/sum_weights
    # Count how many items dominate each item
    dominator_count = np.zeros(n_items, dtype=np.int32)
    for i in range(n_items):
        prize_i = total_prize[i]
        w_i = weights[:, i]
        prize_ok = total_prize <= prize_i
        weight_ok = np.all(w_i <= weights.T, axis=1)
        candidates_idx = np.where(prize_ok & weight_ok)[0]
        for j in candidates_idx:
            if i != j:
                if total_prize[i] > total_prize[j] or np.any(w_i < weights[:, j]):
                    dominator_count[j] += 1
    
    h4 = total_prize / (sum_weights + eps) * (1.0 / (1.0 + 3.0 * dominator_count))
    candidates.append(h4)
    
    # Candidate 5: Pareto dominance filter on prize/max_weights
    h5 = total_prize / (max_weights + eps) * (1.0 / (1.0 + 3.0 * dominator_count))
    candidates.append(h5)
    
    # Candidate 6: Normalized prize (higher prize = higher attractiveness)
    prize_min = total_prize.min()
    prize_max = total_prize.max()
    prize_range = prize_max - prize_min + eps
    h6 = (total_prize - prize_min) / prize_range + 0.1
    candidates.append(h6)
    
    # Candidate 7: Inverse of normalized sum_weights (lighter = better)
    weight_min = sum_weights.min()
    weight_max = sum_weights.max()
    weight_range = weight_max - weight_min + eps
    h7 = 1.0 - (sum_weights - weight_min) / weight_range + 0.1
    candidates.append(h7)
    
    # Candidate 8: Combined: prize_ratio * lightness
    h8 = h1 * h7
    candidates.append(h8)
    
    # Candidate 9: Prize per unit of most constraining resource, with Pareto boost
    # For each item, find which constraint is most binding (highest weight relative to capacity)
    # Since capacity is 25 for all, the most binding is the max weight
    # Use prize / max_weight as base
    h9 = total_prize / (max_weights + eps)
    # Boost items that are Pareto optimal in the (prize, -max_weight) space
    pareto_boost = np.ones(n_items)
    for i in range(n_items):
        for j in range(n_items):
            if i != j:
                if total_prize[j] >= total_prize[i] and max_weights[j] <= max_weights[i]:
                    if total_prize[j] > total_prize[i] or max_weights[j] < max_weights[i]:
                        pareto_boost[i] *= 0.5
                        break
    h9 = h9 * pareto_boost
    candidates.append(h9)
    
    # Candidate 10: Geometric mean of prize/sum and prize/max
    h10 = np.sqrt(h1 * h2)
    candidates.append(h10)
    
    # Ensure all candidates are positive and finite
    cleaned_candidates = []
    for h in candidates:
        h = np.asarray(h, dtype=np.float64)
        h = np.maximum(h, 1e-9)
        h = np.where(np.isfinite(h), h, 1e-9)
        cleaned_candidates.append(h)
    
    return cleaned_candidates


def _compute_heuristics(prizes, weights, deadline=None):
    """
    Compute heuristic attractiveness scores by simulating ACO with multiple candidates
    and selecting the best-performing one.
    """
    start_time = time.time()
    
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    if prizes.ndim != 2 or weights.ndim != 2:
        raise ValueError("prizes and weights must be 2D arrays")
    
    if prizes.shape[0] != weights.shape[0]:
        raise ValueError("prizes and weights must have same number of constraints")
    
    if prizes.shape[1] != weights.shape[1]:
        raise ValueError("prizes and weights must have same number of items")
    
    num_constraints, n_items = prizes.shape
    
    if not np.all(np.isfinite(prizes)) or not np.all(np.isfinite(weights)):
        raise ValueError("prizes and weights must contain only finite values")
    
    # Generate candidate heuristics
    candidates = _generate_candidate_heuristics(prizes, weights)
    
    # Check time budget - if we have plenty of time, evaluate all candidates
    elapsed = time.time() - start_time
    time_per_candidate = 0.05  # Rough estimate for a fast ACO sim
    
    # If we have enough time, evaluate all; otherwise, use a subset
    max_candidates_to_test = min(len(candidates), max(1, int((290 - elapsed) / time_per_candidate)))
    
    if max_candidates_to_test < len(candidates):
        # Prioritize diverse candidates
        indices = list(range(len(candidates)))
        # Shuffle with a fixed seed for reproducibility
        rng = np.random.default_rng(12345)
        rng.shuffle(indices)
        indices = indices[:max_candidates_to_test]
        candidates_to_test = [candidates[i] for i in indices]
    else:
        candidates_to_test = candidates
    
    # Evaluate each candidate
    best_score = -1.0
    best_heuristic = candidates[0]  # Fallback
    
    for idx, heuristic in enumerate(candidates_to_test):
        # Check time
        if time.time() > (start_time + 280):
            break
        
        # Run ACO simulation with this heuristic
        score = _run_aco_simulation(prizes, weights, heuristic, n_ants=20, n_iterations=30, capacity=25, seed=42 + idx)
        
        if score > best_score:
            best_score = score
            best_heuristic = heuristic
    
    # Ensure positive and finite
    best_heuristic = np.maximum(best_heuristic, 1e-9)
    best_heuristic = np.where(np.isfinite(best_heuristic), best_heuristic, 1e-9)
    
    return best_heuristic


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in ant colony optimization
    of the multiple-constraint knapsack problem.
    """
    start_time = time.time()
    
    try:
        result = _compute_heuristics(prizes, weights, start_time + 290)
        
        result = np.asarray(result, dtype=np.float64)
        
        if result.ndim != 1:
            raise ValueError("Heuristic must be a 1D array")
        
        if not np.all(np.isfinite(result)):
            raise ValueError("Heuristic must contain only finite values")
        
        prizes_arr = np.asarray(prizes)
        expected_n_items = prizes_arr.shape[1] if prizes_arr.ndim == 2 else len(prizes_arr)
        
        if result.shape[0] != expected_n_items:
            raise ValueError(f"Heuristic shape {result.shape} doesn't match expected ({expected_n_items},)")
        
        return result
        
    except Exception:
        prizes_arr = np.asarray(prizes, dtype=np.float64)
        weights_arr = np.asarray(weights, dtype=np.float64)
        
        if prizes_arr.ndim == 2:
            n_items = prizes_arr.shape[1]
        else:
            n_items = len(prizes_arr)
            
        return np.ones(n_items, dtype=np.float64)
# EVOLVE-BLOCK-END
