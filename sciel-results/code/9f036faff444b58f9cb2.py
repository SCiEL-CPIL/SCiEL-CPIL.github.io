"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    start_time = time.time()
    deadline = 300.0 - 5.0  # Safety margin

    # Convert to numpy arrays for consistent behavior
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # Handle potential 2D prizes by reducing to 1D if necessary
    # The task says "one attractiveness score per item", implying 1D output.
    # The seed `prizes / ...` scores 1.0, suggesting `prizes` is likely 1D (100,)
    # or the evaluator is lenient. To be robust:
    if prizes.ndim == 2:
        # If prizes is (5, 100), reduce to 1D.
        # Summing is a reasonable aggregation for "total prize" if it were a vector.
        # However, given the seed works with direct division, and the contract
        # says "one score per item", 1D is the expected input.
        # If it is 2D, summing axis=0 is a safe fallback.
        prize_vector = np.sum(prizes, axis=0)
    else:
        prize_vector = prizes.flatten()

    # Ensure weights is (5, 100)
    # weights.shape[0] should be 5 (constraints), weights.shape[1] should be 100 (items)
    
    # Calculate weights sum per item (axis=0 sums over constraints)
    # This is the baseline heuristic denominator that scored 1.0
    weight_sums = np.sum(weights, axis=0)
    
    # Calculate max weight per item (bottleneck)
    weight_maxs = np.max(weights, axis=0)
    
    # The baseline uses `prizes / (weight_sums + 1e-9)` and scores 1.0.
    # The previous attempt used `prizes / (weight_maxs + 1e-9)` and scored ~0.994.
    # This suggests that penalizing total weight (sum) is slightly better than
    # penalizing only the bottleneck weight (max) for these random instances.
    # 
    # Why? 
    # In multi-constraint knapsack, an item consumes resources from ALL constraints.
    # Even if one constraint is the bottleneck, consuming from others still reduces
    # feasibility for subsequent items. The sum captures the total "bulk" of the item.
    # 
    # Can we improve on the sum?
    # Maybe a weighted combination?
    # Or perhaps normalizing the weights first?
    # 
    # Let's try a hybrid that emphasizes the sum but also considers the max.
    # If an item is very heavy in one dimension, it's risky.
    # If an item is moderately heavy in all dimensions, it's also risky.
    # 
    # Let's try: cost = weight_sums + alpha * weight_maxs
    # If alpha=0, it's the baseline.
    # If alpha=1, it's sum + max.
    # 
    # Alternatively, we can look at the variance of weights.
    # But let's keep it simple and close to the baseline to avoid regression.
    # 
    # The baseline is `prizes / (weight_sums + 1e-9)`.
    # Let's try to slightly refine the denominator.
    # What if we use the mean? `weight_sums / 5`.
    # `prizes / (weight_sums/5 + 1e-9)` is proportional to `5 * prizes / (weight_sums + 5e-9)`.
    # This is effectively the same as the baseline for relative probabilities.
    # 
    # Let's try adding a small penalty for high variance?
    # No, let's stick to the most robust simple heuristic.
    # 
    # One potential improvement: The ACO pheromone update uses `best prize / 10`.
    # The heuristic is static.
    # 
    # Let's try a slight modification: use the geometric mean of weights?
    # `geometric_mean = np.exp(np.mean(np.log(weights + 1e-9), axis=0))`
    # This penalizes items where ANY weight is very low? No, if one weight is 0, GM is 0.
    # If one weight is high, GM is pulled up, but less than sum.
    # 
    # Let's try: `cost = weight_sums` (baseline)
    # vs `cost = weight_maxs` (worse)
    # vs `cost = np.linalg.norm(weights, axis=0)` (Euclidean)
    # 
    # Let's test the Euclidean norm idea.
    # `norm = np.linalg.norm(weights, axis=0)`
    # 
    # Or perhaps a weighted sum where constraints with less remaining capacity are weighted more?
    # But capacities are fixed at 25 and not passed in. We don't know the remaining capacity
    # at heuristic calculation time (it's computed once before the colony runs).
    # 
    # So the heuristic must be static based on item properties.
    # 
    # Let's try a combination: `cost = 0.8 * weight_sums + 0.2 * weight_maxs * 5`?
    # Since max <= sum <= 5*max (for 5 constraints).
    # Actually, sum is between max and 5*max.
    # 
    # Let's try to just use the baseline `weight_sums` but ensure numerical stability.
    # The baseline scores 1.0. Any change risks regression.
    # 
    # Is there a way to boost the best items more?
    # `heuristic = prize_vector / (weight_sums + 1e-9)`
    # `heuristic = heuristic ** 1.1`? (Amplify differences)
    # If we raise to a power > 1, high ratios become even higher relative to low ones.
    # This might help the ACO focus on the best items.
    # 
    # Let's try `heuristic = (prize_vector / (weight_sums + 1e-9)) ** 1.2`
    # 
    # Or maybe `heuristic = prize_vector / (weight_sums + 1e-9) * (1 + prize_vector)`?
    # 
    # Let's try the power approach. It's a common way to sharpen heuristics.
    
    cost = weight_sums + 1e-9
    heuristic = prize_vector / cost
    
    # Amplify the heuristic to focus on high-value items
    # This makes the probability distribution steeper
    heuristic = heuristic ** 1.1
    
    # Ensure positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Check time
    if time.time() > deadline:
        return heuristic

    return heuristic

# EVOLVE-BLOCK-END
