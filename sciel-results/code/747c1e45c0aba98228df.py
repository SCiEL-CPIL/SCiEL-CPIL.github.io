import numpy as np
import time

def heuristics(prizes, weights):
    start_time = time.time()
    deadline = start_time + 285  # Safety margin before 330s kill

    # Ensure input is float
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    # prizes: (5, 100), weights: (5, 100)
    # 5 constraints, capacity 25 each

    # Total prize per item (sum across constraints)
    total_prize = np.sum(prizes, axis=0)  # (100,)

    epsilon = 1e-9

    # Safe weights to avoid division by zero
    safe_weights = np.where(weights > epsilon, weights, 1.0)  # (5, 100)

    # Per-constraint ratios: prizes[k, j] / weights[k, j]
    # If weight is 0 (replaced by 1.0), ratio is just prize (small)
    ratios = prizes / safe_weights  # (5, 100)

    # 1. Min-ratio (bottleneck efficiency): min over constraints
    # min(p_k / w_k) for each item
    min_ratio = np.min(ratios, axis=0)  # (100,)

    # 2. Harmonic mean of ratios
    # H_j = 5 / sum(1/r_kj) = 5 / sum(w_k / p_k)
    # If any p_k is 0, 1/r_kj is inf, H_j = 0
    denom = np.where(prizes > epsilon, prizes, 1.0)
    inv_ratios = safe_weights / denom  # (5, 100)
    sum_inv_ratios = np.sum(inv_ratios, axis=0)  # (100,)
    harm_mean = np.where(sum_inv_ratios > epsilon, 5.0 / sum_inv_ratios, 1e6)

    # 3. Geometric slack: reward items with balanced (low-variance) weights
    # Lower std of weights means more "slack" in packing
    weight_std = np.std(weights, axis=0)  # (100,)
    # Penalize high variance: 1 / (1 + std)
    slack_factor = 1.0 / (1.0 + weight_std)

    # 4. Combine: emphasize bottleneck strongly, reward overall efficiency, add slack
    # h = (min_ratio ** a) * (harm_mean ** b) * (slack_factor ** c) * (total_prize ** d)
    
    # Safe versions for power operations
    min_ratio_safe = np.maximum(min_ratio, epsilon)
    harm_mean_safe = np.maximum(harm_mean, epsilon)
    slack_safe = np.maximum(slack_factor, epsilon)
    total_prize_safe = np.maximum(total_prize, epsilon)

    # Exponents: strong bottleneck emphasis, moderate efficiency, mild slack and prize
    a = 1.5  # Strong bottleneck emphasis (from inspiration 1)
    b = 0.5  # Moderate overall efficiency
    c = 0.5  # Moderate slack preference
    d = 0.3  # Mild total prize preference

    heuristic = (np.power(min_ratio_safe, a) * 
                 np.power(harm_mean_safe, b) * 
                 np.power(slack_safe, c) * 
                 np.power(total_prize_safe, d))

    # Safety: replace non-finite or non-positive with epsilon
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, epsilon)

    # Normalize to mean 1 to match pheromone scale
    mean_h = np.mean(heuristic)
    if mean_h > epsilon and np.isfinite(mean_h):
        heuristic = heuristic / mean_h
    else:
        heuristic = np.ones(100, dtype=np.float64)

    # Final safety check
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, epsilon)

    if heuristic.shape != (100,):
        heuristic = heuristic.reshape(100)

    if time.time() > deadline:
        return heuristic

    return heuristic
