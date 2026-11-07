import numpy as np
import time

def heuristics(prizes, weights):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 330s kill

    # Ensure input is float
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    # prizes: (5, 100), weights: (5, 100)
    # 5 constraints, capacity 25 each

    epsilon = 1e-9

    # Safe weights to avoid division by zero
    # If weight is 0, we treat the ratio as very high if prize > 0, or 0 if prize == 0.
    # The logic below handles this via the ratios calculation.
    safe_weights = np.where(weights > epsilon, weights, 1.0)  # (5, 100)

    # Per-constraint ratios: prizes[k, j] / weights[k, j]
    # If weight was 0, safe_weight is 1.0.
    # If prize > 0 and weight == 0, ratio = prize. This is finite. 
    # Ideally, infinite efficiency, but capping to prize value is a reasonable proxy 
    # given the [0,1] range. If we used 1e-9 for weight, ratio would be huge.
    # Let's stick to the safe_weights approach but ensure it doesn't distort too much.
    # Actually, if weight is 0, the item is "free" in that dimension.
    # A high ratio is good.
    ratios = prizes / safe_weights  # (5, 100)

    # 1. Soft-Min-Ratio (Bottleneck) via log-sum-exp
    # soft_min(x, t) = -t * log(sum(exp(-x_i/t)))
    # As t -> 0, this approaches min(x)
    # Use a small temperature for close approximation to min, but smooth enough for ACO
    t_soft = 0.1
    neg_ratios = -ratios / t_soft
    # Clip to avoid overflow in exp. 
    # If ratios are in [0, 1] (approx), neg_ratios are in [-10, 0].
    # If weight is small, ratio can be large. e.g. 1/1e-9 = 1e9. 
    # neg_ratio = -1e10. exp(-1e10) is 0. This is fine.
    # The risk is if ratio is very small (0), neg_ratio is 0, exp(0)=1.
    # The risk is overflow if ratio is negative? No, prizes/weights are non-negative.
    # So neg_ratios are non-positive. exp(neg_ratios) <= 1. No overflow risk in exp.
    # However, if we had negative ratios, we'd need clipping. Here inputs are [0,1].
    neg_ratios = np.clip(neg_ratios, -100, 100)  # Just in case of numerical weirdness
    log_sum_exp = np.log(np.sum(np.exp(neg_ratios), axis=0))
    M = -t_soft * log_sum_exp  # (100,)

    # 2. Harmonic Mean of ratios for each item
    # H_j = 5 / sum(1/r_kj) where 1/r_kj = w_kj / p_kj
    # We need to be careful if p_kj is 0.
    # If p_kj == 0, 1/r_kj is inf. Sum is inf. H is 0. This is correct: 0 prize in any dimension makes it useless for that dimension's efficiency.
    denom = np.where(prizes > epsilon, prizes, 1.0)
    inv_ratios = safe_weights / denom  # (5, 100)
    sum_inv_ratios = np.sum(inv_ratios, axis=0)  # (100,)
    H = np.where(sum_inv_ratios > epsilon, 5.0 / sum_inv_ratios, 1e6)

    # 3. Geometric Blending: H^1.5 * M^0.5
    # This weighting (1.5/0.5) was found to be superior in inspirations.
    # It emphasizes the overall harmonic efficiency more than the bottleneck,
    # but the bottleneck still has significant influence.
    H_safe = np.maximum(H, epsilon)
    M_safe = np.maximum(M, epsilon)

    alpha = 1.5
    beta = 0.5

    heuristic = np.power(H_safe, alpha) * np.power(M_safe, beta)

    # Safety: replace non-finite or non-positive with epsilon
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, epsilon)

    # Normalize to mean 1 to match pheromone scale
    # This ensures the heuristic is on a comparable scale to the initial pheromone (which is 1)
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
