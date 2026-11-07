import numpy as np
import time

def heuristics(prizes, weights):
    """
    Compute attractiveness scores for ant colony optimization of MCKP.
    """
    start_time = time.time()
    deadline = start_time + 290  # Safety margin

    # Input validation
    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        # Try to reshape if possible, otherwise return dummy
        try:
            prizes = np.asarray(prizes).reshape(5, 100)
            weights = np.asarray(weights).reshape(5, 100)
        except:
            return np.ones(100) * 1e-9

    epsilon = 1e-9
    capacity = 25.0

    # 1. Per-constraint efficiency ratios
    # r[k, j] = prize[k, j] / weight[k, j]
    safe_weights = weights + epsilon
    ratios = prizes / safe_weights  # Shape (5, 100)

    # 2. Weighted Harmonic Mean across constraints
    # Weights are based on the average weight of each constraint.
    # Constraints with higher average weight are "tighter" on average and should
    # have a stronger influence on the heuristic.
    avg_weights = np.mean(weights, axis=1)  # Shape (5,)
    W = avg_weights + epsilon  # Avoid zero division
    
    # Weighted Harmonic Mean: H_w = sum(W) / sum(W / r)
    # For each item j: H_w[j] = sum_k(W_k) / sum_k(W_k / r_kj)
    inv_ratios_weighted = W[:, np.newaxis] / ratios  # Shape (5, 100)
    sum_inv_weighted = np.sum(inv_ratios_weighted, axis=0)  # Shape (100,)
    sum_W = np.sum(W)
    
    harmonic_w = sum_W / (sum_inv_weighted + epsilon)

    # 3. Bottleneck Factor
    # Penalize items that have a very low ratio in any single constraint.
    # Using min ratio with a power < 1 softens the penalty slightly compared to min itself,
    # but still emphasizes the worst constraint.
    min_ratio = np.min(ratios, axis=0)
    bottleneck_factor = np.power(min_ratio + epsilon, 0.5)

    # 4. Slack Factor
    # Reward items that leave more capacity unused.
    # Geometric mean of (1 - w/C) across constraints.
    slack_factors = 1.0 - weights / capacity
    slack_factors = np.clip(slack_factors, epsilon, 1.0)
    # Geometric mean = exp(mean(log(x)))
    log_slack = np.log(slack_factors)
    geometric_slack = np.exp(np.mean(log_slack, axis=0))
    
    # Apply a mild power to slack to not dominate completely
    slack_factor = np.power(geometric_slack, 0.3)

    # 5. Combine Components
    # Base is the Weighted Harmonic Mean
    # Multiplied by Bottleneck and Slack factors
    heuristic = harmonic_w * bottleneck_factor * slack_factor

    # 6. Amplify Differences
    # Power 1.5 sharpens the selection, making ants prefer the top items more strongly.
    heuristic = np.power(heuristic + epsilon, 1.5)

    # 7. Ensure Positivity and Finiteness
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, epsilon)

    # Check deadline
    if time.time() > deadline:
        return heuristic

    # Verify shape
    if heuristic.shape != (100,):
        heuristic = heuristic.reshape(100)

    return heuristic
