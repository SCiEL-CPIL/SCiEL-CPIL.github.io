import numpy as np

def heuristics(prizes, weights):
    # Per-constraint efficiency: prize / weight for each constraint
    # Avoid division by zero
    safe_weights = np.where(weights < 1e-9, 1e-9, weights)
    efficiency = prizes / safe_weights  # shape (5, 100)
    
    # Smooth surrogate for the worst-case (min) constraint efficiency.
    # soft-min via log-sum-exp:  -log(sum(exp(-x)))  approximates min(x)
    # with a temperature that controls smoothness.
    # Use a moderate temperature: small enough to approximate min,
    # large enough to remain smooth and avoid the kink of hard min.
    beta = 10.0  # inverse temperature; larger -> closer to hard min
    neg_eff = -beta * efficiency  # shape (5, 100)
    # Stable log-sum-exp along axis 0
    m = neg_eff.max(axis=0, keepdims=True)
    log_sum = np.log(np.sum(np.exp(neg_eff - m), axis=0)) + m[0]
    soft_min_efficiency = -log_sum / beta  # shape (100,)
    
    # Total prize per item (reward high-value items)
    total_prize = prizes.sum(axis=0)  # shape (100,)
    
    # Combine: primary signal is soft worst-case efficiency, secondary is total prize
    # Normalize both to [0, 1] range for balanced combination
    soft_min_norm = (soft_min_efficiency - soft_min_efficiency.min()) / (soft_min_efficiency.max() - soft_min_efficiency.min() + 1e-9)
    prize_norm = (total_prize - total_prize.min()) / (total_prize.max() - total_prize.min() + 1e-9)
    
    # Heuristic: emphasize soft worst-case efficiency, add prize as tiebreaker
    heuristic = soft_min_norm + 0.1 * prize_norm
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
