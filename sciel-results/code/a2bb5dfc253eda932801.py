"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 300s limit, leave 10s margin
    
    # Ensure inputs are 2D with shape (5, 100)
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    if prizes.ndim == 1:
        prizes = prizes.reshape(5, -1)
    if weights.ndim == 1:
        weights = weights.reshape(5, -1)
    
    n_items = prizes.shape[1]
    
    # Check deadline
    if time.time() > deadline:
        return np.ones(n_items)
    
    # Per-constraint efficiency: prize / weight for each constraint
    # Avoid division by zero
    safe_weights = np.where(weights < 1e-9, 1e-9, weights)
    efficiency = prizes / safe_weights  # shape (5, 100)
    
    # Geometric mean of per-constraint efficiency ratios
    # This balances value against multiplicative cost across all dimensions
    # log(geometric_mean) = mean(log(efficiency))
    # Use log-sum-exp style for numerical stability
    log_efficiency = np.log(efficiency)  # shape (5, 100)
    log_geo_mean = np.mean(log_efficiency, axis=0)  # shape (100,)
    geo_mean_efficiency = np.exp(log_geo_mean)  # shape (100,)
    
    # Worst-case (minimum) constraint efficiency per item
    min_efficiency = efficiency.min(axis=0)  # shape (100,)
    
    # Total prize per item
    total_prize = prizes.sum(axis=0)  # shape (100,)
    
    # Normalize both to [0, 1] range for balanced combination
    geo_mean_norm = (geo_mean_efficiency - geo_mean_efficiency.min()) / (geo_mean_efficiency.max() - geo_mean_efficiency.min() + 1e-9)
    min_eff_norm = (min_efficiency - min_efficiency.min()) / (min_efficiency.max() - min_efficiency.min() + 1e-9)
    prize_norm = (total_prize - total_prize.min()) / (total_prize.max() - total_prize.min() + 1e-9)
    
    # Combine: primary signal is geometric mean efficiency (smooth surrogate for min),
    # secondary is worst-case efficiency, tertiary is total prize
    # Geometric mean naturally penalizes items with high variance in resource consumption
    heuristic = 0.5 * geo_mean_norm + 0.4 * min_eff_norm + 0.1 * prize_norm
    
    # Ensure all values are positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Check deadline
    if time.time() > deadline:
        return heuristic
    
    return heuristic
# EVOLVE-BLOCK-END
