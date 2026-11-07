import numpy as np

def heuristics(prizes, weights):
    # Per-constraint efficiency: prize / weight for each constraint
    # Avoid division by zero
    safe_weights = np.where(weights < 1e-9, 1e-9, weights)
    efficiency = prizes / safe_weights  # shape (5, 100)
    
    # Harmonic mean of per-constraint efficiencies:
    #   5 / sum(1/efficiency_i)
    # This penalizes items where ANY constraint has a low ratio more heavily
    # than geometric mean, guiding ants away from bottleneck-risky items.
    inv_eff = 1.0 / efficiency  # shape (5, 100)
    harmonic_mean = 5.0 / inv_eff.sum(axis=0)  # shape (100,)
    
    # Total prize per item (reward high-value items)
    total_prize = prizes.sum(axis=0)  # shape (100,)
    
    # Normalize both to [0, 1] range for balanced combination
    hm_norm = (harmonic_mean - harmonic_mean.min()) / (harmonic_mean.max() - harmonic_mean.min() + 1e-9)
    prize_norm = (total_prize - total_prize.min()) / (total_prize.max() - total_prize.min() + 1e-9)
    
    # Heuristic: emphasize harmonic-mean efficiency, add prize as secondary signal
    heuristic = hm_norm + 0.1 * prize_norm
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
