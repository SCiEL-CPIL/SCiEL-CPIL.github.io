import numpy as np

def heuristics(prizes, weights):
    # Per-constraint efficiency: prize / weight for each constraint
    # Avoid division by zero
    safe_weights = np.where(weights < 1e-9, 1e-9, weights)
    efficiency = prizes / safe_weights  # shape (5, 100)
    
    # Worst-case (minimum) constraint efficiency per item
    min_efficiency = efficiency.min(axis=0)  # shape (100,)
    
    # Total prize per item (reward high-value items)
    total_prize = prizes.sum(axis=0)  # shape (100,)
    
    # Combine: primary signal is worst-case efficiency, secondary is total prize
    # Normalize both to [0, 1] range for balanced combination
    min_eff_norm = (min_efficiency - min_efficiency.min()) / (min_efficiency.max() - min_efficiency.min() + 1e-9)
    prize_norm = (total_prize - total_prize.min()) / (total_prize.max() - total_prize.min() + 1e-9)
    
    # Heuristic: emphasize worst-case efficiency, add prize as tiebreaker
    heuristic = min_eff_norm + 0.1 * prize_norm
    
    # Ensure all values are positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
