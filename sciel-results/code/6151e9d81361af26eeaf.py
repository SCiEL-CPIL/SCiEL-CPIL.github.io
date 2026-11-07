import numpy as np
import time

def heuristics(prizes, weights):
    start_time = time.time()
    
    # prizes: (5, 100)
    # weights: (5, 100)
    
    # Calculate total prize per item
    total_prize = np.sum(prizes, axis=0)
    
    # Calculate total weight per item
    total_weight = np.sum(weights, axis=0)
    
    # Heuristic: Total Prize / Total Weight
    # Reverting to simple value density as the previous penalty reduced performance.
    # This aligns closer to the seed program's effective behavior.
    eta = total_prize / (total_weight + 1e-9)
    
    # Ensure positivity
    eta = np.where(eta > 0, eta, 1e-9)
    
    # Deadline guard
    if time.time() - start_time > 290:
        return eta
        
    return eta
