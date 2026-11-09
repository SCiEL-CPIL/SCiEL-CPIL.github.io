import numpy as np

def heuristics(prizes, weights):
    # Calculate the total prize for each item
    # Assuming the objective is to maximize the sum of all prize components
    item_prizes = prizes.sum(axis=0)
    
    # Calculate the average weight for each constraint to determine tightness
    # Constraints with higher average weights are tighter
    avg_weights = weights.mean(axis=1)
    
    # Calculate a weighted sum of weights for each item
    # This penalizes items that are heavy in tight constraints more
    weighted_weights = weights @ avg_weights
    
    # Calculate the heuristic: Prize / Weighted Weight
    # This favors items with high prize and low weight in tight constraints
    heuristic = item_prizes / (weighted_weights + 1e-9)
    
    # Normalize the heuristic to have a mean of 1
    # This ensures a balanced interaction with the pheromone trail (which starts at 1)
    mean_heuristic = heuristic.mean()
    if mean_heuristic > 0:
        heuristic = heuristic / mean_heuristic
    else:
        heuristic = np.ones(100)
        
    # Ensure all values are positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
