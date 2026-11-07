"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 300s limit, leave 10s margin
    
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Handle potential shape issues, though contract says (5, 100)
    if prizes.ndim == 1:
        prizes = prizes.reshape(1, -1)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
        
    # prizes: (5, 100), weights: (5, 100)
    
    # Calculate the geometric mean of the prize/weight ratios for each item
    # This balances efficiency across all constraints
    eps = 1e-9
    
    # Avoid division by zero and log of zero
    safe_prizes = np.maximum(prizes, eps)
    safe_weights = np.maximum(weights, eps)
    
    # Ratios shape (5, 100)
    ratios = safe_prizes / safe_weights
    
    # Log of ratios shape (5, 100)
    log_ratios = np.log(ratios)
    
    # Mean of log ratios per item shape (100,)
    mean_log_ratios = np.mean(log_ratios, axis=0)
    
    # Geometric mean shape (100,)
    geo_mean_scores = np.exp(mean_log_ratios)
    
    # Handle non-finite values
    geo_mean_scores = np.where(np.isfinite(geo_mean_scores), geo_mean_scores, 0.0)
    
    # Ensure positivity
    geo_mean_scores = np.maximum(geo_mean_scores, eps)
    
    # Optional: Normalize to [0, 1] or keep raw.
    # The ACO probability is proportional to pheromone * heuristic.
    # Normalizing to max 1.0 is a common practice to keep values stable.
    max_score = np.max(geo_mean_scores)
    if max_score > 0:
        heuristic_scores = geo_mean_scores / max_score
    else:
        heuristic_scores = np.ones_like(geo_mean_scores)
        
    # Ensure strictly positive
    heuristic_scores = np.maximum(heuristic_scores, eps)
    
    # Check deadline
    if time.time() > deadline:
        pass # Return whatever we have
        
    return heuristic_scores
# EVOLVE-BLOCK-END
