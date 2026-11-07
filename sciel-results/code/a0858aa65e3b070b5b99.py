import numpy as np

def heuristics(prizes, weights):
    """
    Compute heuristic values for ant colony optimization using Geometric Mean
    of per-constraint value-to-weight ratios.
    
    This structural change moves away from the Arithmetic Mean (seed) and
    Max-Weight Denominator (current failed attempt) to a multiplicative
    aggregation that penalizes inefficiency in any single constraint,
    which is more appropriate for multiple-constraint knapsack problems.
    """
    # Ensure weights are positive to avoid division by zero
    # Using a small epsilon prevents inf/nan
    epsilon = 1e-9
    safe_weights = np.where(weights > epsilon, weights, epsilon)
    
    # Calculate per-constraint efficiency ratios: prize / weight
    ratios = prizes / safe_weights
    
    # Calculate Geometric Mean: exp(mean(log(ratios)))
    # This is robust to outliers compared to arithmetic mean and
    # captures the "balanced" efficiency across all 5 constraints.
    log_ratios = np.log(ratios)
    mean_log_ratio = np.mean(log_ratios, axis=0)
    
    heuristic = np.exp(mean_log_ratio)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
