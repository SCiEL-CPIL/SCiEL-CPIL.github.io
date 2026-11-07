import numpy as np
import time

def heuristics(prizes, weights):
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Handle potential shape issues
    if prizes.ndim == 1:
        prizes = prizes.reshape(1, -1)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    
    # Ensure both have the same number of rows
    # If one is (1, 100) and other is (5, 100), expand the single constraint
    if prizes.shape[0] == 1 and weights.shape[0] > 1:
        prizes = np.repeat(prizes, weights.shape[0], axis=0)
    elif weights.shape[0] == 1 and prizes.shape[0] > 1:
        weights = np.repeat(weights, prizes.shape[0], axis=0)
    
    # Now both should have the same shape
    num_constraints, num_items = prizes.shape
    
    # Add small epsilon to avoid division by zero
    eps = 1e-9
    
    # For each constraint j and item i, compute the efficiency ratio:
    # prize[i, j] / weight[i, j]
    # But we need to be careful: prizes and weights are both (constraints, items)
    # So prize[i, j] is the prize of item i under constraint j? 
    # Actually, looking at the shape: prizes is (5, 100) where 5 is constraints and 100 is items.
    # This is a bit ambiguous. Typically in MCKP, each item has a prize (scalar) and 
    # weights for each constraint. But here prizes is (5, 100).
    # Let's assume prizes[i, j] is the prize associated with item j under some dimension i,
    # or perhaps it's a multi-objective prize.
    # However, the scoring says "highest total prize", implying a scalar prize per item.
    # Let's re-read: "prizes and weights, of shape (5, 100)".
    # If prizes is (5, 100), maybe each item has 5 prize components?
    # And the "total prize" is the sum? Or maybe the first row is the prize?
    # Given the ambiguity, and the previous heuristic using sum of normalized prizes,
    # let's assume the "total prize" for an item is the sum across constraints (rows).
    # But the strategy suggests geometric mean of per-constraint efficiency ratios (prize_i / weight_ij).
    # If prize_i is a scalar per item, it should be (1, 100) or (100,).
    # If prize is (5, 100), then for constraint j, item i has prize p[j, i] and weight w[j, i].
    # The efficiency for constraint j, item i is p[j, i] / w[j, i].
    # The geometric mean across constraints j for item i is prod_j (p[j, i] / w[j, i])^(1/J).
    
    # Let's compute the ratio for each constraint and item
    # Avoid division by zero
    safe_weights = np.where(weights < eps, eps, weights)
    ratios = prizes / safe_weights  # shape (5, 100)
    
    # Geometric mean across constraints (axis=0)
    # Geometric mean = exp(mean(log(x)))
    # Ensure ratios are positive
    ratios = np.maximum(ratios, eps)
    log_ratios = np.log(ratios)
    mean_log_ratio = np.mean(log_ratios, axis=0)  # shape (100,)
    geo_mean_efficiency = np.exp(mean_log_ratio)  # shape (100,)
    
    # The heuristic is the geometric mean efficiency
    # Ensure it's positive
    heuristic = np.maximum(geo_mean_efficiency, eps)
    
    return heuristic
