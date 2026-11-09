import numpy as np
import time

# Global deadline guard
_start_time = None

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _start_time
    
    # Initialize timer on first call
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: stop if we've used more than 115 seconds (leave 5s buffer)
    if _get_elapsed() > 115:
        return -1
    
    # Convert to numpy arrays for speed
    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    c = float(remaining_capacity)
    
    n = len(w)
    if n == 0:
        return -1
    
    # Filter feasible items
    mask = w <= c
    if not np.any(mask):
        return -1
    
    # Get feasible indices and sub-arrays
    feasible_idx = np.where(mask)[0]
    w_f = w[feasible_idx]
    v_f = v[feasible_idx]
    
    if len(w_f) == 0:
        return -1
    
    # Calculate various heuristics for feasible items
    # 1. Density: v / w
    with np.errstate(divide='ignore', invalid='ignore'):
        density = v_f / w_f
        # Handle zero weight (shouldn't happen but safe)
        density[w_f == 0] = np.inf if v_f[w_f == 0] > 0 else 0
        
    # 2. Value
    value = v_f.copy()
    
    # 3. Inverse weight
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_weight = 1.0 / w_f
        inv_weight[w_f == 0] = np.inf
    
    # 4. Regularized density: v / (w + alpha * c)
    # Choose alpha to balance between pure density and value
    alpha = 0.1
    reg_density = v_f / (w_f + alpha * c)
    
    # 5. Value * sqrt(1/w)
    with np.errstate(divide='ignore', invalid='ignore'):
        val_sqrt_inv_w = v_f * np.sqrt(1.0 / w_f)
        val_sqrt_inv_w[w_f == 0] = np.inf if v_f[w_f == 0] > 0 else 0
    
    # 6. Surrogate for "fits well": prefer items that are a good fraction of capacity
    #    but not too close to full (to leave room for others)
    #    Score = v * (1 - |w - c/2| / c)  -- peaks at w = c/2
    fit_score = v_f * (1.0 - np.abs(w_f - c * 0.5) / (c + 1e-9))
    
    # Normalize each score to [0, 1] using min-max
    def normalize(s):
        s_min = np.min(s)
        s_max = np.max(s)
        if s_max == s_min:
            return np.zeros_like(s)
        return (s - s_min) / (s_max - s_min)
    
    d1 = normalize(density)
    d2 = normalize(value)
    d3 = normalize(inv_weight)
    d4 = normalize(reg_density)
    d5 = normalize(val_sqrt_inv_w)
    d6 = normalize(fit_score)
    
    # Composite score with weights tuned for 0/1 knapsack
    # Density is usually strongest, but combining with fit and value helps
    composite = (2.5 * d1 + 
                 1.5 * d2 + 
                 1.0 * d3 + 
                 2.0 * d4 + 
                 1.0 * d5 + 
                 1.5 * d6)
    
    # Find best item in feasible set
    best_local_idx = np.argmax(composite)
    
    # Map back to original index
    best_global_idx = feasible_idx[best_local_idx]
    
    return int(best_global_idx)
