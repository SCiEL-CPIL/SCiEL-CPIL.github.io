import numpy as np
import time

def _compute_ratio(P):
    P = np.clip(P, 0, None)
    s = P.sum()
    if s <= 1e-15:
        return 0.0
    P = P / s
    
    y_idx, x_idx = np.indices((10, 10))
    x_flat = x_idx.flatten()
    y_flat = y_idx.flatten()
    w = P.flatten()
    
    def entropy_from_values(values, weights):
        min_v = np.min(values)
        shifted = values - min_v
        counts = np.bincount(shifted, weights=weights)
        probs = counts[counts > 1e-15]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))
    
    d_vals = x_flat - y_flat
    D = entropy_from_values(d_vals, w)
    
    h0_vals = x_flat
    h1_vals = x_flat + y_flat
    h2_vals = x_flat + 2 * y_flat
    
    H0 = entropy_from_values(h0_vals, w)
    H1 = entropy_from_values(h1_vals, w)
    H2 = entropy_from_values(h2_vals, w)
    
    M = max(H0, H1, H2)
    
    if M <= 1e-9:
        return 0.0
        
    return D / M

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    P = np.ones((n, n), dtype=np.float64) / (n * n)
    
    start_time = time.time()
    timeout = 600
    safety_margin = 10
    
    best_score = _compute_ratio(P)
    best_P = P.copy()
    
    np.random.seed(42)
    
    # Multi-start local search
    num_restarts = 50
    time_per_restart = (timeout - safety_margin) / num_restarts
    
    for r in range(num_restarts):
        if time.time() - start_time > timeout - safety_margin:
            break
            
        # Initialize current P
        if r == 0:
            current_P = P.copy()
        else:
            # Perturb best found or uniform
            current_P = best_P + np.random.normal(0, 0.1, (n, n))
            current_P = np.clip(current_P, 0, None)
            s = current_P.sum()
            if s > 0:
                current_P = current_P / s
            else:
                current_P = P.copy()
                
        current_score = _compute_ratio(current_P)
        
        # Hill climbing
        step_size = 0.2
        local_time_limit = time_per_restart
        local_start = time.time()
        
        while time.time() - local_start < local_time_limit:
            i, j = np.random.randint(0, n, 2)
            delta = np.random.uniform(-step_size, step_size)
            
            new_P = current_P.copy()
            new_P[i, j] += delta
            
            if np.any(new_P < -1e-9):
                continue
                
            new_P = np.clip(new_P, 0, None)
            s = new_P.sum()
            if s <= 1e-9:
                continue
            new_P = new_P / s
            
            new_score = _compute_ratio(new_P)
            
            if new_score > current_score:
                current_P = new_P
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_P = current_P.copy()
                step_size *= 0.99
            else:
                # Occasionally accept worse moves to escape local minima?
                # Simple hill climbing is safer for time limits.
                pass
                
    return best_P
