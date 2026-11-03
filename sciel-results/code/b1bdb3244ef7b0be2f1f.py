import numpy as np
from scipy.optimize import minimize

N = 100
STEP_WIDTH = 2.0 / N

def compute_overlaps(h):
    """Compute overlap for all N discrete shifts (periodic)."""
    overlaps = np.zeros(N)
    for k in range(N):
        shifted = np.roll(h, k)
        overlaps[k] = STEP_WIDTH * np.sum(h * (1 - shifted))
    return overlaps

def objective(params):
    """Objective: max overlap, with params mapped to feasible h."""
    # Map params to [0,1] via sigmoid
    h = 1.0 / (1.0 + np.exp(-params))
    # Project to satisfy sum constraint: sum(h) = N/2
    # Simple projection: shift and scale
    target_sum = N / 2.0
    current_sum = np.sum(h)
    if current_sum > 0:
        h = h * (target_sum / current_sum)
    # Clip to [0,1]
    h = np.clip(h, 0, 1)
    # Re-project after clip (simple fixed-point)
    for _ in range(10):
        target_sum = N / 2.0
        current_sum = np.sum(h)
        if abs(current_sum - target_sum) < 1e-12:
            break
        h = h + (target_sum - current_sum) / N
        h = np.clip(h, 0, 1)
    
    overlaps = compute_overlaps(h)
    return np.max(overlaps)

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Initial guess: uniform
    best_h = np.full(N, 0.5)
    best_score = np.max(compute_overlaps(best_h))
    
    # Try multiple random initializations
    np.random.seed(42)
    
    for init_idx in range(20):
        if time.time() > deadline:
            break
        
        if init_idx == 0:
            params = np.zeros(N)
        else:
            # Random initialization
            params = np.random.randn(N) * 2
        
        # Use SLSQP
        try:
            res = minimize(
                objective,
                params,
                method='SLSQP',
                options={'maxiter': 500, 'ftol': 1e-8}
            )
            
            if time.time() > deadline:
                break
            
            # Extract h from result
            h = 1.0 / (1.0 + np.exp(-res.x))
            target_sum = N / 2.0
            current_sum = np.sum(h)
            if current_sum > 0:
                h = h * (target_sum / current_sum)
            h = np.clip(h, 0, 1)
            for _ in range(20):
                current_sum = np.sum(h)
                if abs(current_sum - target_sum) < 1e-12:
                    break
                h = h + (target_sum - current_sum) / N
                h = np.clip(h, 0, 1)
            
            score = np.max(compute_overlaps(h))
            if score < best_score:
                best_score = score
                best_h = h.copy()
        except Exception:
            continue
    
    return best_h
