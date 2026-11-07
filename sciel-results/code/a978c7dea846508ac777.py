"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _objective(params):
    """Objective: maximize the max correlation term (we minimize negative)."""
    h = params
    # Ensure h is in [0,1] via soft projection for smoothness during optimization
    # But we'll handle constraints separately
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr)

def _construct_with_opt():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_h = np.full(N, 0.5)
    best_val = _objective(best_h) / N * 2
    
    # Try multiple initial patterns
    patterns = []
    
    # Pattern 1: uniform (baseline)
    patterns.append(np.full(N, 0.5))
    
    # Pattern 2: sinusoidal with various periods
    for period in [10, 20, 25, 50, 100]:
        t = np.arange(N)
        h = 0.5 + 0.5 * np.sin(2 * np.pi * t / period)
        h = np.clip(h, 0, 1)
        # Adjust to make sum exactly N/2
        diff = N / 2.0 - np.sum(h)
        h = h + diff / N
        h = np.clip(h, 0, 1)
        patterns.append(h)
    
    # Pattern 3: cosine
    for period in [10, 20, 25, 50]:
        t = np.arange(N)
        h = 0.5 + 0.5 * np.cos(2 * np.pi * t / period)
        h = np.clip(h, 0, 1)
        diff = N / 2.0 - np.sum(h)
        h = h + diff / N
        h = np.clip(h, 0, 1)
        patterns.append(h)
    
    # Pattern 4: piecewise linear (triangle wave)
    for period in [10, 20, 25, 50]:
        t = np.arange(N)
        phase = (t % period) / period
        h = 0.5 + 0.5 * np.where(phase < 0.5, 4 * phase - 1, 3 - 4 * phase)
        h = np.clip(h, 0, 1)
        diff = N / 2.0 - np.sum(h)
        h = h + diff / N
        h = np.clip(h, 0, 1)
        patterns.append(h)
    
    # Evaluate all patterns
    for h in patterns:
        if time.time() > deadline:
            break
        # Ensure sum constraint
        s = np.sum(h)
        if abs(s - N / 2.0) > 1e-9 * N:
            continue
        val = _objective(h) / N * 2
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Local optimization from best pattern
    if time.time() < deadline:
        def _obj_smooth(x):
            h = x
            corr = np.correlate(h, 1 - h, mode="full")
            # Use soft-max for smoothness
            scale = 100.0
            m = np.max(corr)
            sm = m + np.log(np.sum(np.exp((corr - m) / scale))) * scale
            return sm
        
        def _constraint_sum(x):
            return np.sum(x) - N / 2.0
        
        constraints = [
            {'type': 'eq', 'fun': _constraint_sum},
        ]
        bounds = [(0.0, 1.0)] * N
        
        try:
            res = minimize(_obj_smooth, best_h, method='SLSQP', bounds=bounds, 
                          constraints=constraints, options={'maxiter': 200, 'ftol': 1e-12})
            if time.time() < deadline:
                h_opt = np.clip(res.x, 0, 1)
                # Fix sum
                diff = N / 2.0 - np.sum(h_opt)
                h_opt = h_opt + diff / N
                h_opt = np.clip(h_opt, 0, 1)
                val = _objective(h_opt) / N * 2
                if val < best_val and abs(np.sum(h_opt) - N/2.0) < 1e-9 * N:
                    best_val = val
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final polish: try to reduce max correlation by adjusting peaks
    if time.time() < deadline - 5.0:
        for _ in range(50):
            if time.time() > deadline - 2.0:
                break
            h = best_h.copy()
            corr = np.correlate(h, 1 - h, mode="full")
            max_idx = np.argmax(corr)
            # The lag with max correlation
            # Try to reduce it by adjusting h values
            # Simple gradient-free: perturb and keep if better
            improved = False
            for trial in range(20):
                h_trial = h.copy()
                # Random small perturbation
                mask = np.random.random(N) < 0.3
                h_trial[mask] += np.random.normal(0, 0.01, size=np.sum(mask))
                h_trial = np.clip(h_trial, 0, 1)
                # Fix sum
                diff = N / 2.0 - np.sum(h_trial)
                h_trial = h_trial + diff / N
                h_trial = np.clip(h_trial, 0, 1)
                if abs(np.sum(h_trial) - N/2.0) > 1e-9 * N:
                    continue
                val_trial = _objective(h_trial) / N * 2
                if val_trial < best_val:
                    best_val = val_trial
                    best_h = h_trial.copy()
                    h = h_trial.copy()
                    improved = True
                    break
            if not improved:
                break
    
    return best_h

# EVOLVE-BLOCK-START
def construct():
    return _construct_with_opt()
# EVOLVE-BLOCK-END
