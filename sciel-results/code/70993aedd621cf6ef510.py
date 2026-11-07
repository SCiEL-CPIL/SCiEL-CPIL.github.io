"""Optimized program for Erdos minimum overlap task."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100

def compute_value(h):
    """Compute the objective value."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / len(h) * 2

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Start with flat seed
    h_best = np.full(N, 0.5)
    val_best = compute_value(h_best)
    
    # Strategy: use sinusoidal perturbations with various frequencies/phases
    # h[i] = 0.5 + A * sin(2*pi*f*i/N + phase)
    # Need to ensure h[i] in [0,1] and sum(h) = 50
    
    # Try multiple frequency/phase combinations
    candidates = []
    
    # First, try simple periodic patterns
    for freq in range(1, 50):
        for phase in [0, np.pi/2, np.pi, 3*np.pi/2]:
            t = np.arange(N)
            h = 0.5 + 0.3 * np.sin(2 * np.pi * freq * t / N + phase)
            # Clip to [0,1]
            h = np.clip(h, 0, 1)
            # Adjust to meet sum constraint
            diff = 50.0 - np.sum(h)
            if abs(diff) < 1e-6:
                candidates.append(h.copy())
            else:
                # Add uniform offset to fix sum, then re-clip
                h2 = h + diff / N
                h2 = np.clip(h2, 0, 1)
                if abs(np.sum(h2) - 50.0) < 1e-6:
                    candidates.append(h2.copy())
    
    # Try random smooth patterns
    np.random.seed(42)
    for _ in range(200):
        # Generate smooth random perturbation
        freq = np.random.uniform(0.5, 10)
        amp = np.random.uniform(0.1, 0.4)
        phase = np.random.uniform(0, 2 * np.pi)
        t = np.arange(N)
        h = 0.5 + amp * np.sin(2 * np.pi * freq * t / N + phase)
        h = np.clip(h, 0, 1)
        diff = 50.0 - np.sum(h)
        h2 = h + diff / N
        h2 = np.clip(h2, 0, 1)
        if abs(np.sum(h2) - 50.0) < 1e-6:
            candidates.append(h2.copy())
    
    # Evaluate all candidates
    for h in candidates:
        v = compute_value(h)
        if v < val_best:
            val_best = v
            h_best = h.copy()
    
    # Check time
    if time.time() > deadline - 30:
        return h_best
    
    # Local optimization using scipy
    # Parameterize: h = 0.5 + x, where sum(x) = 0, x[i] in [-0.5, 0.5]
    # Use a smooth max surrogate for the objective
    
    def objective(x):
        h = 0.5 + x
        # Enforce bounds softly
        h_clipped = np.clip(h, 0, 1)
        # Compute max correlation with soft-max
        corr = np.correlate(h_clipped, 1 - h_clipped, mode="full")
        # Soft max with temperature
        temp = 0.01
        log_sum_exp = np.log(np.sum(np.exp(corr / temp))) * temp
        return log_sum_exp / N * 2
    
    # Start from best candidate
    x0 = h_best - 0.5
    
    # Constraints: sum(x) = 0, bounds on x
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x)}]
    bounds = [(-0.5, 0.5)] * N
    
    if time.time() < deadline - 20:
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                            constraints=constraints, options={'maxiter': 500, 'ftol': 1e-10})
            if result.success:
                h_opt = 0.5 + result.x
                h_opt = np.clip(h_opt, 0, 1)
                # Fix sum exactly
                diff = 50.0 - np.sum(h_opt)
                h_opt = h_opt + diff / N
                h_opt = np.clip(h_opt, 0, 1)
                # Final fix if needed
                if abs(np.sum(h_opt) - 50.0) > 1e-9:
                    # Adjust largest elements
                    h_opt = h_best  # Fallback
                else:
                    v_opt = compute_value(h_opt)
                    if v_opt < val_best:
                        val_best = v_opt
                        h_best = h_opt.copy()
        except Exception:
            pass
    
    # Final polish: ensure exact sum
    diff = 50.0 - np.sum(h_best)
    if abs(diff) > 1e-15:
        # Distribute adjustment to elements not at boundary
        mask = (h_best > 1e-6) & (h_best < 1 - 1e-6)
        if np.sum(mask) > 0:
            h_best[mask] += diff / np.sum(mask)
            h_best = np.clip(h_best, 0, 1)
        else:
            h_best = np.full(N, 0.5)
    
    return h_best

# EVOLVE-BLOCK-END
