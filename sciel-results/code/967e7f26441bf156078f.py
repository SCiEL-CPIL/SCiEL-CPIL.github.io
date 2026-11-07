import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_value(h):
    h = np.asarray(h, dtype=float)
    # Ensure non-negative and finite
    h = np.maximum(h, 0.0)
    if np.max(h) < 1e-15:
        return 0.0
    
    # Normalize to max 1 to keep numbers stable, though value is scale invariant
    h_norm = h / np.max(h)
    
    g = np.convolve(h_norm, h_norm)
    m = len(g)
    y = np.concatenate([[0.0], g, [0.0]])
    w = 1.0 / (m + 1)
    
    # Calculate L2 term
    # l2 = sum (w/3) * (y[i]^2 + y[i]y[i+1] + y[i+1]^2)
    y_sq = y ** 2
    y_cross = y[:-1] * y[1:]
    l2 = (w / 3.0) * np.sum(y_sq[:-1] + y_cross + y_sq[1:])
    
    norm_1 = np.sum(g) / (m + 1)
    norm_inf = np.max(g)
    
    if norm_1 * norm_inf < 1e-15:
        return 0.0
        
    return l2 / (norm_1 * norm_inf)

def objective(z):
    """
    Objective function for scipy.optimize.minimize.
    We minimize -value(h).
    h is derived from z. To ensure h >= 0 and smoothness, we use h = z**2.
    """
    h = z ** 2
    val = compute_value(h)
    return -val

def construct():
    start_time = time.time()
    time_limit = 100.0  # Leave buffer for overhead
    
    best_val = -np.inf
    best_h = np.zeros(N)
    
    # Try a few different initializations
    rng = np.random.default_rng(42)
    
    # Strategies for initialization:
    # 1. Random
    # 2. Triangular
    # 3. Gaussian
    # 4. Uniform
    
    init_strategies = [
        lambda: rng.uniform(-1, 1, N),
        lambda: np.linspace(-1, 1, N),
        lambda: np.maximum(1.0 - 2.0 * np.abs(np.linspace(-0.5, 0.5, N)), 0.0),
        lambda: np.exp(-5 * np.linspace(-0.5, 0.5, N)**2),
        lambda: np.ones(N) * 0.1 + rng.uniform(0, 0.5, N)
    ]
    
    # We will run multiple optimizations
    max_runs = 20
    
    for run in range(max_runs):
        if time.time() - start_time > time_limit:
            break
            
        # Select an initialization strategy
        strat_idx = run % len(init_strategies)
        z_init = init_strategies[strat_idx]()
        
        # Add some noise to break symmetry if needed, especially for uniform
        if strat_idx == 4:
             z_init = z_init + rng.normal(0, 0.1, N)
             
        try:
            res = minimize(objective, z_init, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if res.success or res.fun < best_val:
                h_candidate = res.x ** 2
                val_candidate = compute_value(h_candidate)
                
                if val_candidate > best_val:
                    best_val = val_candidate
                    best_h = h_candidate.copy()
                    
        except Exception:
            continue

    # Fallback if optimization failed or didn't improve over seed
    # The seed is 50 equal heights. Let's ensure we return at least that.
    seed_h = np.ones(N)
    seed_val = compute_value(seed_h)
    
    if best_val < seed_val:
        return list(seed_h)
        
    return list(best_h)
