import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 50
DEADLINE = 115.0  # 120s limit, 5s margin

def compute_ratio(h):
    """
    Compute the ratio: ||h||_2^2 / (||h||_1 * ||h * h||_inf)
    h is a 1D array of non-negative values.
    """
    h = np.array(h)
    if np.any(h < -1e-9):
        return -1.0
    h = np.clip(h, 0, None)
    
    l1 = np.sum(h)
    if l1 < 1e-12:
        return -1.0
        
    l2_sq = np.sum(h**2)
    
    # Linear convolution
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(np.abs(h_conv))
    
    if max_conv < 1e-12:
        return -1.0
        
    return (l2_sq / (l1 * max_conv))

def params_to_heights(params, family='gaussian'):
    """
    Convert continuous parameters to discrete heights on [-1/4, 1/4].
    params: [mean, std/width]
    """
    x = np.linspace(-0.25, 0.25, N)
    
    if family == 'gaussian':
        mu, sigma = params[0], params[1]
        if sigma < 1e-6: sigma = 1e-6
        h = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    elif family == 'cosine':
        mu, width = params[0], params[1]
        if width < 1e-6: width = 1e-6
        # Support is [mu - width, mu + width]
        # Cosine bump: 0.5 * (1 + cos(pi * (x - mu) / width)) within support
        z = (x - mu) / width
        h = np.zeros(N)
        mask = np.abs(z) <= 1
        h[mask] = 0.5 * (1 + np.cos(np.pi * z[mask]))
    elif family == 'triangle':
        mu, width = params[0], params[1]
        if width < 1e-6: width = 1e-6
        z = (x - mu) / width
        h = np.maximum(0, 1 - np.abs(z))
    else:
        h = np.ones(N)
        
    # Normalize to L1=1 for scale invariance stability in optimization
    s = np.sum(h)
    if s > 1e-12:
        h = h / s
    else:
        h = np.ones(N) / N
        
    return h

def objective(params, family='gaussian'):
    """
    Objective function for minimization (negative log ratio).
    """
    try:
        h = params_to_heights(params, family)
        r = compute_ratio(h)
        if r <= 0:
            return 1000.0
        return -np.log(r)
    except Exception:
        return 1000.0

def optimize_continuous(family, time_limit):
    """
    Optimize the continuous parameters for a given family.
    """
    start_time = time.time()
    
    # Bounds for parameters:
    # mean in [-0.25, 0.25]
    # width/std in [0.01, 0.25] (to avoid degenerate spikes or flat lines)
    bounds = [(-0.25, 0.25), (0.01, 0.25)]
    
    best_h = None
    best_val = -1.0
    
    # 1. Differential Evolution for global search (robust to local minima)
    if time.time() - start_time < time_limit * 0.5:
        try:
            de_res = differential_evolution(
                lambda p: objective(p, family),
                bounds,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                polish=False  # We will polish manually
            )
            if de_res.success or de_res.fun < 1000:
                h_de = params_to_heights(de_res.x, family)
                val_de = compute_ratio(h_de)
                if val_de > best_val:
                    best_val = val_de
                    best_h = h_de
        except Exception:
            pass

    # 2. Local refinement from best DE result and random starts
    candidates = []
    if best_h is not None:
        # Recover params from best_h? No, keep de_res.x
        candidates.append(de_res.x)
    
    # Add some random starts
    for _ in range(10):
        if time.time() - start_time > time_limit * 0.8:
            break
        rand_params = np.array([
            np.random.uniform(-0.25, 0.25),
            np.random.uniform(0.02, 0.20)
        ])
        candidates.append(rand_params)
        
    # Specific good starts based on intuition
    candidates.append([0.0, 0.05]) # Narrow Gaussian
    candidates.append([0.0, 0.15]) # Wide Gaussian
    candidates.append([0.0, 0.10]) # Cosine mid
    candidates.append([0.0, 0.05]) # Triangle narrow

    for p0 in candidates:
        if time.time() - start_time > time_limit:
            break
        
        try:
            # BFGS is good for smooth objectives
            res = minimize(
                lambda p: objective(p, family),
                p0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            if res.fun < 1000:
                h_opt = params_to_heights(res.x, family)
                val_opt = compute_ratio(h_opt)
                if val_opt > best_val:
                    best_val = val_opt
                    best_h = h_opt
        except Exception:
            pass

    if best_h is None:
        # Fallback to a reasonable default
        best_h = params_to_heights([0.0, 0.1], family)
        
    return best_h

def construct():
    start_time = time.time()
    
    best_h = None
    best_val = -1.0
    
    # Try different continuous families
    families = ['gaussian', 'cosine', 'triangle']
    
    # Allocate time roughly equally, but give more to Gaussian which is often optimal for such ratios
    time_per_family = (DEADLINE - start_time - 5) / len(families)
    
    for family in families:
        if time.time() > DEADLINE - 5:
            break
            
        h_fam = optimize_continuous(family, time_limit=time_per_family)
        val_fam = compute_ratio(h_fam)
        
        if val_fam > best_val:
            best_val = val_fam
            best_h = h_fam

    # If no valid result, fallback to uniform
    if best_h is None or best_val < 0:
        best_h = np.ones(N) / N
        best_val = compute_ratio(best_h)

    # Final check: ensure non-negative and valid
    best_h = np.clip(best_h, 0, None)
    s = np.sum(best_h)
    if s > 1e-12:
        best_h = best_h / s
    else:
        best_h = np.ones(N) / N

    return list(best_h)
