import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50

def compute_score(h):
    """
    Compute the ratio of squared L2 norm to (L1 norm * sup norm of autoconvolution).
    h: array of nonnegative heights
    Returns the ratio value (higher is better).
    """
    h = np.asarray(h, dtype=float)
    if len(h) != N:
        return 0.0
    if np.any(h < -1e-12):
        return 0.0
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    
    l2_sq = np.sum(h**2)
    
    # Autoconvolution: c[k] = sum_i h[i] * h[i+k] for k = -(N-1)..(N-1)
    # Using np.correlate for autoconvolution
    c = np.correlate(h, h, mode='full')
    sup_c = np.max(c)
    
    if sup_c < 1e-15:
        return 0.0
    
    return l2_sq / (l1 * sup_c)

def negative_score(h):
    """Negative of score for minimization."""
    return -compute_score(h)

def soft_max_score(h, temperature=0.01):
    """
    Smooth surrogate using log-sum-exp for the supremum of autoconvolution.
    This makes the objective differentiable for gradient-based optimization.
    """
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    
    l2_sq = np.sum(h**2)
    
    # Autoconvolution
    c = np.correlate(h, h, mode='full')
    
    # Soft max using log-sum-exp trick for numerical stability
    c_max = np.max(c)
    if c_max < 1e-15:
        return 0.0
    
    # log(sum(exp(c_i / T))) * T ≈ max(c_i) when T is small
    # Use scaled version for stability
    scaled_c = c / temperature
    scaled_max = np.max(scaled_c)
    log_sum_exp = scaled_max + np.log(np.sum(np.exp(scaled_c - scaled_max)))
    soft_sup_c = log_sum_exp * temperature
    
    return l2_sq / (l1 * soft_sup_c)

def negative_soft_score(h, temperature=0.01):
    """Negative of soft max score for minimization."""
    return -soft_max_score(h, temperature)

def construct():
    # Keep the known good solution as baseline
    best_h = [0.0] * N
    best_h[0] = 2.0 / 3.0
    best_h[1] = 1.0 / 3.0
    best_score = compute_score(best_h)
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Strategy 1: Use smooth surrogate with gradient-based optimization
    # Try multiple temperatures to balance smoothness and accuracy
    
    temperatures = [0.05, 0.02, 0.01, 0.005]
    bounds = [(0.0, 1.0)] * N
    
    # Initial points: known good + random starts
    x0_list = []
    
    # Known good solution
    x0_good = np.zeros(N)
    x0_good[0] = 2.0 / 3.0
    x0_good[1] = 1.0 / 3.0
    x0_list.append(x0_good)
    
    # Random initializations
    rng = np.random.RandomState(42)
    for _ in range(10):
        x0_rand = rng.random(N)
        x0_rand /= x0_rand.sum() if x0_rand.sum() > 0 else 1
        x0_list.append(x0_rand)
    
    for x0 in x0_list:
        if time.time() > deadline:
            break
        
        for temp in temperatures:
            if time.time() > deadline:
                break
            
            try:
                res = minimize(
                    lambda h: negative_soft_score(h, temp),
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.fun < 0:
                    h_candidate = res.x
                    score = compute_score(h_candidate)
                    if score > best_score:
                        best_score = score
                        best_h = list(h_candidate)
            except Exception:
                pass
    
    # Check time
    if time.time() > deadline:
        return best_h
    
    # Strategy 2: Fine-tune with exact (non-smooth) objective from best found
    try:
        x0_best = np.array(best_h)
        res = minimize(negative_score, x0_best, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10})
        if res.fun < 0:
            score = -res.fun
            if score > best_score:
                best_score = score
                best_h = list(res.x)
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        return best_h
    
    # Strategy 3: Try structured initializations (Sidon-like patterns)
    # Sidon sets have unique pairwise sums, reducing autoconvolution peaks
    
    # Generate some Sidon-like patterns
    sidon_patterns = []
    
    # Simple Sidon set: {0, 1, 4, 10, 20, ...}
    for start in range(0, N - 5):
        if time.time() > deadline:
            break
        
        # Pattern with increasing gaps
        positions = [start]
        gap = 1
        while len(positions) < 8 and positions[-1] + gap < N:
            positions.append(positions[-1] + gap)
            gap += 1
        
        if len(positions) >= 3:
            x0_sidon = np.zeros(N)
            for i, pos in enumerate(positions):
                x0_sidon[pos] = 1.0 / len(positions)
            sidon_patterns.append(x0_sidon)
    
    for x0_sidon in sidon_patterns[:5]:
        if time.time() > deadline:
            break
        
        for temp in [0.02, 0.01]:
            if time.time() > deadline:
                break
            
            try:
                res = minimize(
                    lambda h: negative_soft_score(h, temp),
                    x0_sidon,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.fun < 0:
                    h_candidate = res.x
                    score = compute_score(h_candidate)
                    if score > best_score:
                        best_score = score
                        best_h = list(h_candidate)
            except Exception:
                pass
    
    # Check time
    if time.time() > deadline:
        return best_h
    
    # Strategy 4: Final polish with tight tolerance
    try:
        x0_final = np.array(best_h)
        res = minimize(negative_score, x0_final, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12})
        if res.fun < 0:
            score = -res.fun
            if score > best_score:
                best_score = score
                best_h = list(res.x)
    except Exception:
        pass
    
    return best_h
