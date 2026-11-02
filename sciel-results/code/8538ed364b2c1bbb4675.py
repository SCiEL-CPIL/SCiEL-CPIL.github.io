"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy import optimize

N = 600

def compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of nonnegative values.
    Returns max_k (h*h)[k] / (sum h)^2
    """
    s = np.sum(h)
    if s == 0:
        return 0.0
    # Autoconvolution (linear)
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    return max_conv / (s * s)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    x = np.linspace(-0.25, 0.25, N)
    
    # Generate initial candidates
    candidates = []
    
    # 1. Uniform
    candidates.append(np.ones(N))
    
    # 2. Cosine window (smooth, zero at edges)
    candidates.append(0.5 * (1 + np.cos(4 * np.pi * x)))
    
    # 3. Gaussian (centered)
    for sigma in [0.05, 0.1, 0.15, 0.2]:
        candidates.append(np.exp(-x**2 / (2*sigma**2)))
        
    # 4. Two-humped (mass at edges)
    for center in [0.1, 0.15, 0.2]:
        for w in [0.01, 0.02, 0.05]:
            candidates.append(np.exp(-(x-center)**2/w) + np.exp(-(x+center)**2/w))
            
    # 5. Polynomial bumps: (1 - (x/0.25)^2)^p
    for p in [0.5, 1, 2, 3, 4, 6, 8]:
        t = np.abs(x) / 0.25
        h_p = np.maximum(0, 1 - t**2) ** p
        candidates.append(h_p)

    # Evaluate initial candidates
    best_h = candidates[0].copy()
    best_score = compute_score(best_h)
    
    for h in candidates[1:]:
        if time.time() > deadline:
            break
        score = compute_score(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()

    # Optimization using L-BFGS-B
    # Objective: minimize max_k conv_k / (sum h)^2
    # Since max is non-smooth, we can use a soft-max approximation or just rely on L-BFGS-B
    # finding a local minimum of the max function (subgradient).
    # However, L-BFGS-B requires a smooth objective.
    # We can define the objective as the max value, and let L-BFGS-B handle the non-smoothness
    # or use a smooth surrogate like log-sum-exp.
    # Let's try direct minimization of the max value. L-BFGS-B might struggle with non-smoothness.
    # Alternative: Minimize the sum of squares of the top-k values?
    # Or just use a simple coordinate descent / projected gradient if needed.
    
    # Let's try a simple local search with random restarts and gradient-free optimization
    # since the dimension is 600.
    
    # Actually, let's try to use `scipy.optimize.minimize` with method 'L-BFGS-B'
    # on the log-sum-exp of the convolution values, which is a smooth upper bound on the max.
    # log(sum(exp(conv/T))) * T approximates max(conv) as T -> 0.
    
    def objective_smooth(h_flat):
        h = np.maximum(0, h_flat) # Project to non-negative
        s = np.sum(h)
        if s < 1e-10:
            return 1e10
        conv = np.convolve(h, h, mode='full')
        # Normalize
        norm_conv = conv / (s * s)
        # Soft max
        T = 1.0
        max_val = np.max(norm_conv)
        # Shift for numerical stability
        shifted = norm_conv - max_val
        log_sum_exp = max_val + np.log(np.sum(np.exp(shifted / T)) / len(norm_conv)) * T
        # We want to minimize the max, so minimizing log-sum-exp is a good proxy
        return log_sum_exp

    # We also want to encourage the sum to be non-zero, but the objective handles it.
    # Let's run L-BFGS-B from the best candidate.
    
    h0 = best_h.copy()
    # Scale h0 so that sum is around 1 to help optimization?
    # The objective is scale invariant, but numerical conditioning might be better.
    s0 = np.sum(h0)
    if s0 > 0:
        h0 = h0 / s0
    
    # Bounds: [0, inf)
    bounds = [(0, None)] * N
    
    # Run optimization
    try:
        res = optimize.minimize(
            objective_smooth, 
            h0, 
            method='L-BFGS-B', 
            bounds=bounds, 
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if time.time() < deadline:
            h_opt = np.maximum(0, res.x)
            score_opt = compute_score(h_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
    except Exception:
        pass

    # Try a few more random restarts if time permits
    if time.time() < deadline - 30:
        for _ in range(5):
            if time.time() > deadline:
                break
            # Random initialization
            h_rand = np.random.uniform(0, 1, N)
            # Maybe mix with best_h
            h_rand = 0.5 * h_rand + 0.5 * best_h
            s_rand = np.sum(h_rand)
            if s_rand > 0:
                h_rand = h_rand / s_rand
            
            try:
                res = optimize.minimize(
                    objective_smooth, 
                    h_rand, 
                    method='L-BFGS-B', 
                    bounds=bounds, 
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                h_opt = np.maximum(0, res.x)
                score_opt = compute_score(h_opt)
                if score_opt < best_score:
                    best_score = score_opt
                    best_h = h_opt.copy()
            except Exception:
                pass

    return best_h.tolist()
# EVOLVE-BLOCK-END
