import numpy as np
from scipy.optimize import minimize
import time

N = 600
x_min, x_max = -0.25, 0.25
dx = (x_max - x_min) / N
x_grid = np.linspace(x_min + dx/2, x_max - dx/2, N)

def compute_autoconvolution(f_vals):
    """Compute autoconvolution of a step function via FFT."""
    n = len(f_vals)
    f_padded = np.zeros(2 * n)
    f_padded[:n] = f_vals
    F = np.fft.rfft(f_padded)
    auto = np.fft.irfft(F * np.conj(F), n=2*n)
    return auto

def compute_score(f_vals):
    """Compute the normalized peak autoconvolution."""
    conv = compute_autoconvolution(f_vals)
    peak = np.max(conv)
    integral = np.sum(f_vals) * dx
    if integral < 1e-12:
        return 1e12
    return peak / (integral * integral)

def softplus(x, beta=100.0):
    """Numerically stable softplus."""
    return np.where(x > 1.0/beta, x, np.log1p(np.exp(x)) / beta)

def parametric_to_step(params, K):
    """Convert parametric form to step heights."""
    a = params[0::3]
    mu = params[1::3]
    sigma = params[2::3]
    sigma = np.maximum(sigma, 1e-6)
    
    f_vals = np.zeros(N)
    for k in range(K):
        f_vals += a[k] * np.exp(-0.5 * ((x_grid - mu[k]) / sigma[k])**2)
    return np.maximum(f_vals, 0.0)

def objective(params, K, beta=50.0):
    """
    Objective function using soft-max surrogate for the max.
    """
    f_vals = parametric_to_step(params, K)
    integral = np.sum(f_vals) * dx
    if integral < 1e-12:
        return 1e12
    
    conv = compute_autoconvolution(f_vals)
    
    # Use log-sum-exp for smooth max approximation
    # Scale by 1/beta to control smoothness
    scaled = conv / beta
    max_scaled = np.max(scaled)
    logsumexp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    smooth_max = logsumexp * beta
    
    normalized = smooth_max / (integral * integral)
    return normalized

def exact_objective(params, K):
    """Objective function using exact max."""
    f_vals = parametric_to_step(params, K)
    integral = np.sum(f_vals) * dx
    if integral < 1e-12:
        return 1e12
    
    conv = compute_autoconvolution(f_vals)
    peak = np.max(conv)
    return peak / (integral * integral)

def local_refinement(f_vals, max_iters=50, tol=1e-8):
    """
    Local refinement using coordinate descent on the step heights directly.
    For each bin, try adjusting its value to reduce the peak autoconvolution.
    """
    f = f_vals.copy()
    best_score = compute_score(f)
    
    for iteration in range(max_iters):
        improved = False
        for i in range(N):
            # Try increasing and decreasing f[i]
            for delta in [0.01, -0.01, 0.005, -0.005]:
                f_new = f.copy()
                f_new[i] = max(0.0, f_new[i] + delta)
                score = compute_score(f_new)
                if score < best_score - tol:
                    f = f_new
                    best_score = score
                    improved = True
                    break
        
        if not improved:
            break
    
    return f, best_score

def polish_pass(f_vals, max_iters=100):
    """
    Final polish pass with finer steps.
    """
    f, score = local_refinement(f_vals, max_iters=max_iters, tol=1e-10)
    return f, score

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5 second safety margin
    
    best_result = None
    best_score = np.inf
    
    # Try different numbers of Gaussian components
    K_list = [1, 2, 3, 4]
    
    for K in K_list:
        if time.time() > deadline - 20:
            break
        
        np.random.seed(42 + K)
        
        # Multiple initializations
        for trial in range(8):
            if time.time() > deadline - 15:
                break
            
            if trial == 0:
                # Uniform-like initialization
                a_init = np.ones(K)
                mu_init = np.zeros(K)
                sigma_init = np.full(K, 0.15)
            elif trial == 1:
                # Single wide Gaussian
                a_init = np.array([2.0]) + np.zeros(K-1)
                mu_init = np.zeros(K)
                sigma_init = np.array([0.2]) + np.zeros(K-1)
            elif trial == 2:
                # Two peaks
                a_init = np.array([1.5, 1.5]) + np.zeros(max(0, K-2))
                mu_init = np.array([-0.1, 0.1]) + np.zeros(max(0, K-2))
                sigma_init = np.array([0.1, 0.1]) + np.zeros(max(0, K-2))
            else:
                # Random initialization
                a_init = np.random.uniform(0.5, 2.0, K)
                mu_init = np.random.uniform(-0.2, 0.2, K)
                sigma_init = np.random.uniform(0.05, 0.2, K)
            
            params_init = np.concatenate([a_init, mu_init, sigma_init])
            
            bounds = []
            for k in range(K):
                bounds.append((0.0, 10.0))
                bounds.append((-0.25, 0.25))
                bounds.append((0.01, 0.5))
            
            try:
                # Main optimization with soft-max surrogate
                result = minimize(
                    lambda p: objective(p, K, beta=50.0),
                    params_init,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-8}
                )
                
                score = result.fun
                if score < best_score:
                    best_score = score
                    best_result = result.x
                    
            except Exception:
                pass
        
        # Polish with exact objective
        if best_result is not None and time.time() < deadline - 10:
            try:
                result2 = minimize(
                    lambda p: exact_objective(p, K),
                    best_result,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10}
                )
                
                if result2.fun < best_score:
                    best_score = result2.fun
                    best_result = result2.x
                    
            except Exception:
                pass
    
    if best_result is None:
        # Fallback: constant function
        return [1.0] * N
    
    # Convert to step heights
    K = len(best_result) // 3
    f_vals = parametric_to_step(best_result, K)
    
    # Normalize
    integral = np.sum(f_vals) * dx
    if integral > 0:
        f_vals = f_vals / integral
    
    # Local refinement on the step heights
    if time.time() < deadline - 5:
        f_vals, score = local_refinement(f_vals, max_iters=30)
        if score < best_score:
            best_score = score
    
    # Final polish
    if time.time() < deadline - 2:
        f_vals, score = polish_pass(f_vals, max_iters=50)
    
    return list(f_vals)
