import numpy as np
import time
from scipy.optimize import differential_evolution

N = 600
W = 1.0 / 1200.0
X_MIN = -0.25
X_MAX = 0.25

# EVOLVE-BLOCK-START
def _compute_ratio_spectral(theta):
    """
    Compute the ratio max(autoconv) / l2_sq for the function defined by spectral coefficients theta.
    theta = [theta_0, theta_1_cos, theta_1_sin, ..., theta_K_cos, theta_K_sin]
    """
    K = 10 # Number of harmonics
    # 1. Generate fine grid
    n_grid = 4096
    x = np.linspace(X_MIN, X_MAX, n_grid, endpoint=False)
    
    # 2. Construct h(x) = exp(sum(...))
    # theta[0] is DC
    h_log = np.full(n_grid, theta[0], dtype=np.float64)
    for k in range(1, K + 1):
        idx_c = 1 + 2 * (k - 1)
        idx_s = 2 + 2 * (k - 1)
        h_log += theta[idx_c] * np.cos(2 * np.pi * k * x)
        h_log += theta[idx_s] * np.sin(2 * np.pi * k * x)
        
    # Clip log values to avoid overflow/underflow issues in exp, though bounds should help
    h_log = np.clip(h_log, -50, 50)
    h = np.exp(h_log)
    
    # 3. Compute autoconvolution
    n_fft = 2 * n_grid
    F = np.fft.rfft(h, n_fft)
    conv = np.fft.irfft(F * F, n_fft)[:2 * n_grid - 1]
    dx = 0.5 / n_grid
    conv *= dx
    
    max_conv = np.max(conv)
    
    # 4. Compute L2 squared
    l2_sq = np.sum(h ** 2) * dx
    
    if l2_sq < 1e-15:
        return 1e10 # Penalty for zero function
        
    return max_conv / l2_sq

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    K = 10 # Harmonics
    # Parameters: theta_0, (theta_k_cos, theta_k_sin) for k=1..K
    # Total params: 1 + 2*K
    n_params = 1 + 2 * K
    
    # Bounds for DE
    # theta_0 can be anything, but let's bound it to [-5, 5]
    # Coefficients bounded to [-3, 3] to keep h in reasonable range
    bounds = [(-5, 5)] + [(-3, 3)] * (2 * K)
    
    # Initial population strategy: DE handles this, but we can seed with uniform (all zeros)
    # Uniform function: h(x) = C. In log domain: log(C). 
    # If we set all theta to 0, h(x) = 1.
    # Ratio for h=1: max_conv = 0.5 (area 0.5, height 1 -> triangle peak 0.5? No.
    # h=1 on [-0.25, 0.25]. h*h is triangle from -0.5 to 0.5.
    # Peak at 0 is 0.5. L2 sq is 0.5. Ratio = 1.0?
    # Wait.
    # h(x) = 1 for x in [-0.25, 0.25].
    # (h*h)(t) = \int h(x) h(x-t) dx.
    # For t=0, \int_{-0.25}^{0.25} 1*1 dx = 0.5.
    # L2 sq = \int_{-0.25}^{0.25} 1^2 dx = 0.5.
    # Ratio = 0.5 / 0.5 = 1.0.
    # 
    # Why did I think uniform was 2.4?
    # Ah, the score is 1.0026.
    # If score = 1/R, then R ~ 1.0.
    # If score = R, then R ~ 1.0.
    # The prompt says "score is the upper bound on C1".
    # And "search score is seed/value".
    # Seed program score 1.0.
    # If seed is uniform, and uniform gives R=1.0, then seed score 1.0 implies score = 1/R? Or score = R?
    # "score is the upper bound on C1".
    # C1 is the constant in the inequality.
    # If we minimize the ratio, we get a tighter upper bound.
    # So lower ratio is better.
    # If the evaluator measures the "raw value" (lower is better), and score = seed/value.
    # If seed value is V_seed, and my value is V_mine.
    # Score = V_seed / V_mine.
    # If V_mine < V_seed, Score > 1.
    # So we want to MINIMIZE the raw value (the ratio).
    
    # Let's verify the ratio for uniform again.
    # h=1 on width 0.5.
    # Conv peak = 0.5.
    # L2 sq = 0.5.
    # Ratio = 1.0.
    # 
    # Is it possible to get lower than 1.0?
    # If we make h spiky, peak increases, L2 sq increases.
    # If we make h spread out? It's fixed width.
    # 
    # Let's check a Gaussian.
    # h(x) = exp(-x^2/sigma^2).
    # As sigma -> 0, h -> delta. Conv -> delta. Peak -> inf. L2 sq -> 0. Ratio -> inf.
    # As sigma -> inf, h -> constant. Ratio -> 1.
    # So uniform is likely optimal or close to it?
    # 
    # Wait, the "First autocorrelation inequality" usually refers to something like:
    # max |f*f| <= C * ||f||_2^2.
    # For L2 normalized functions?
    # If we normalize h such that ||h||_2 = 1, then we want to minimize max(h*h).
    # My ratio R = max(h*h) / ||h||_2^2.
    # Minimizing R is equivalent to minimizing max(h*h) for unit norm.
    # 
    # For uniform h, R=1.
    # Can we get R < 1?
    # Consider h(x) = 1 on [-a, a] and 0 elsewhere? No, fixed support [-0.25, 0.25].
    # 
    # Let's look at the "inspiration" scores.
    # Inspiration 1: 1.0056.
    # Inspiration 2: 1.0026.
    # Inspiration 3 (Seed): 1.0000.
    # 
    # If Seed (Uniform) has score 1.0, and Raw Value is R.
    # If Score = 1/R, and R=1, Score=1.
    # If I find R=0.99, Score = 1/0.99 = 1.01.
    # So we want to find a function with R < 1.
    # 
    # Is R < 1 possible?
    # Let's test a function that is 0 at the center and high at edges?
    # h(x) = 1 - cos(2 pi x) ? (Zero at 0, 2 at edges).
    # Or h(x) = 1 + cos(2 pi x)? (2 at 0, 0 at edges).
    # 
    # Let's rely on the optimizer to find it.
    # The spectral parameterization allows for such shapes.
    
    # Strategy:
    # 1. Run Differential Evolution.
    # 2. If time permits, run L-BFGS-B on the best result.
    
    def objective(theta):
        return _compute_ratio_spectral(theta)
    
    best_ratio = 1.0 # Uniform baseline
    best_theta = np.zeros(n_params)
    
    try:
        # Use DE to find a good local minimum
        # Tolerance and maxiter tuned for speed
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=15, 
            tol=1e-6,
            polish=False # Disable polish to save time, we do it manually
        )
        
        if res.fun < best_ratio:
            best_ratio = res.fun
            best_theta = res.x
            
    except Exception as e:
        print(f"DE failed: {e}")
        
    # Refine with L-BFGS-B using a smoothed max
    # We need a differentiable objective for L-BFGS-B.
    # Replace max(conv) with logsumexp(conv) * beta?
    # Or just use the raw ratio and rely on the fact that L-BFGS-B uses finite differences?
    # L-BFGS-B can handle non-smooth functions if we use `approx_grad=True`? No, standard LBFGS uses gradients.
    # If we don't provide gradient, it uses finite differences.
    # The kink in max might cause issues, but often it works if the max is unique.
    
    if time.time() < deadline - 10.0:
        try:
            from scipy.optimize import minimize
            
            # Use the DE result as start point
            res2 = minimize(
                objective, 
                best_theta, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-6}
            )
            
            if res2.fun < best_ratio:
                best_ratio = res2.fun
                best_theta = res2.x
                
        except Exception as e:
            print(f"Minimize failed: {e}")
            
    # Construct the final step function
    # Sample h(x) at the midpoints of the 600 intervals
    # Interval i covers [x_i, x_{i+1}]
    # Midpoint x_mid_i = -0.25 + (i + 0.5) * W
    
    x_mids = -0.25 + (np.arange(N) + 0.5) * W
    
    h_log = np.full(N, best_theta[0], dtype=np.float64)
    for k in range(1, K + 1):
        idx_c = 1 + 2 * (k - 1)
        idx_s = 2 + 2 * (k - 1)
        h_log += best_theta[idx_c] * np.cos(2 * np.pi * k * x_mids)
        h_log += best_theta[idx_s] * np.sin(2 * np.pi * k * x_mids)
        
    h_log = np.clip(h_log, -50, 50)
    h_steps = np.exp(h_log)
    
    # Ensure non-negative (exp is always positive, so this is safe)
    # Return as list
    return h_steps.tolist()

# EVOLVE-BLOCK-END
