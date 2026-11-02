import numpy as np
import time
from scipy.optimize import minimize

N = 400

def construct():
    start_time = time.time()
    deadline = 140.0 # 140s limit, 10s margin
    
    # Initial guess: Blackman window
    x0 = np.blackman(N)
    # Ensure non-zero sum
    if np.sum(x0) == 0:
        x0 = np.ones(N)
    
    # Normalize sum to 1 to fix scale invariance
    x0 = x0 / np.sum(x0)
    
    def objective(x):
        # Compute autoconvolution
        # Linear convolution
        # Use FFT for speed
        # Zero pad to 2*N
        n_fft = 2 * N
        X = np.fft.rfft(x, n=n_fft)
        A = np.fft.irfft(X * X, n=n_fft)[:N]
        # A[k] is sum x_i x_{i+k}
        # Exclude k=0
        if len(A) > 1:
            side_lobes = np.abs(A[1:])
            max_side = np.max(side_lobes)
        else:
            max_side = 0.0
        
        # Denominator: (sum x)^2
        s = np.sum(x)
        denom = s * s
        if denom < 1e-12:
            return 1e12 # Penalty for zero sum
        
        return max_side / denom

    # We want to minimize the objective
    # But L-BFGS needs gradient.
    # We can use finite differences or analytic gradient.
    # Analytic gradient of max is hard.
    # We can use a smooth approximation: log-sum-exp of squared sidelobes.
    
    def smooth_objective(x):
        n_fft = 2 * N
        X = np.fft.rfft(x, n=n_fft)
        A = np.fft.irfft(X * X, n=n_fft)[:N]
        
        if len(A) > 1:
            side_lobes_sq = A[1:]**2
            # Soft max
            # max(y) approx log(sum(exp(y/beta))) * beta
            beta = 10.0
            # Clip to avoid overflow
            max_val = np.max(side_lobes_sq)
            if max_val > 0:
                log_sum_exp = max_val + np.log(np.sum(np.exp((side_lobes_sq - max_val)/beta)))
                smooth_max = log_sum_exp * beta
            else:
                smooth_max = 0.0
        else:
            smooth_max = 0.0
        
        s = np.sum(x)
        denom = s * s
        if denom < 1e-12:
            return 1e12
        
        return smooth_max / denom

    # Use L-BFGS-B
    # We need to re-normalize x to keep sum=1?
    # If we let x vary, the denominator changes.
    # It's better to optimize x with sum=1 constraint.
    # L-BFGS-B doesn't handle equality constraints well.
    # We can parameterize x = y / sum(y).
    # Or just optimize x and accept the scale invariance.
    # Since the function is scale invariant, the optimizer might drift.
    # We will normalize x after each step? No, that breaks gradient.
    # Instead, we optimize y where x = y / ||y||_1?
    # Simpler: Just optimize x. The minimum will be at some scale.
    # The value of the objective is scale invariant.
    
    # However, L-BFGS might struggle with the flat direction of scaling.
    # We can fix one variable or use a constraint.
    # Let's just run L-BFGS on x.
    
    res = None
    try:
        # Limit iterations
        max_iter = 500
        # Check time
        if time.time() - start_time > deadline - 10:
            return x0
            
        res = minimize(smooth_objective, x0, method='L-BFGS-B', 
                       options={'maxiter': max_iter, 'disp': False})
        
        if res.success:
            x_opt = res.x
        else:
            x_opt = res.x # Use best found
        
        # Final polish: Normalize to sum 1
        x_opt = x_opt / np.sum(x_opt)
        
    except Exception as e:
        # Fallback to initial window
        x_opt = x0

    return x_opt

# EVOLVE-BLOCK-END
