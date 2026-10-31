"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 600
TOTAL_TIME = 150.0  # seconds
SAFETY_MARGIN = 5.0  # seconds before kill

# EVOLVE-BLOCK-START

def _autoconv_peak(heights):
    """Compute the normalized maximum of the autoconvolution of a step function.
    
    The step function is defined on [-1/4, 1/4] with N equal intervals.
    The autoconvolution is computed and we return the max value normalized appropriately.
    """
    # The step function has constant height on each interval
    # Interval width
    dx = (0.25 - (-0.25)) / N  # 0.5 / N
    
    # Autoconvolution: (f * f)(t) = integral f(s) f(t-s) ds
    # For step functions, we can compute via FFT or direct convolution
    # Using FFT for efficiency
    n_fft = 2 * N + 1  # enough for linear convolution
    f_padded = np.zeros(n_fft)
    f_padded[:N] = heights
    
    # Linear convolution via FFT
    conv = np.fft.ifft(np.fft.fft(f_padded, n=n_fft) ** 2).real
    conv *= dx  # scale by dx for integration
    
    # The autoconvolution is supported on [-0.5, 0.5]
    # We want the maximum value
    return np.max(conv)

def _objective(params):
    """Objective: maximize the autoconvolution peak (we minimize its negative)."""
    heights = np.maximum(params, 0)  # enforce non-negativity via projection
    peak = _autoconv_peak(heights)
    return peak

def _project_nonnegative(x):
    """Project onto nonnegative orthant."""
    return np.maximum(x, 0)

def construct():
    """Construct a nonnegative step function with 600 equal intervals on [-1/4, 1/4]
    to minimize the normalized maximum of its autoconvolution."""
    start_time = time.time()
    
    # Initial guess: uniform distribution
    best_heights = np.ones(N)
    best_score = _autoconv_peak(best_heights)
    
    # Try a few different initializations and local optimizations
    # Use projected gradient descent with soft-max penalty approach
    
    # Strategy: Use Legendre polynomial expansion
    # Represent f(x) = sum_{k=0}^{K} c_k * P_k(x) where P_k are Legendre polynomials
    # on [-1/4, 1/4], scaled to be orthonormal
    
    # Actually, let's use a simpler approach: direct optimization with projected gradient
    
    # Multiple restarts with different initializations
    n_restarts = 5
    max_iter = 2000
    lr = 0.01
    
    for restart in range(n_restarts):
        if time.time() - start_time > TOTAL_TIME - SAFETY_MARGIN:
            break
        
        # Different initializations
        if restart == 0:
            # Uniform
            x = np.ones(N)
        elif restart == 1:
            # Slightly peaked in center
            x = np.ones(N)
            center = N // 2
            width = N // 4
            x[center - width:center + width] = 1.5
        elif restart == 2:
            # Bimodal
            x = np.ones(N)
            x[:N//4] = 1.5
            x[3*N//4:] = 1.5
            x[N//4:3*N//4] = 0.5
        elif restart == 3:
            # Random with mean 1
            np.random.seed(restart)
            x = np.abs(np.random.randn(N))
            x = x / np.mean(x)
        else:
            # Smooth random
            np.random.seed(restart + 100)
            x = np.cumsum(np.random.randn(N))
            x = np.abs(x)
            x = x / np.mean(x)
        
        # Projected gradient descent
        current_score = _autoconv_peak(np.maximum(x, 0))
        
        for it in range(max_iter):
            if time.time() - start_time > TOTAL_TIME - SAFETY_MARGIN:
                break
            
            # Compute gradient of autoconvolution peak w.r.t. heights
            # Use finite differences for simplicity, or analytic gradient
            # For speed, use a simple gradient estimate
            
            # Numerical gradient (sparse)
            eps = 1e-5
            grad = np.zeros(N)
            # Use random subset for gradient estimation to save time
            n_sample = min(50, N)
            indices = np.random.choice(N, n_sample, replace=False)
            for i in indices:
                xp = x.copy()
                xp[i] += eps
                xm = x.copy()
                xm[i] -= eps
                fp = _autoconv_peak(np.maximum(xp, 0))
                fm = _autoconv_peak(np.maximum(xm, 0))
                grad[i] = (fp - fm) / (2 * eps)
            
            # Scale gradient to full
            grad_full = grad / (n_sample / N)
            
            # Projected gradient step
            x_new = x - lr * grad_full
            x_new = _project_nonnegative(x_new)
            
            new_score = _autoconv_peak(x_new)
            if new_score < current_score:
                x = x_new
                current_score = new_score
                lr = min(lr * 1.05, 0.1)  # increase step size
            else:
                lr = lr * 0.9  # decrease step size
            
            if current_score < best_score:
                best_score = current_score
                best_heights = x.copy()
        
        # Local polish with L-BFGS-B on the best found so far
        if time.time() - start_time < TOTAL_TIME - SAFETY_MARGIN - 10:
            try:
                def obj_and_grad(x_opt):
                    heights = np.maximum(x_opt, 0)
                    val = _autoconv_peak(heights)
                    # Approximate gradient
                    eps = 1e-6
                    grad = np.zeros_like(x_opt)
                    for i in range(0, N, max(1, N // 200)):  # sparse gradient
                        xp = x_opt.copy()
                        xp[i] += eps
                        fm = x_opt.copy()
                        fm[i] -= eps
                        grad[i] = (_autoconv_peak(np.maximum(xp, 0)) - 
                                  _autoconv_peak(np.maximum(fm, 0))) / (2 * eps)
                    return val, grad
                
                res = minimize(obj_and_grad, x, method='L-BFGS-B', 
                              options={'maxiter': 100, 'maxls': 50})
                if res.success or res.fun < best_score:
                    x_final = _project_nonnegative(res.x)
                    final_score = _autoconv_peak(x_final)
                    if final_score < best_score:
                        best_score = final_score
                        best_heights = x_final.copy()
            except:
                pass
    
    # Final polish on the best result
    if time.time() - start_time < TOTAL_TIME - SAFETY_MARGIN - 5:
        try:
            def obj_only(x_opt):
                heights = np.maximum(x_opt, 0)
                return _autoconv_peak(heights)
            
            res = minimize(obj_only, best_heights, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            x_final = _project_nonnegative(res.x)
            final_score = _autoconv_peak(x_final)
            if final_score < best_score:
                best_score = final_score
                best_heights = x_final.copy()
        except:
            pass
    
    # Ensure non-negative and return as list
    best_heights = np.maximum(best_heights, 0)
    return list(best_heights)

# EVOLVE-BLOCK-END
