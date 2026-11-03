import numpy as np
import time
import sys
from scipy.optimize import minimize
from scipy.fft import dct, idct

N = 600
TIME_LIMIT = 140  # Safety margin for 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def smooth_max(x, beta=10.0):
    """
    Smooth approximation of max(x) using log-sum-exp.
    beta controls the sharpness. Higher beta -> closer to max.
    """
    max_x = np.max(x)
    # Avoid overflow
    x_scaled = x - max_x
    # Clip to avoid exp overflow, though x_scaled <= 0
    # exp(x_scaled) is safe
    return max_x + np.log(np.sum(np.exp(beta * x_scaled))) / beta

def spectral_to_time(coeffs):
    """
    Convert DCT coefficients to a non-negative time-domain step function.
    coeffs: array of length M (number of basis functions).
    We use DCT-II basis.
    To ensure non-negativity, we apply a softplus or simply clip.
    For optimization, we optimize raw coefficients and project.
    """
    M = len(coeffs)
    # Reconstruct f
    # idct expects the full length N, so pad coeffs with zeros
    coeffs_full = np.zeros(N)
    coeffs_full[:M] = coeffs
    
    f = idct(coeffs_full, type=2, norm='ortho')
    
    # Enforce non-negativity
    # A hard clip introduces non-differentiability at 0, which can hurt gradient methods.
    # However, L-BFGS-B can handle bounds if we formulated it differently.
    # Here we are optimizing coeffs unconstrained, so we must handle non-negativity internally.
    # Using softplus: f = alpha * log(1 + exp(f_raw/alpha))
    # But we want f to be the signal. 
    # Alternative: Optimize log(f) + epsilon? 
    # Let's stick to clipping for now, but use a very small epsilon to keep it smooth-ish 
    # or rely on the fact that the optimum likely has f>0 everywhere if we use enough basis functions.
    
    # Actually, a better parametric constraint: 
    # f = exp(g) where g is the sum of basis functions? 
    # Sum(f) = 1 constraint becomes hard.
    
    # Let's use the direct DCT reconstruction and clip.
    f = np.maximum(f, 0.0)
    
    # Normalize to sum 1
    s = np.sum(f)
    if s > 1e-15:
        f = f / s
    else:
        # Fallback to uniform
        f = np.ones(N) / N
        
    return f

def objective_and_grad(coeffs):
    """
    Objective: Smooth max of autoconvolution.
    Gradient: Approximated via finite differences or analytical?
    Analytical gradient of smooth_max(autoconv(f)) w.r.t coeffs is complex.
    We will use numerical gradients via L-BFGS-B's default if we don't provide jac,
    or provide a finite difference jac.
    Given N=600 and M~50, numerical gradient is 50 evaluations of objective.
    Objective eval involves FFT conv: O(N log N).
    This is fast.
    """
    f = spectral_to_time(coeffs)
    conv = np.convolve(f, f, mode='full')
    
    # We want to minimize the max.
    # Use smooth max for differentiability
    beta = 20.0
    val = smooth_max(conv, beta)
    
    # Return value and gradient
    # L-BFGS-B can compute gradient numerically.
    # But to be safe and fast, let's just return val.
    # If we provide jac, it must be accurate.
    # Let's try without jac first, L-BFGS-B approximates it.
    return val

def run_spectral_optimization(M, time_budget):
    """
    Run L-BFGS-B optimization on DCT coefficients.
    """
    start_time = time.time()
    
    # Initial guess: Low frequency dominant (smooth function)
    # This corresponds to a "bell-shaped" time function, which is a good start.
    coeffs0 = np.zeros(M)
    # Put some energy in the first few modes
    coeffs0[0] = 1.0
    coeffs0[1] = 0.5
    coeffs0[2] = 0.2
    
    # Bounds for coefficients to prevent explosion
    # The scale is handled by normalization in spectral_to_time,
    # so bounds can be loose.
    bounds = [(-10, 10)] * M
    
    best_f = None
    best_score = np.inf
    
    try:
        # Use L-BFGS-B
        # maxiter: limit iterations
        # ftol: tolerance on function value
        # gtol: tolerance on gradient
        res = minimize(
            objective_and_grad,
            coeffs0,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 200,
                'ftol': 1e-8,
                'gtol': 1e-5,
                'maxls': 20
            }
        )
        
        if res.fun < best_score:
            best_score = res.fun
            best_f = spectral_to_time(res.x)
            
        # Also check the initial point
        f0 = spectral_to_time(coeffs0)
        s0 = compute_score(f0)
        if s0 < best_score:
            best_score = s0
            best_f = f0
            
    except Exception as e:
        print(f"Spectral optimization failed: {e}")
        
    # If optimization didn't finish or failed, use the best found or initial
    if best_f is None:
        best_f = spectral_to_time(coeffs0)
        best_score = compute_score(best_f)
        
    return best_f, best_score

def local_refinement_spectral(f0, time_budget):
    """
    Refine the time-domain function using a gradient step on the smooth max.
    This acts as a polish.
    """
    f = f0.copy()
    start_time = time.time()
    best_score = compute_score(f)
    
    # We can try a simple projected gradient descent on the time domain
    # using the smooth max gradient.
    # grad_smooth_max(conv) w.r.t f:
    # d/d f_j smooth_max(conv) = sum_k w_k * d conv_k / d f_j
    # where w_k = exp(beta * (conv_k - max_conv)) / sum(exp(...))
    
    beta = 20.0
    step_size = 0.05
    
    for i in range(100):
        if time.time() - start_time > time_budget:
            break
            
        conv = np.convolve(f, f, mode='full')
        max_conv = np.max(conv)
        
        # Weights for smooth max
        exp_terms = np.exp(beta * (conv - max_conv))
        weights = exp_terms / np.sum(exp_terms)
        
        # Gradient of conv w.r.t f
        # conv_k = sum_i f_i f_{k-i}
        # d conv_k / d f_j = f_{k-j} + f_{j-k} (with bounds)
        # grad_f[j] = sum_k weights[k] * (f_{k-j} + f_{j-k})
        
        grad_f = np.zeros(N)
        # Vectorized computation of gradient
        # This is a cross-correlation of weights with f, plus reverse
        # grad_f = np.convolve(weights, f, mode='full') ... indices need care.
        
        # Let's do it explicitly for correctness, N=600 is small enough for a few iterations
        # Or use FFT?
        # d conv / d f is the convolution of f with the "impulse" response of the max operator?
        # No, it's the convolution of f with the "weights" of the max?
        
        # conv = f * f
        # d (f*f) / d f = 2 * (f * I) where I is the indicator of the max?
        # For smooth max:
        # d smooth_max(conv) / d f_j = 2 * sum_k weights[k] * (f_{k-j} + f_{j-k}) / 2 ?
        # Actually, d conv_k / d f_j = f_{k-j} + f_{j-k}
        # So grad_f[j] = sum_k weights[k] * (f_{k-j} + f_{j-k})
        
        # This is 2 * (weights * f) cross-correlated?
        # Let's use np.convolve on weights and f.
        # conv_w_f = np.convolve(weights, f, mode='full')
        # The term f_{k-j} corresponds to indices in conv_w_f.
        
        # Let's just do a simple loop for the gradient step, it's only 100 iters * 600 * 600 = 36M ops max, which is fine in Python if vectorized.
        
        # Vectorized gradient:
        # G[j] = sum_k W[k] f[k-j] + sum_k W[k] f[j-k]
        # Term 1: Correlation of W and f.
        # Term 2: Correlation of W and f reversed?
        
        # Actually, since f is real, and we are dealing with linear conv:
        # Let's compute H = np.convolve(weights, f, mode='full')
        # H[m] = sum_k weights[k] f[m-k]
        # We want sum_k weights[k] f[k-j]. Let m = k-j => k = m+j.
        # sum_m weights[m+j] f[m]. This is cross-correlation.
        
        # Simpler: Just use finite difference gradient for the polish?
        # Or just use the analytical gradient computed via FFT?
        # FFT approach:
        # grad_conv_k = f_{k-j} + f_{j-k}
        # This is effectively 2 * (f * delta) but weighted by W.
        
        # Let's stick to a robust numerical gradient for the polish step, 
        # as the spectral optimization already did the heavy lifting.
        
        # Numerical Gradient
        eps = 1e-5
        grad_num = np.zeros(N)
        for j in range(0, N, 2): # Step 2 to save time
            f_plus = f.copy()
            f_plus[j] += eps
            f_minus = f.copy()
            f_minus[j] -= eps
            # Ensure non-negative for eval
            f_plus = np.maximum(f_plus, 0)
            f_minus = np.maximum(f_minus, 0)
            # Normalize? No, compute_score normalizes.
            # But for gradient, we need to be consistent.
            # compute_score(f) = max(conv)/s^2
            # This is homogeneous of degree 0.
            # So gradient is well-defined on the simplex.
            
            val_plus = compute_score(f_plus)
            val_minus = compute_score(f_minus)
            grad_num[j] = (val_plus - val_minus) / (2 * eps)
            grad_num[j+1] = grad_num[j] # Approximation
            
        # Projected Gradient Descent
        # f_new = f - step_size * grad_num
        # Project to simplex (sum=1, f>=0)
        
        f_new = f - step_size * grad_num
        f_new = np.maximum(f_new, 0)
        s = np.sum(f_new)
        if s > 1e-15:
            f_new = f_new / s
        
        new_score = compute_score(f_new)
        if new_score < best_score:
            best_score = new_score
            f = f_new
            step_size *= 0.95
        else:
            step_size *= 0.5
            if step_size < 1e-6:
                break
                
    return f, best_score

def construct():
    start_time = time.time()
    
    # 1. Spectral Optimization
    # Use 30 DCT basis functions. This allows for moderately complex shapes.
    M = 30
    time_budget_spectral = 60.0
    
    f_spectral, score_spectral = run_spectral_optimization(M, time_budget_spectral)
    
    best_f = f_spectral
    best_score = score_spectral
    
    # 2. Local Refinement
    remaining_time = TIME_LIMIT - (time.time() - start_time) - 5.0
    
    if remaining_time > 5.0:
        # Polish with time-domain gradient descent
        f_refined, score_refined = local_refinement_spectral(best_f, remaining_time)
        
        if score_refined < best_score:
            best_f = f_refined
            best_score = score_refined
    
    # Final cleanup
    best_f = np.maximum(best_f, 0.0)
    s = np.sum(best_f)
    if s > 1e-15:
        best_f = best_f / s
        
    # Ensure it's a list of floats
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
