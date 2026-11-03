"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Convex optimization in the frequency domain using L-BFGS-B.
The problem is to minimize max(conv(f,f)) / (sum(f))^2 subject to f >= 0.
Since the objective is convex in f (max of convex functions divided by square of linear function is not strictly convex, but we can reformulate or use the fact that for fixed sum, minimizing max conv is convex), 
we can use a smooth approximation of the max function and optimize using L-BFGS-B which handles box constraints (f >= 0) natively.

We use a log-sum-exp smoothing for the max function to make it differentiable.
J(f) = (1/beta) * log(sum(exp(beta * conv(f,f)))) / (sum(f))^2
We fix the scale by constraining sum(f) = 1 (or N) via normalization, but L-BFGS-B doesn't handle equality constraints well directly. 
Instead, we can optimize over f with f >= 0 and let the normalization be part of the objective, or fix sum(f) by working in a simplex.
Actually, the score is scale-invariant. We can fix sum(f) = 1 and minimize max(conv(f,f)).
Let's fix sum(f) = 1. Then we minimize max(conv(f,f)) subject to f >= 0, sum(f) = 1.
This is a convex problem if we use a smooth max.
We can parameterize f to satisfy sum(f)=1 and f>=0? Or just use L-BFGS-B on f>=0 and normalize after? 
No, the gradient of the normalized objective is tricky.
Better: Fix sum(f) = 1. Use a softmax-like parameterization? No, that's for probability vectors.
We can just use L-BFGS-B on f with bounds (0, inf) and add a penalty for sum(f) != 1? 
Or, simpler: The objective is homogeneous of degree 0. We can fix sum(f) = 1.
Let's optimize g(f) = smooth_max(conv(f,f)) subject to f >= 0, sum(f) = 1.
This is a convex constraint.
L-BFGS-B can handle bounds. It cannot handle equality constraints directly.
We can reduce the dimensionality by fixing f[0] = 1 - sum(f[1:])? 
But f[0] must be >= 0.
Alternatively, we can just optimize the original objective J(f) = smooth_max(conv(f,f)) / (sum(f))^2 with f >= 0.
The gradient is well-defined. L-BFGS-B is a quasi-Newton method that should converge faster than FISTA for smooth problems.

Let's use scipy.optimize.minimize with method='L-BFGS-B'.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 600
TIME_LIMIT = 140  # Safety margin before 150s kill

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

def smooth_max_and_grad(f, beta):
    """
    Compute the smooth max of the autoconvolution and its gradient.
    Objective: M_beta(conv(f,f)) / (sum f)^2
    We will compute the gradient of this quantity.
    """
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf, np.zeros_like(f)
    
    conv = np.convolve(f, f, mode='full')
    
    # Smooth max: (1/beta) * log(sum(exp(beta * x)))
    # To avoid overflow, shift by max
    x = conv
    x_max = np.max(x)
    if x_max < 1e-15:
        return 0.0, np.zeros_like(f)
        
    shifted = beta * (x - x_max)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    # Value of smooth max
    M = x_max + (1.0 / beta) * np.log(sum_exp)
    
    # Gradient of M with respect to x
    # dM/dx = exp(beta*(x - x_max)) / sum_exp
    grad_M_dx = exp_shifted / sum_exp
    
    # Gradient of M with respect to f
    # M = h(conv(f,f))
    # dM/df[i] = sum_k grad_M_dx[k] * d(conv[k])/d(f[i])
    # d(conv[k])/d(f[i]) = f[k-i] + f[i-k] (with zero padding)
    # This is equivalent to 2 * (grad_M_dx * f)[i] in the convolution sense, 
    # but we need to be careful with the full convolution length.
    
    # Let's compute the gradient via FFT for speed.
    # We need the vector g such that g[i] = sum_k grad_M_dx[k] * (f[k-i] + f[i-k])
    # This is the cross-correlation of grad_M_dx and f, plus the correlation of f and grad_M_dx?
    # Actually, since conv is symmetric, the gradient is 2 * correlate(grad_M_dx, f, 'same')?
    # Let's verify with a small example or use the explicit formula via FFT.
    
    # Using FFT:
    # We want to compute the gradient of the quadratic form.
    # Let's use the property that the gradient of sum_k h[k] (f*f)[k] is 2 * (h * f) in the appropriate domain.
    # Here h = grad_M_dx.
    # We need the part of the convolution that corresponds to the indices of f.
    
    # Extend f to length 2N-1
    f_ext = np.zeros(2*N - 1)
    f_ext[:N] = f
    
    # grad_M_dx is length 2N-1
    # We want the gradient vector of length N.
    # dM/df[i] = sum_{k} grad_M_dx[k] * (f[k-i] + f[i-k])
    
    # Let's compute term1 = sum_k grad_M_dx[k] f[k-i]
    # This is the correlation of grad_M_dx and f.
    # np.correlate(grad_M_dx, f, 'full')
    # The index for lag i (where i=0..N-1) in the full correlation is i + (len(f)-1).
    # Wait, np.correlate(a, v) computes c_k = sum_n a[n+k] v[n].
    # We want sum_k grad_M_dx[k] f[k-i].
    # Let n = k-i => k = n+i.
    # sum_n grad_M_dx[n+i] f[n].
    # This matches c_i = sum_n a[n+i] v[n] with a=grad_M_dx, v=f.
    # So term1[i] = corr[i + (len(f)-1)]? 
    # The full correlation has length len(a) + len(v) - 1.
    # The lags range from -(len(v)-1) to len(a)-1.
    # The index for lag i is i + (len(v)-1).
    
    corr = np.correlate(grad_M_dx, f, mode='full')
    term1 = corr[N-1 : 2*N-1]
    
    # term2 = sum_k grad_M_dx[k] f[i-k]
    # This is the convolution of grad_M_dx and f at index i.
    # conv_gf = np.convolve(grad_M_dx, f, 'full')
    # term2[i] = conv_gf[i + (len(f)-1)]?
    # np.convolve(a, v) computes c_k = sum_n a[n] v[k-n].
    # We want sum_k grad_M_dx[k] f[i-k].
    # Let n = i-k => k = i-n.
    # sum_n grad_M_dx[i-n] f[n].
    # This is c_i = sum_n a[i-n] v[n] = sum_n a[n'] v[i-n'] with n'=i-n? No.
    # Standard convolution: c_k = sum_n a[n] v[k-n].
    # If we set k=i, c_i = sum_n a[n] v[i-n].
    # This matches sum_n grad_M_dx[n] f[i-n].
    # So term2 is just the convolution of grad_M_dx and f.
    
    conv_gf = np.convolve(grad_M_dx, f, mode='full')
    term2 = conv_gf[N-1 : 2*N-1]
    
    dM_df = term1 + term2
    
    # Objective J = M / s^2
    # dJ/df = (1/s^2) dM_df - (2 M / s^3) * 1
    
    J = M / (s * s)
    grad_J = dM_df / (s * s) - (2 * M / (s * s * s)) * np.ones(N)
    
    return J, grad_J

def construct():
    start_time = time.time()
    
    # Initial guess: constant function
    f0 = np.ones(N) / N  # Normalize to sum=1 to start in a reasonable scale
    
    best_f = f0.copy()
    best_score = compute_score(f0)
    
    # Try different betas for the smooth max approximation
    betas = [10.0, 50.0, 100.0]
    
    for beta in betas:
        if time.time() - start_time > TIME_LIMIT - 30:
            break
            
        def objective_and_grad(f):
            val, grad = smooth_max_and_grad(f, beta)
            if np.isinf(val):
                return 1e10, np.zeros_like(f)
            # Ensure gradient is finite
            if not np.all(np.isfinite(grad)):
                return 1e10, np.zeros_like(f)
            return val, grad
        
        # Use L-BFGS-B
        # Bounds: f >= 0
        bounds = [(0, None)] * N
        
        try:
            res = minimize(
                objective_and_grad, 
                f0, 
                method='L-BFGS-B', 
                jac=True,
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
            )
            
            if res.success or res.fun < best_score:
                f_res = res.x
                f_res = np.maximum(f_res, 0.0)
                # Normalize to sum=1 for consistency in comparison? 
                # The score is scale invariant, so we can just compute the score.
                score = compute_score(f_res)
                if score < best_score:
                    best_score = score
                    best_f = f_res.copy()
                    
        except Exception as e:
            pass
            
        # Local refinement with a smaller beta (closer to true max)
        if time.time() - start_time < TIME_LIMIT - 10:
            beta_ref = 1000.0
            def objective_and_grad_ref(f):
                val, grad = smooth_max_and_grad(f, beta_ref)
                if np.isinf(val):
                    return 1e10, np.zeros_like(f)
                if not np.all(np.isfinite(grad)):
                    return 1e10, np.zeros_like(f)
                return val, grad
                
            try:
                res_ref = minimize(
                    objective_and_grad_ref, 
                    best_f, 
                    method='L-BFGS-B', 
                    jac=True,
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
                )
                if res_ref.success or res_ref.fun < best_score:
                    f_ref = res_ref.x
                    f_ref = np.maximum(f_ref, 0.0)
                    score_ref = compute_score(f_ref)
                    if score_ref < best_score:
                        best_score = score_ref
                        best_f = f_ref.copy()
            except Exception:
                pass

    # Final check
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
