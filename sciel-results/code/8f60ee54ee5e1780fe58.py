"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

New approach: Differentiable optimization using a smooth approximation of the
maximum (Softmax) and scipy.optimize.minimize (L-BFGS-B or SLSQP).
This replaces the stochastic peak-shaving with deterministic gradient-based
optimization on a smoothed objective function.
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import convolve

N = 600
TIME_LIMIT = 115  # safety margin before 120s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    conv = np.convolve(f, f, mode='full')
    return np.max(conv) / (s * s)

def smooth_max_objective(f, beta=50.0, lambda_sum=1000.0):
    """
    Objective function to minimize.
    Uses a softmax approximation of the max of the autoconvolution.
    Also penalizes deviation of sum(f) from 1 to fix scale.
    """
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-12:
        return 1e10
    # Normalize for calculation stability, but we want to optimize shape
    # The score is scale invariant, so we can fix sum(f)=1.
    # However, L-BFGS-B works on the raw f.
    # Let's define the objective on f directly.
    
    conv = np.convolve(f, f, mode='full')
    
    # Softmax approximation of max(conv)
    # To avoid overflow, subtract max
    c_max = np.max(conv)
    # If c_max is 0, return 0
    if c_max < 1e-12:
        return 0.0
    
    # Stable log-sum-exp
    shifted = beta * (conv - c_max)
    # Clip shifted to avoid underflow issues in exp, though it's safe here
    log_sum_exp = np.log(np.sum(np.exp(shifted))) + beta * c_max
    
    # The term log_sum_exp is approx max(conv) * beta + log(N)
    # We want to minimize max(conv).
    # So we minimize log_sum_exp.
    
    # Penalty for sum(f) != 1
    # This helps the optimizer find a unique scale, although score is invariant.
    # It prevents the optimizer from drifting to very small or large f values
    # which might cause numerical issues in gradients.
    sum_penalty = lambda_sum * (s - 1.0) ** 2
    
    return log_sum_exp + sum_penalty

def gradient_smooth_max(f, beta=50.0, lambda_sum=1000.0):
    """
    Analytic gradient of the smooth max objective.
    """
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    
    conv = np.convolve(f, f, mode='full')
    c_max = np.max(conv)
    
    if c_max < 1e-12:
        return np.zeros_like(f)
    
    shifted = beta * (conv - c_max)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    # Gradient of log(sum(exp(beta * conv))) w.r.t. conv is beta * exp_shifted / sum_exp
    # Gradient of conv w.r.t. f is the convolution of f with itself (symmetric)
    # d(conv[k])/d(f[i]) = f[k-i] (with appropriate bounds)
    # So grad_f = beta * sum_k (exp_shifted[k] / sum_exp) * (d conv[k] / d f)
    
    # We can compute the weighted convolution directly
    # w = exp_shifted / sum_exp
    w = exp_shifted / sum_exp
    
    # grad_f[i] = beta * sum_k w[k] * f[k-i]
    # This is the convolution of w with f.
    # Note: conv(f, f) is full mode.
    # The gradient of the quadratic form f^T A f is 2 A f.
    # Here it's a bit more complex because of the softmax weighting.
    # Let's use numerical gradient if analytic is tricky, or derive carefully.
    
    # Actually, simpler: use finite differences if analytic is too slow to write correctly?
    # No, let's try to compute it.
    # J = log(sum_k exp(beta * conv_k))
    # dJ/df_i = (1/sum_exp) * sum_k exp(beta*conv_k) * beta * d(conv_k)/df_i
    # d(conv_k)/df_i = f_{k-i} (if 0 <= k-i < N, else 0)
    # So dJ/df_i = beta * sum_k (exp(beta*conv_k)/sum_exp) * f_{k-i}
    # This is beta * (w * f)_i where * is convolution.
    
    grad_conv_part = beta * np.convolve(w, f, mode='full')
    # We only care about the first N elements? No, f is length N.
    # The gradient vector has length N.
    # conv(w, f, 'full') has length 2N-1.
    # The term f_{k-i} is non-zero when k-i is in [0, N-1].
    # For a fixed i, the sum is over k.
    # Let's just take the slice [0:N] or center?
    # Let's verify with a small example or just use scipy's automatic differentiation?
    # For 600 variables, numerical gradient (600 evals) is too slow for many iterations.
    # Let's stick to analytic.
    
    # The convolution w * f in full mode:
    # g[j] = sum_k w[k] f[j-k]
    # We want dJ/df[i] = sum_k w[k] beta f[k-i]
    # Let m = k-i => k = m+i.
    # dJ/df[i] = beta sum_m w[m+i] f[m]
    # This is the cross-correlation of w and f.
    # np.correlate(w, f, mode='same') or 'full'?
    # Let's use np.convolve(w, f[::-1])? No.
    # Correlation: C[i] = sum_m w[m+i] f[m].
    # This is equivalent to convolving w with reversed f?
    # Let's just compute it explicitly for safety, it's O(N^2) which is 360k ops, very fast.
    
    # But wait, we can use FFT for speed if needed, but N=600 is small.
    # Let's compute the correlation directly.
    # w is length 2N-1. f is length N.
    # We need sum over k where f[k-i] is valid.
    
    grad_f = np.zeros(N)
    # w is indexed 0..2N-2.
    # f is indexed 0..N-1.
    # For each i in 0..N-1:
    #   sum over k such that 0 <= k-i < N => i <= k < i+N
    #   term is w[k] * f[k-i]
    
    # We can do this with a loop or vectorized slicing.
    # Vectorized:
    # Create a matrix? No.
    # Let's use the property that this is a convolution of w and f.
    # h = np.convolve(w, f, mode='full')
    # h[j] = sum_k w[k] f[j-k]
    # We want sum_k w[k] f[k-i]. Let l = k-i => k=l+i.
    # sum_l w[l+i] f[l]
    # This is the correlation of w and f at lag i.
    # np.correlate(w, f, mode='same')? 
    # np.correlate(a, v) computes sum a[m+k] v[m].
    # If we want sum w[k] f[k-i], this is correlate(w, f) with lag?
    # Let's just use the explicit loop for the gradient part, it's fast enough.
    
    # Actually, `np.convolve(w, f, mode='full')` gives h[j] = sum_k w[k] f[j-k].
    # We want g[i] = sum_k w[k] f[k-i].
    # Let j = k-i => k = j+i.
    # g[i] = sum_j w[j+i] f[j].
    # This is not a standard convolution of w and f directly in numpy's simple API without reversing.
    # It is the cross-correlation.
    
    # Let's just do the O(N^2) calculation.
    # w is length 2N-1.
    for i in range(N):
        # k ranges from i to i+N-1
        k_start = i
        k_end = i + N
        # w[k_start:k_end] and f[0:N]
        # Sum of element-wise product
        grad_f[i] = np.dot(w[k_start:k_end], f)
        
    grad_f *= beta
    
    # Add gradient of sum penalty
    # d/d f_i [ lambda (sum f - 1)^2 ] = 2 lambda (sum f - 1)
    grad_f += 2.0 * lambda_sum * (s - 1.0)
    
    return grad_f

def initial_guess_gaussian():
    """Create a Gaussian initial guess."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    sigma = 0.1
    f = np.exp(-x**2 / (2 * sigma**2))
    return f / np.sum(f)

def initial_guess_cosine():
    """Create a cosine bump initial guess."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    f = 0.5 * (1.0 + np.cos(2 * np.pi * x))
    return f / np.sum(f)

def construct():
    start_time = time.time()
    
    best_f = None
    best_score = np.inf
    
    # Try multiple initial guesses
    initial_guesses = [
        initial_guess_gaussian(),
        initial_guess_cosine(),
        np.ones(N) / N,
    ]
    
    beta = 50.0  # Softmax sharpness
    lambda_sum = 1000.0
    
    for guess in initial_guesses:
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        
        # Use L-BFGS-B with bounds [0, inf)
        # We fix sum(f)=1 via penalty, so bounds are just non-negativity.
        bounds = [(0, None) for _ in range(N)]
        
        # Initial point
        x0 = guess.copy()
        
        try:
            # Limit iterations and time
            # L-BFGS-B can be slow if not converged.
            # We'll run it for a fixed number of iterations or until time limit.
            
            # We can use a callback to check time
            def callback(xk):
                if time.time() - start_time > TIME_LIMIT - 5:
                    raise RuntimeError("Time limit reached")
                return False
            
            res = minimize(
                smooth_max_objective, 
                x0, 
                method='L-BFGS-B', 
                jac=gradient_smooth_max,
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6},
                args=(beta, lambda_sum),
                callback=callback
            )
            
            if res.success or res.status == 1: # 1 means max iter
                f_opt = res.x
                # Ensure non-negative
                f_opt = np.maximum(f_opt, 0.0)
                
                # Calculate actual score (not the smooth approx)
                score = compute_score(f_opt)
                
                if score < best_score:
                    best_score = score
                    best_f = f_opt
                    
        except RuntimeError:
            # Time limit reached, use the best point found so far?
            # L-BFGS-B doesn't easily expose the last point in callback without global state.
            # We'll just skip this guess if it times out.
            pass
        except Exception as e:
            # Catch any other errors
            pass
            
    # If no good result, fallback to a simple heuristic
    if best_f is None:
        # Use the cosine guess normalized
        f_cos = initial_guess_cosine()
        best_f = f_cos
        best_score = compute_score(best_f)
        
    # Final refinement with higher beta to sharpen the max approximation
    if time.time() - start_time < TIME_LIMIT - 10:
        beta_refine = 200.0
        x0 = best_f.copy()
        bounds = [(0, None) for _ in range(N)]
        
        try:
            def callback2(xk):
                if time.time() - start_time > TIME_LIMIT - 2:
                    raise RuntimeError("Time limit reached")
                return False
                
            res = minimize(
                smooth_max_objective, 
                x0, 
                method='L-BFGS-B', 
                jac=gradient_smooth_max,
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8},
                args=(beta_refine, lambda_sum),
                callback=callback2
            )
            
            if res.success or res.status == 1:
                f_opt = np.maximum(res.x, 0.0)
                score = compute_score(f_opt)
                if score < best_score:
                    best_score = score
                    best_f = f_opt
        except RuntimeError:
            pass
        except Exception:
            pass

    # Ensure nonnegative and return list
    best_f = np.maximum(best_f, 0.0)
    
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
