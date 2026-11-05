"""
Autocorrelation inequality optimizer.
Minimizes |2*N*max(conv(h,h))/sum(h)^2| over N=400 step heights.
Strategy:
1. Project solution to a low-dimensional Chebyshev basis.
2. Use Differential Evolution for global search in this low-dim space.
3. Refine the best solution in full N-dimensional space using L-BFGS-B 
   with a smooth Log-Sum-Exp surrogate for the max operation.
"""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 400

def construct():
    """
    Construct a sequence of N step heights minimizing:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h)**2)
    """
    start_time = time.time()
    DEADLINE = 145.0  # 5s safety margin before 150s kill

    best_value = np.inf
    best_h = np.ones(N).astype(float)

    # --- 1. Precompute Basis and Utilities ---
    
    # Chebyshev basis for low-dimensional representation
    # We use K=12 to capture enough structure without overfitting noise
    K = 12
    i_arr = np.arange(N, dtype=float)
    x_norm = 2.0 * i_arr / (N - 1) - 1.0
    # Clip to avoid arccos domain errors due to floating point
    x_safe = np.clip(x_norm, -1.0, 1.0)
    cheb_basis = np.zeros((K, N))
    for k in range(K):
        cheb_basis[k] = np.cos(k * np.arccos(x_safe))

    def coeffs_to_h(coeffs):
        """Map low-dim coefficients to N-dim heights."""
        return coeffs @ cheb_basis

    # FFT setup for fast convolution
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2
        
    # FFT setup for fast cross-correlation (needed for gradient)
    # We need to compute sum_m weights[i+m]*h[m]
    # This is a sliding dot product. 
    # Using FFT: convolve(weights, h[::-1]) 
    # Length of result: (2N-1) + (N-1) = 3N-2.
    fft_size_cc = 1
    while fft_size_cc < 3 * N - 2:
        fft_size_cc *= 2

    def fast_convolve(h):
        """Fast auto-convolution via FFT."""
        H = np.fft.rfft(h, n=fft_size)
        conv = np.fft.irfft(H * H, n=fft_size)
        return conv[:2 * N - 1]

    def fast_grad_lse(weights, h):
        """
        Compute grad_lse[i] = 2 * sum_{m=0}^{N-1} weights[i+m] * h[m]
        using FFT-based cross-correlation.
        """
        # We want C[i] = sum_m weights[i+m] * h[m]
        # Let g = h[::-1]. Then conv(weights, g)[i + N - 1] = sum_m weights[i+m] * h[m]
        h_rev = h[::-1]
        H_rev = np.fft.rfft(h_rev, n=fft_size_cc)
        W = np.fft.rfft(weights, n=fft_size_cc)
        conv_result = np.fft.irfft(H_rev * W, n=fft_size_cc)
        
        # The relevant part starts at index N-1
        # We need N values: i=0 to N-1
        grad_part = conv_result[N-1 : 2*N-1]
        
        return 2.0 * grad_part

    def evaluate(h):
        """Exact objective."""
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf, False
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        valid = np.all(np.isfinite(h)) and abs(s) > 1e-12
        return val, valid

    val_seed, _ = evaluate(best_h)
    best_value = val_seed

    # --- 2. Smooth Surrogate for Local Optimization ---
    
    BETA = 50.0  # Sharpness of max approximation. Higher is closer to max, harder to optimize.

    def smooth_objective_and_grad(x, beta=BETA):
        """
        Compute objective and gradient for the smooth surrogate:
        f(h) = | 2*N * LSE(conv(h,h)) / sum(h)^2 |
        """
        h = x
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10, np.zeros_like(h)
        
        conv = fast_convolve(h)
        max_c = np.max(conv)
        
        # Numerically stable LSE
        shifted = conv - max_c
        max_shifted = np.max(shifted)
        if max_shifted > 500:
            return 1e10, np.zeros_like(h)
        
        scaled = beta * (shifted - max_shifted)
        exp_scaled = np.exp(scaled)
        sum_exp = np.sum(exp_scaled)
        if sum_exp < 1e-300:
            return 1e10, np.zeros_like(h)
        
        lse = max_c + max_shifted + np.log(sum_exp) / beta
        weights = exp_scaled / sum_exp  # Normalized weights for gradient
        
        f_val_raw = 2.0 * N * lse / (s * s)
        
        if not np.isfinite(f_val_raw):
            return 1e10, np.zeros_like(h)
        
        # Gradient of lse
        grad_lse = fast_grad_lse(weights, h)
        
        # Gradient of f_raw = 2*N*lse/s^2
        # df/dh = 2*N * (grad_lse / s^2 - 2*lse * s / s^4 * ones)
        #        = 2*N * grad_lse / s^2 - 4*N*lse / s^3
        df_dh = 2.0 * N * grad_lse / (s * s) - 4.0 * N * lse / (s * s * s)
        
        # Objective is |f|
        if f_val_raw >= 0:
            grad = df_dh
        else:
            grad = -df_dh
            
        return abs(f_val_raw), grad

    def smooth_objective(x):
        val, _ = smooth_objective_and_grad(x)
        return val

    def smooth_grad(x):
        _, grad = smooth_objective_and_grad(x)
        return grad

    # --- 3. Global Search in Low-Dimensional Space ---
    
    def low_dim_objective(coeffs):
        h = coeffs_to_h(coeffs)
        val, valid = evaluate(h)
        if not valid:
            return 1e10
        return val

    bounds = [(-5.0, 5.0)] * K
    
    # Use a time limit for DE to ensure we leave time for polishing
    de_time_limit = 60.0 
    de_start = time.time()
    
    try:
        # Differential Evolution is stochastic, so we run it with a reasonable popsize
        # maxiter is high, but we rely on the callback to stop if time runs out
        de_result = differential_evolution(
            low_dim_objective,
            bounds=bounds,
            seed=42,
            maxiter=100,
            popsize=20,
            tol=1e-6,
            callback=lambda xk, convergence: (time.time() - de_start > de_time_limit)
        )
        
        if de_result.success or de_result.fun < best_value:
            best_coeffs = de_result.x
            h_de = coeffs_to_h(best_coeffs)
            val_de, valid_de = evaluate(h_de)
            if valid_de:
                best_value = val_de
                best_h = h_de.copy()
                print(f"DE found: {val_de:.6f}")
    except Exception as e:
        print(f"DE failed: {e}")

    # --- 4. Local Refinement in Full N-Dimensional Space ---
    
    # We perform local refinement from the best solution found by DE.
    # We might also try refining from the seed if DE didn't improve much.
    
    candidates = [best_h]
    # Add seed as a fallback candidate
    if not np.allclose(best_h, np.ones(N)):
        candidates.append(np.ones(N).astype(float))
        
    # Add a few random low-freq perturbations to escape potential local minima
    rng = np.random.RandomState(42)
    for _ in range(3):
        noise = rng.normal(0, 0.1, K)
        h_pert = coeffs_to_h(de_result.x + noise)
        candidates.append(h_pert)

    for h_init in candidates:
        if time.time() - start_time > DEADLINE - 10:
            break
            
        s = np.sum(h_init)
        if abs(s) < 1e-10:
            h_init = h_init + 0.1
            
        try:
            # High precision polish
            result = minimize(
                smooth_objective,
                h_init,
                jac=smooth_grad,
                method='L-BFGS-B',
                options={
                    'maxiter': 500,
                    'maxfun': 1000,
                    'gtol': 1e-8,
                    'ftol': 1e-12
                }
            )
            
            h_opt = result.x
            val_opt, valid = evaluate(h_opt)
            if valid and val_opt < best_value:
                best_value = val_opt
                best_h = h_opt.copy()
                
        except Exception:
            pass

    # --- 5. Final Exact Check and Return ---
    
    # One last check with the exact objective to ensure we didn't drift
    # due to the smooth approximation
    val_final, valid_final = evaluate(best_h)
    if valid_final and val_final < best_value:
        best_value = val_final
        
    return best_h
