import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT for speed."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return np.inf
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _smooth_max_and_grad(x, beta=100.0):
    """
    Compute smooth approximation of max|autoconv| and its gradient.
    Uses log-sum-exp for smooth max.
    Returns (smooth_max, gradient_wrt_x)
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    
    # Handle both positive and negative peaks
    # max|C| = max(max(C), -min(C))
    # Use log-sum-exp for smooth max
    beta_safe = max(beta, 1e-6)
    
    # For max(C): log-sum-exp
    C_pos = C
    C_neg = -C
    
    # Numerically stable log-sum-exp
    max_pos = np.max(C_pos)
    max_neg = np.max(C_neg)
    
    # Combine: max(max_pos, max_neg) smoothly
    # Use log-sum-exp on [max_pos, max_neg]
    combined_max = max(max_pos, max_neg)
    if combined_max > 1e10:
        # Overflow protection
        return combined_max, np.zeros(N)
    
    exp_pos = np.exp(beta_safe * (C_pos - combined_max))
    exp_neg = np.exp(beta_safe * (C_neg - combined_max))
    
    sum_exp = np.sum(exp_pos) + np.sum(exp_neg)
    if sum_exp < 1e-300:
        return combined_max, np.zeros(N)
    
    # Smooth max value
    smooth_val = combined_max + np.log(sum_exp) / beta_safe
    
    # Gradient: d/dx_j smooth_max
    # d/dx_j C[k] = x[j] + x[k-j] (for valid indices)
    # This is complex to compute directly, use FFT-based approach
    # 
    # Alternative: use the fact that C = IFFT(|FFT(x)|^2)
    # dC/dx_j involves circular convolution structure
    # 
    # Simpler approach: use finite differences or just return the value
    # and let L-BFGS handle it via numerical gradients
    
    # For now, return just the value and let scipy use numerical gradients
    # But we can compute the gradient more efficiently:
    # 
    # C[k] = sum_i x[i] * x[i+k]
    # dC[k]/dx[j] = x[j+k] + x[j-k] (with boundary handling)
    # 
    # Using FFT: if C = IFFT(X * X), then dC/dx involves...
    # Actually, let's use a different approach: compute gradient via
    # the relationship between autoconv and cross-correlation
    
    # Gradient of sum_k w[k] * C[k] where w[k] are weights from log-sum-exp
    # dC[k]/dx[j] = sum of terms where j appears in the product
    # C[k] = sum_i x[i] x[i+k]
    # x[j] appears when i=j (gives x[j+k]) or i+k=j (gives x[j-k])
    # So dC[k]/dx[j] = x[j+k] + x[j-k] (with bounds)
    
    # Weighted sum: grad[j] = sum_k w[k] * (x[j+k] + x[j-k])
    # where w[k] = exp(beta*C[k])/sum or similar
    
    # This is still O(N^2). Let's use FFT trick:
    # The gradient of max_k C[k] w.r.t x is related to the "active" lags
    # 
    # For simplicity and speed, let's just return the smooth value
    # and use numerical gradients in L-BFGS (slower but simpler)
    
    return smooth_val, None

def construct():
    START_TIME = time.time()
    TIME_LIMIT = 145  # safety margin before 150s kill
    
    best_x = np.ones(N)
    best_val = _objective(best_x)  # baseline: 1/N = 0.0025
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Try structured initializations
    initializations = [
        np.ones(N),
        np.linspace(2.0, 1.0, N),
        np.linspace(1.0, 2.0, N),
    ]
    for r in [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5]:
        initializations.append(r ** np.arange(N))
    for r in [1.05, 1.1, 1.15, 1.2, 1.3, 1.5, 2.0]:
        initializations.append(r ** np.arange(N))
    
    t = np.linspace(0, 1, N)
    initializations.append(1.0 + 0.5 * np.cos(np.pi * t))
    initializations.append(1.0 + np.cos(np.pi * t))
    initializations.append(1.0 + t**2)
    initializations.append(1.0 + 4.0 * (t - 0.5)**2)
    initializations.append(np.exp(2.0 * t))
    initializations.append(np.exp(-2.0 * t))
    initializations.append(1.0 + 0.3 * np.sin(2 * np.pi * t))
    initializations.append(1.0 + 0.5 * np.sin(np.pi * t))
    
    for x0 in initializations:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        val = _objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
    
    # Phase 2: L-BFGS optimization with smooth objective
    # Use a smooth approximation: sum of exp(beta * |C[k]|) / N as proxy for max|C|
    # This is differentiable and captures the max behavior
    
    def smooth_objective_and_grad(x, beta=50.0):
        """
        Smooth objective: (1/N) * sum_k exp(beta * |C[k]|) / (sum x)^2
        This approximates max|C| / (sum x)^2 when beta is large.
        Returns (value, gradient)
        """
        s = np.sum(x)
        if abs(s) < 1e-10:
            return 1e10, np.zeros(N)
        
        L = 2 * N - 1
        L_fft = 1
        while L_fft < L:
            L_fft *= 2
        
        X = np.fft.rfft(x, n=L_fft)
        C = np.fft.irfft(X * X, n=L_fft)[:L]
        
        # Compute exp(beta * |C|) stably
        max_abs_C = np.max(np.abs(C))
        if max_abs_C > 1e10:
            return 1e10, np.zeros(N)
        
        # Use log-sum-exp for stability
        abs_C = np.abs(C)
        max_abs = np.max(abs_C)
        
        # exp(beta * abs_C) can overflow, so use log-sum-exp
        # log(sum exp(beta * abs_C)) = beta * max_abs + log(sum exp(beta * (abs_C - max_abs)))
        shifted = beta * (abs_C - max_abs)
        # Clip to avoid underflow
        shifted = np.clip(shifted, -50, 50)
        log_sum = max_abs * beta + np.log(np.sum(np.exp(shifted)))
        
        # Value: exp(log_sum) / (s^2) -- but this can be huge
        # Better: work in log space or scale
        # Actually, let's just compute the value directly with clipping
        # For the gradient, we need d/dx of this quantity
        
        # Simpler approach: use a moderate beta and compute directly
        # with gradient via FFT
        beta_safe = min(beta, 20.0)  # Keep beta moderate for stability
        
        exp_vals = np.exp(beta_safe * abs_C)
        sum_exp = np.sum(exp_vals)
        
        if sum_exp < 1e-100:
            return 1e10, np.zeros(N)
        
        value = sum_exp / (s * s)
        
        # Gradient:
        # d/dx_j [sum_k exp(beta * |C[k]|) / s^2]
        # = [sum_k exp(beta*|C[k]|) * beta * sign(C[k]) * dC[k]/dx_j * s^2 
        #    - sum_k exp(beta*|C[k]|) * 2*s * (dC/dx_j part for s)] / s^4
        # 
        # dC[k]/dx[j] = x[j+k] + x[j-k] (with boundary checks)
        # ds/dx[j] = 1
        # 
        # This is O(N^2) per gradient evaluation. Let's use FFT trick:
        # 
        # Define w[k] = exp(beta * |C[k]|) * sign(C[k]) * beta / sum_exp
        # Then d/dx_j sum_k w[k] * C[k] = sum_k w[k] * (x[j+k] + x[j-k])
        # 
        # This is a convolution! Can compute via FFT.
        
        # Weights for the gradient
        signs = np.sign(C)
        # Where C is zero, sign is 0, which is fine
        w = exp_vals * signs * beta_safe / sum_exp
        
        # grad_from_C[j] = sum_k w[k] * (x[j+k] + x[j-k])
        # This is: (w * x)[j] + (w * x)[j] with appropriate shifts
        # Actually: sum_k w[k] * x[j+k] is cross-correlation
        # sum_k w[k] * x[j-k] is also related
        
        # Let's compute: A[j] = sum_k w[k] * x[j+k]
        # This is (w * x)[j] in circular convolution sense, but we need linear
        # 
        # Use FFT: A = IFFT(FFT(w) * FFT(x)) but with proper alignment
        # 
        # For linear convolution/correlation, pad to length >= N + L - 1
        
        L_conv = N + L - 1
        L_fft2 = 1
        while L_fft2 < L_conv:
            L_fft2 *= 2
        
        W = np.fft.rfft(w, n=L_fft2)
        Xf = np.fft.rfft(x, n=L_fft2)
        
        # A[j] = sum_k w[k] * x[j+k]
        # In circular conv: (w * x)[j] = sum_k w[k] * x[(j-k) mod L]
        # We want sum_k w[k] * x[j+k], which is (w * x_flipped)[j]
        # or equivalently cross-correlation
        
        # Let's define: corr[j] = sum_k w[k] * x[j+k]
        # This is like cross-correlation of w and x
        
        # Using FFT: corr = IFFT(conj(FFT(w)) * FFT(x)) with proper handling
        # Actually, cross-correlation R[j] = sum_k w[k] * x[j+k]
        # R = IFFT(conj(W) * Xf) gives circular cross-correlation
        
        R1 = np.fft.irfft(np.conj(W) * Xf, n=L_fft2)[:L_conv]
        # R1[j] = sum_k w[k] * x[(j-k) mod L_fft2]
        # We want sum_k w[k] * x[j+k], which is R1 with flipped w
        # Or: define w_flip[k] = w[-k] and compute (w_flip * x)[j]
        
        # Simpler: just compute it directly for the gradient
        # O(N^2) is 400^2 = 160000, which is fast enough
        
        grad_C = np.zeros(N)
        for j in range(N):
            # dC[k]/dx[j] = x[j+k] + x[j-k] for valid k
            total = 0.0
            for k in range(L):
                # x[j+k] valid if 0 <= j+k < N
                if 0 <= j + k < N:
                    total += w[k] * x[j + k]
                # x[j-k] valid if 0 <= j-k < N
                if 0 <= j - k < N:
                    total += w[k] * x[j - k]
            grad_C[j] = total
        
        # Full gradient:
        # d/dx_j [sum_exp / s^2] = (grad_C[j] * s^2 - sum_exp * 2*s) / s^4
        # = grad_C[j] / s^2 - 2 * sum_exp / s^3
        
        grad = grad_C / (s * s) - 2 * sum_exp / (s ** 3)
        
        return value, grad
    
    # Run L-BFGS from best initial point
    if time.time() - START_TIME < TIME_LIMIT - 5:
        try:
            result = minimize(
                smooth_objective_and_grad,
                best_x,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            if result.success or result.fun < best_val:
                # Evaluate the true objective on the result
                candidate_x = result.x
                candidate_val = _objective(candidate_x)
                if candidate_val < best_val:
                    best_val = candidate_val
                    best_x = candidate_x.copy()
        except Exception:
            pass
    
    # Phase 3: Coordinate descent polish on peak lags
    if time.time() - START_TIME < TIME_LIMIT - 2:
        # Identify the lag with maximum |autoconv|
        for polish_iter in range(200):
            if time.time() - START_TIME > TIME_LIMIT - 1:
                break
            
            ac = _autoconv_fft(best_x)
            s = np.sum(best_x)
            if abs(s) < 1e-10:
                break
            
            # Find the lag with max |ac|
            peak_lag = np.argmax(np.abs(ac))
            peak_val = ac[peak_lag]
            
            # Try small perturbations to reduce this peak
            # Perturb indices that contribute to this lag
            # C[peak_lag] = sum_i x[i] * x[i+peak_lag]
            # To reduce |C[peak_lag]|, we can adjust pairs (i, i+peak_lag)
            
            improved = False
            for _ in range(100):
                if time.time() - START_TIME > TIME_LIMIT - 1:
                    break
                
                # Random pair of indices that affect this lag
                valid_i = np.arange(max(0, -peak_lag), min(N, N - peak_lag))
                if len(valid_i) == 0:
                    break
                i = rng.integers(0, len(valid_i))
                i1 = valid_i[i]
                i2 = i1 + peak_lag
                if i2 >= N:
                    continue
                
                # Try adjusting x[i1] and x[i2]
                delta = rng.normal(0, 0.05)
                x_trial = best_x.copy()
                x_trial[i1] += delta
                x_trial[i2] -= delta  # Keep sum roughly constant
                
                val = _objective(x_trial)
                if val < best_val - 1e-10:
                    best_val = val
                    best_x = x_trial
                    improved = True
            
            if not improved:
                # Try a broader perturbation
                perturbation = rng.normal(0, 0.02, size=N)
                x_trial = best_x + perturbation
                val = _objective(x_trial)
                if val < best_val:
                    best_val = val
                    best_x = x_trial
                else:
                    break
    
    return best_x
