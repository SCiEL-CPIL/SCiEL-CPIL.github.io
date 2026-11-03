import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _autoconv_from_fft(x):
    """Compute autoconvolution using FFT (circular, then extract linear part)."""
    n = len(x)
    # Use FFT of size 2n to compute linear autoconvolution
    fft_size = 2 * n
    X = np.fft.fft(x, n=fft_size)
    autoconv_full = np.fft.ifft(X * X.conj()).real[:n]
    return autoconv_full

def _score_continuous(x):
    """Score for continuous relaxation: max|autoconv| / (sum(x))^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    autoconv = _autoconv_from_fft(x)
    max_c = np.max(np.abs(autoconv))
    return max_c / (s * s)

def _score_discrete(x):
    """Score for discrete ±1 sequence."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    c = np.correlate(x, x, mode='full')
    max_c = np.max(np.abs(c))
    return max_c / (s * s)

def _grad_score(x):
    """Compute gradient of the score with respect to x.
    
    We minimize f(x) = max_k |C[k]| / (sum(x))^2
    
    Using subgradient for max|C[k]| and chain rule.
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return np.zeros_like(x)
    
    n = len(x)
    fft_size = 2 * n
    X = np.fft.fft(x, n=fft_size)
    
    # Autoconvolution in frequency domain: C = ifft(|X|^2)
    # C[k] = sum_j x[j] x[j+k] for linear autoconv
    # Actually for circular: C_circ[k] = ifft(X * conj(X))[k]
    # For linear, we use zero-padding
    
    # Compute linear autoconvolution
    C = np.fft.ifft(X * X.conj()).real[:n]
    
    # Find which lags have max absolute value
    abs_C = np.abs(C)
    max_val = np.max(abs_C)
    
    if max_val < 1e-10:
        return np.zeros_like(x)
    
    # Subgradient of max|C[k]|: weight on the argmax components
    # For simplicity, use uniform weight over all near-max components
    threshold = max_val * 0.95
    mask = abs_C >= threshold
    num_max = np.sum(mask)
    if num_max == 0:
        mask = np.argmax(abs_C)
        mask = np.array([1 if i == np.argmax(abs_C) else 0 for i in range(n)], dtype=float)
        num_max = 1
    
    weights = np.where(mask, 1.0 / num_max, 0.0)
    
    # d(max|C|)/dx = sum_k weights[k] * sign(C[k]) * dC[k]/dx
    # dC[k]/dx[j] = x[j+k] + x[j-k] (with appropriate bounds for linear conv)
    # This is the gradient of the autoconvolution
    
    # Compute gradient of C with respect to x
    # C[k] = sum_j x[j] x[j+k] (for k >= 0)
    # dC[k]/dx[j] = x[j+k] + x[j-k] if in bounds
    
    # Using FFT: C = ifft(X * conj(X))
    # dC/dx[j] involves the inverse FFT of derivative of |X|^2
    
    # Simpler: use finite differences or analytic gradient
    # dC[k]/dx[j] = sum over pairs where one index is j
    
    # Let's compute the gradient directly
    grad_C = np.zeros_like(x)
    for k in range(n):
        if not mask[k]:
            continue
        sign_C = np.sign(C[k])
        if sign_C == 0:
            continue
        # dC[k]/dx[j] for each j
        # C[k] = sum_j x[j]*x[j+k] for k>=0
        # For linear autoconv with zero padding:
        # C[k] = sum_{j=0}^{N-1-k} x[j] * x[j+k]
        # dC[k]/dx[j] = x[j+k] (if j+k < N) + x[j-k] (if j-k >= 0, for k>0)
        # Actually for k>=0: C[k] = sum_{j=0}^{N-1-k} x[j]*x[j+k]
        # dC[k]/dx[j] = x[j+k] for j <= N-1-k, and x[j-k] for j >= k (when considering C[-k])
        
        # Let me use the full autoconvolution approach
        pass
    
    # Simpler approach: use the fact that autoconv is quadratic
    # C = autoconv(x)
    # dC/dx is a matrix (Toeplitz/Hankel structure)
    
    # For now, use a simpler gradient: perturb and estimate
    # This is expensive but N=400 is small
    
    eps = 1e-7
    grad = np.zeros_like(x)
    for j in range(n):
        x_plus = x.copy()
        x_plus[j] += eps
        x_minus = x.copy()
        x_minus[j] -= eps
        
        C_plus = _autoconv_from_fft(x_plus)
        C_minus = _autoconv_from_fft(x_minus)
        
        # Subgradient of max|C|
        abs_plus = np.abs(C_plus)
        abs_minus = np.abs(C_minus)
        
        max_plus = np.max(abs_plus)
        max_minus = np.max(abs_minus)
        
        # Approximate derivative of max|C|
        # This is tricky due to max and abs
        # Use: d(max|C|)/dx[j] ≈ (max|C_plus| - max|C_minus|) / (2*eps)
        dmax_dc = (max_plus - max_minus) / (2 * eps)
        
        # d(score)/dx[j] = dmax_dc / s^2 - max_C * 2 * d(s)/dx[j] / s^3
        # d(s)/dx[j] = 1
        d_score = dmax_dc / (s * s) - max_val * 2 * 1.0 / (s * s * s)
        
        grad[j] = d_score
    
    return grad

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, use 120s margin
    
    # Start with all ones
    x_best = np.ones(N, dtype=float)
    best_score = _score_discrete(x_best)
    
    # Try several random initializations for continuous relaxation
    np.random.seed(42)
    
    for trial in range(10):
        if time.time() > deadline - 20:  # Leave 20s for final processing
            break
        
        # Random continuous initialization
        x_cont = np.random.uniform(-1, 1, N)
        
        # Optimize using L-BFGS-B
        def objective(x_cont):
            if time.time() > deadline - 10:
                return 1e10
            return _score_continuous(x_cont)
        
        def gradient(x_cont):
            if time.time() > deadline - 10:
                return np.zeros_like(x_cont)
            return _grad_score(x_cont)
        
        try:
            result = minimize(
                objective,
                x_cont,
                method='L-BFGS-B',
                jac=gradient,
                bounds=[(-1, 1)] * N,
                options={'maxiter': 100, 'maxfun': 500, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            # Round to ±1
            x_rounded = np.sign(result.x)
            x_rounded[x_rounded == 0] = 1  # Handle zeros
            
            score_rounded = _score_discrete(x_rounded)
            
            if score_rounded < best_score:
                best_score = score_rounded
                x_best = x_rounded.copy()
                
        except Exception as e:
            pass
    
    # Local search refinement on the best discrete solution
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                old_val = x_best[i]
                x_best[i] = -old_val
                new_score = _score_discrete(x_best)
                if new_score < best_score:
                    best_score = new_score
                    improved = True
                else:
                    x_best[i] = old_val
    
    return x_best
