"""Optimized step function for minimizing normalized autoconvolution peak."""
import numpy as np
import time

N = 600
DELTA = (1.0/4.0 - (-1.0/4.0)) / N  # = 1/600
TOTAL_LEN = 1.0/2.0
DEADLINE = 110.0  # seconds, safety margin from 120s limit

def _score(heights):
    """Compute the normalized maximum of the autoconvolution.
    
    For a step function with heights h_i on equal intervals:
    autoconvolution at lag k: c_k = sum_i h_i * h_{i+k} * DELTA  (for appropriate k)
    
    Normalization: divide by ||f||_2^2 = sum_i h_i^2 * DELTA
    
    The maximum of c_k is at k=0 by Cauchy-Schwarz for nonnegative functions,
    so we compute c_0 / ||f||_2^2 = (sum h_i^2 * DELTA) / (sum h_i^2 * DELTA) = 1?
    
    Wait, let me reconsider. The autoconvolution (f*f)(t) = integral f(x)f(x-t) dx.
    For step functions with height h_i on interval i:
    (f*f)(0) = integral f(x)^2 dx = sum h_i^2 * DELTA = ||f||_2^2
    
    So the peak IS at lag 0 and equals ||f||_2^2, giving ratio 1 always?
    
    Actually no - the maximum of autoconvolution is NOT necessarily at lag 0 for
    non-uniform functions. Let me compute it properly.
    
    (f*f)(k*DELTA) = sum_i h_i * h_{i+k} * DELTA for lag k.
    
    The peak could be off-center. We need max over all lags.
    """
    h = np.array(heights, dtype=np.float64)
    # Full autoconvolution
    ac = np.correlate(h, h, mode='full') * DELTA
    # Peak
    peak = np.max(ac)
    # Normalization: ||f||_2^2
    norm2 = np.sum(h**2) * DELTA
    if norm2 < 1e-30:
        return float('inf')
    return peak / norm2

def construct():
    """Construct optimal nonnegative step heights."""
    start_time = time.time()
    
    # Start with uniform (known to give score 1.0)
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    # Try some structured perturbations using Fourier domain insights
    # The key insight: minimize max_lag |autoconv_lag| / ||f||_2^2
    # For nonnegative f, autoconv is nonnegative, peak at lag 0 typically.
    # But for functions with oscillating structure, off-peak sidelobes matter.
    
    # Strategy: Use a cosine-based shape that minimizes sidelobes
    # f(x) proportional to (1 + cos(pi*x / (1/4)))^p on [-1/4, 1/4]
    # This creates a window-like shape.
    
    candidates = [best_h.copy()]
    
    # Generate various window-like candidates
    x = np.linspace(-1.0/4.0, 1.0/4.0, N, endpoint=False) + DELTA/2
    x_mid = x + DELTA/2
    
    for p in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]:
        # Raised cosine to power p
        base = 0.5 * (1.0 + np.cos(2.0 * np.pi * x_mid / TOTAL_LEN))
        # base is 0 at edges, 1 at center
        h = np.maximum(base, 1e-10) ** p
        h = h / np.max(h)  # normalize to max 1
        candidates.append(h)
    
    # Hann window
    h_hann = 0.5 * (1.0 + np.cos(2.0 * np.pi * x_mid / TOTAL_LEN))
    candidates.append(h_hann / np.max(h_hann))
    
    # Hamming window
    h_ham = 0.54 + 0.46 * np.cos(2.0 * np.pi * x_mid / TOTAL_LEN)
    candidates.append(h_ham / np.max(h_ham))
    
    # Blackman window
    h_bl = 0.42 + 0.5 * np.cos(2.0 * np.pi * x_mid / TOTAL_LEN) + 0.08 * np.cos(4.0 * np.pi * x_mid / TOTAL_LEN)
    h_bl = np.maximum(h_bl, 0)
    if np.max(h_bl) > 1e-10:
        candidates.append(h_bl / np.max(h_bl))
    
    # Gaussian
    for sigma in [0.1, 0.15, 0.2, 0.25, 0.3]:
        h_g = np.exp(-x_mid**2 / (2 * sigma**2))
        candidates.append(h_g / np.max(h_g))
    
    # Power law: (1 - |2x|)^p
    for p in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
        base = np.maximum(1.0 - 2.0 * np.abs(x_mid), 0.0)
        h = base ** p
        if np.max(h) > 1e-10:
            candidates.append(h / np.max(h))
    
    # Evaluate all candidates
    for h in candidates:
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # Local optimization via projected gradient / subgradient on log heights
    # Use scipy if available for SLSQP or trust-constr
    try:
        from scipy.optimize import minimize
        import scipy.optimize as sopt
        
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 30:
            # Not enough time for optimization
            return [float(max(v, 0.0)) for v in best_h]
        
        # Objective: minimize max(autoconv) / sum(h^2 * DELTA)
        # Reparameterize: h_i = exp(y_i) to enforce nonnegativity
        # But this can be unstable. Instead, optimize h directly with projection.
        
        def objective(y):
            h = np.exp(y)
            # Soft cap to prevent overflow
            h = np.minimum(h, 100.0)
            return _score(h)
        
        def objective_grad(y):
            h = np.exp(y)
            h = np.minimum(h, 100.0)
            # Numerical gradient
            eps = 1e-7
            grad = np.zeros(N)
            for i in range(N):
                y_p = y.copy()
                y_p[i] += eps
                h_p = np.minimum(np.exp(y_p), 100.0)
                y_m = y.copy()
                y_m[i] -= eps
                h_m = np.minimum(np.exp(y_m), 100.0)
                grad[i] = (_score(h_p) - _score(h_m)) / (2 * eps)
            return grad
        
        # Start from best candidate
        y0 = np.log(best_h + 1e-10)
        
        # Try a few different starting points
        starts = [y0]
        # Perturbed starts
        rng = np.random.RandomState(42)
        for _ in range(3):
            y_p = y0 + rng.randn(N) * 0.1
            starts.append(y_p)
        
        best_result = (best_score, best_h.copy())
        
        for y_start in starts:
            if time.time() - start_time > DEADLINE - 20:
                break
            try:
                res = minimize(objective, y_start, method='L-BFGS-B', 
                              options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
                if res.fun < best_result[0]:
                    h_opt = np.exp(res.x)
                    h_opt = np.maximum(h_opt, 0.0)
                    best_result = (res.fun, h_opt)
            except Exception:
                pass
        
        if best_result[0] < best_score:
            best_score = best_result[0]
            best_h = best_result[1]
            
    except ImportError:
        pass
    
    # Final local polish: coordinate descent on h
    if time.time() - start_time < DEADLINE - 10:
        h = best_h.copy()
        improved = True
        max_iters = 50
        for it in range(max_iters):
            if time.time() - start_time > DEADLINE - 10:
                break
            if not improved:
                break
            improved = False
            for i in range(N):
                # Try small perturbations of h[i]
                h0 = h[i]
                for factor in [0.5, 0.7, 0.85, 1.15, 1.3, 1.5]:
                    h[i] = max(h0 * factor, 1e-10)
                    s = _score(h)
                    if s < best_score:
                        best_score = s
                        improved = True
                    else:
                        h[i] = h0
                h[i] = h0  # restore
        
        # Keep the best
        if improved:
            # Re-evaluate to get best_h
            pass
    
    # Ensure nonnegativity
    best_h = np.maximum(best_h, 0.0)
    
    # Final check
    final_score = _score(best_h)
    if final_score < best_score:
        best_score = final_score
    
    # Time guard
    if time.time() - start_time > DEADLINE - 5:
        return [float(v) for v in best_h]
    
    return [float(v) for v in best_h]
