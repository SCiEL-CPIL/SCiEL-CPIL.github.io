import numpy as np
import time
from scipy.optimize import differential_evolution

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill

    # Precompute Chebyshev basis
    # x in [-1, 1]
    x = np.linspace(-1, 1, N)
    # Chebyshev polynomials T_k(x)
    # T_0(x) = 1
    # T_1(x) = x
    # T_{k+1}(x) = 2x T_k(x) - T_{k-1}(x)
    
    n_basis = 10
    basis = np.zeros((n_basis, N))
    basis[0] = 1.0
    if n_basis > 1:
        basis[1] = x
    for k in range(2, n_basis):
        basis[k] = 2 * x * basis[k-1] - basis[k-2]
        
    # We want to minimize value = abs(2 * N * max(conv(h,h)) / sum(h)^2)
    # Let h = B @ c, where B is (n_basis x N) and c is (n_basis).
    # sum(h) = sum(B @ c) = (sum(B, axis=1)) @ c
    # conv(h, h) depends quadratically on c.
    
    # To make the problem well-posed for DE, we can fix the sum(h) to be 1.
    # Then we minimize abs(2 * N * max(conv(h,h))).
    # However, the constraint sum(h)=1 is linear.
    # Alternatively, since the objective is scale invariant (h -> alpha h doesn't change value),
    # we can just minimize the value directly, but DE might struggle with scale invariance.
    # Better: normalize c such that sum(B@c) = 1.
    # This is tricky to enforce strictly in DE.
    # Instead, let's just minimize the ratio directly.
    # f(c) = abs(2 * N * max(conv(B@c, B@c)) / (sum(B@c))^2)
    
    # Fast scoring for the low-dimensional parameter space
    def fast_score_params(c):
        h = basis.dot(c)
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        # Use FFT for convolution
        size = 1 << (2 * N - 1).bit_length()
        H = np.fft.rfft(h, size)
        Z = np.fft.irfft(H * H, size)[:2 * N - 1]
        return abs(2.0 * N * np.max(Z) / (s * s))
        
    # Bounds for coefficients
    # We expect coefficients to be O(1).
    # Let's start with bounds [-10, 10] for all coefficients.
    bounds = [(-10, 10)] * n_basis
    
    # Initial seed: all ones corresponds to c such that B@c = 1.
    # Since T_0 = 1, c = [1, 0, ..., 0] gives h = 1.
    # Value for seed is 2.0.
    
    # Differential Evolution
    # popsize=15, maxiter=100, tol=0.01
    # We need to be careful with time.
    # Each evaluation is fast (FFT of length ~1024).
    # 15 * 100 * 10 = 15000 evaluations.
    # Each eval ~ 0.1 ms? 15000 * 0.1 ms = 1.5 s. Should be fine.
    
    try:
        result = differential_evolution(
            fast_score_params, 
            bounds, 
            maxiter=200, 
            popsize=20, 
            tol=1e-5, 
            seed=42,
            polish=False  # Skip polish to save time, we can do local refinement if needed
        )
        
        if result.success or (time.time() < deadline - 10):
            best_c = result.x
            best_val = result.fun
            best_h = basis.dot(best_c)
            
            # Local refinement using random perturbation in coefficient space
            rng = np.random.RandomState(42)
            for _ in range(500):
                if time.time() > deadline - 5:
                    break
                c_pert = best_c + rng.randn(n_basis) * 0.1
                val = fast_score_params(c_pert)
                if val < best_val:
                    best_val = val
                    best_c = c_pert
                    best_h = basis.dot(best_c)
                    
            # Verify with exact np.convolve
            s = np.sum(best_h)
            if abs(s) > 1e-10:
                z = np.convolve(best_h, best_h)
                exact_val = abs(2.0 * N * np.max(z) / (s * s))
                if exact_val < best_val:
                    best_val = exact_val
            else:
                # Fallback to seed
                return np.ones(N)
                
            return best_h
            
    except Exception:
        pass

    # Fallback: If DE fails or times out, return seed
    return np.ones(N)
