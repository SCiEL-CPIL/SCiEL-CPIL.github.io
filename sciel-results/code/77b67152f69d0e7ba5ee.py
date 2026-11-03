"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 400

# EVOLVE-BLOCK-START

def _compute_score(x):
    """
    Compute the score: maximum absolute autoconvolution divided by squared integral.
    Lower is better.
    """
    # Integral (sum) of x
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return np.inf
    
    # Autoconvolution: c[k] = sum_i x[i] * x[i+k] for k in -(N-1)..(N-1)
    # Using FFT for efficiency
    n_fft = 1 << (2 * N - 1).bit_length()
    fft_x = np.fft.rfft(x, n_fft)
    autoconv = np.fft.irfft(fft_x * np.conj(fft_x), n_fft)
    
    # The linear autoconvolution is in autoconv[0:N] (positive lags) 
    # and autoconv[n_fft-N+1:n_fft] (negative lags, reversed)
    # Actually, for real input, irfft of |X|^2 gives circular convolution.
    # Since we zero-padded to 2N-1, the first N elements are the linear autoconvolution for lags 0..N-1.
    # The negative lags are symmetric: c[-k] = c[k].
    # So we just need to check c[0:N].
    c = autoconv[:N]
    
    max_abs = np.max(np.abs(c))
    
    return max_abs / (integral * integral)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial guess: alternating signs or random
    # Let's try a few initializations and pick the best
    best_x = None
    best_score = np.inf
    
    # Try a few different starting points
    starts = []
    
    # 1. All ones (baseline)
    starts.append(np.ones(N))
    
    # 2. Random
    np.random.seed(42)
    starts.append(np.random.randn(N))
    
    # 3. Alternating
    starts.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    
    # 4. Block signs
    for block_size in [4, 8, 16, 32]:
        x = np.ones(N)
        for i in range(0, N, block_size):
            if (i // block_size) % 2 == 1:
                x[i:i+block_size] = -1.0
        starts.append(x)
    
    # 5. Random block signs
    for block_size in [4, 8, 16]:
        x = np.ones(N)
        for i in range(0, N, block_size):
            if np.random.rand() < 0.5:
                x[i:i+block_size] = -1.0
        starts.append(x)
    
    # Evaluate initial scores
    for x in starts:
        if time.time() > deadline:
            break
        score = _compute_score(x)
        if score < best_score:
            best_score = score
            best_x = x.copy()
    
    # Optimize from the best starting point
    if best_x is not None:
        # Use L-BFGS-B with gradient
        # The objective is max_k |c[k]| / (sum(x))^2
        # This is non-smooth due to max and abs.
        # We can use a smooth approximation: sum(exp(beta * |c[k]|)) / beta
        # But let's first try direct optimization with subgradient or just L-BFGS on the smooth part
        
        # Alternative: minimize max_k c[k]^2 / (sum(x))^2
        # Since max |c[k]| = sqrt(max c[k]^2), and (sum x)^2 is positive,
        # minimizing max c[k]^2 / (sum x)^2 is equivalent.
        
        def objective(x):
            integral = np.sum(x)
            if abs(integral) < 1e-12:
                return 1e10
            n_fft = 1 << (2 * N - 1).bit_length()
            fft_x = np.fft.rfft(x, n_fft)
            autoconv = np.fft.irfft(fft_x * np.conj(fft_x), n_fft)
            c = autoconv[:N]
            return np.max(c * c) / (integral * integral)
        
        # Try optimization
        try:
            res = minimize(objective, best_x, method='L-BFGS-B', 
                          options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x
        except Exception:
            pass
        
        # If we have time, try a few more local optimizations from different starts
        for x in starts:
            if time.time() > deadline:
                break
            try:
                res = minimize(objective, x, method='L-BFGS-B',
                              options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
                if res.fun < best_score:
                    best_score = res.fun
                    best_x = res.x
            except Exception:
                pass
    
    # Final result: signed step heights
    # The task says "sequence of signed step heights"
    # We should return the optimized x
    if best_x is not None:
        return best_x
    else:
        return np.ones(N)

# EVOLVE-BLOCK-END
