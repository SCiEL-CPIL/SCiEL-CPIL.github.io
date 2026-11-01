"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _compute_score(h):
    """
    Compute the upper bound on C3 for a signed step function with N intervals.
    
    The autoconvolution is computed, and we look at the maximum sidelobe
    (excluding the main lobe at lag 0). The score is this maximum sidelobe value.
    Lower is better.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    
    # Compute autocorrelation via FFT
    # Zero-pad to at least 2n for linear autocorrelation
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    
    H = np.fft.rfft(h, n=fft_size)
    autocorr = np.fft.irfft(H * np.conj(H), n=fft_size)[:n]
    
    # The main lobe is at index 0 (lag 0). Sidelobes are at lags 1..n-1
    # For a real signal, autocorr is symmetric, so we just need lags 1..n-1
    # But actually, the full autocorrelation for a real signal has:
    # autocorr[k] = sum_i h[i] * h[i+k] for k >= 0
    # The sidelobes are autocorr[1], autocorr[2], ..., autocorr[n-1]
    
    # The "upper bound on C3" is the maximum absolute value of the sidelobes
    # We want to minimize this
    max_sidelobe = np.max(np.abs(autocorr[1:]))
    
    return max_sidelobe

def construct():
    """
    Construct a signed step function with N intervals that minimizes the
    maximum autocorrelation sidelobe.
    
    Uses a multi-start local optimization approach with chaotic initializations.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # 120 second limit
    
    # Initial best solution: all ones (the seed)
    best_h = np.ones(N)
    best_score = _compute_score(best_h)
    
    # Try several different initializations and optimization approaches
    rng = np.random.RandomState(42)
    
    # Strategy 1: Use a chaotic logistic map to generate initial patterns
    # Strategy 2: Use random sign patterns
    # Strategy 3: Use low-discrepancy sequences
    
    # Generate multiple initial candidates
    candidates = []
    
    # Candidate 0: All ones (baseline)
    candidates.append(np.ones(N))
    
    # Candidate 1: Random signs
    candidates.append(np.sign(rng.randn(N)))
    
    # Candidate 2: Logistic map chaotic sequence
    x = 0.5
    logistic_vals = []
    for i in range(N):
        x = 3.99 * x * (1 - x)
        logistic_vals.append(1.0 if x > 0.5 else -1.0)
    candidates.append(np.array(logistic_vals))
    
    # Candidate 3: Van der Corput low-discrepancy sequence
    vdc_vals = []
    for i in range(N):
        # Binary van der Corput
        x = 0.0
        d = 0.5
        n = i + 1
        while n > 0:
            if n % 2 == 1:
                x += d
            n //= 2
            d /= 2.0
        vdc_vals.append(1.0 if x > 0.5 else -1.0)
    candidates.append(np.array(vdc_vals))
    
    # Candidate 4: Alternating pattern
    candidates.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    
    # Candidate 5: Thue-Morse sequence
    tm_vals = [1.0]
    for _ in range(9):  # 2^9 = 512 > 400
        tm_vals = tm_vals + [-v for v in tm_vals]
    candidates.append(np.array(tm_vals[:N]))
    
    # Evaluate all candidates
    for h in candidates:
        if time.time() > deadline - 5.0:
            break
        score = _compute_score(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Now try to improve the best candidate using local optimization
    # We'll use continuous relaxation: allow h[i] to be in [-1, 1] and optimize
    # the maximum sidelobe, then round to {-1, 1} at the end
    
    # Use a soft-max surrogate for the max to make it differentiable
    def _soft_max_sidelobe(h, beta=50.0):
        """
        Soft surrogate for the maximum absolute sidelobe.
        Uses log-sum-exp for smoothness.
        """
        h = np.asarray(h, dtype=np.float64)
        n = len(h)
        
        fft_size = 1
        while fft_size < 2 * n:
            fft_size *= 2
        
        H = np.fft.rfft(h, n=fft_size)
        autocorr = np.fft.irfft(H * np.conj(H), n=fft_size)[:n]
        
        sidelobes = np.abs(autocorr[1:])
        
        # Soft max using log-sum-exp
        max_val = np.max(sidelobes)
        if max_val == 0:
            return 0.0
        # Scale to avoid overflow
        scaled = beta * sidelobes
        max_scaled = np.max(scaled)
        log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_max = log_sum_exp / beta
        
        return soft_max
    
    # Try local optimization from the best discrete solution
    h0 = best_h.copy()
    
    if time.time() < deadline - 10.0:
        # Optimize with continuous variables
        def _objective(x):
            return _soft_max_sidelobe(x, beta=100.0)
        
        try:
            result = minimize(
                _objective, 
                h0, 
                method='L-BFGS-B',
                bounds=[(-1.0, 1.0)] * N,
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if time.time() < deadline - 5.0:
                h_opt = result.x
                # Round to discrete signs
                h_rounded = np.sign(h_opt)
                # Handle zeros by keeping sign or defaulting to 1
                h_rounded[h_rounded == 0] = 1.0
                
                score_rounded = _compute_score(h_rounded)
                if score_rounded < best_score:
                    best_score = score_rounded
                    best_h = h_rounded.copy()
        except Exception:
            pass
    
    # Try a few more random restarts if time permits
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Random perturbation of the best solution
        h_perturbed = best_h.copy()
        # Flip a random subset of signs
        flip_mask = rng.rand(N) < 0.1
        h_perturbed[flip_mask] *= -1
        
        score_perturbed = _compute_score(h_perturbed)
        if score_perturbed < best_score:
            best_score = score_perturbed
            best_h = h_perturbed.copy()
        
        # Try local optimization from this perturbed solution
        if time.time() < deadline - 10.0:
            def _objective2(x):
                return _soft_max_sidelobe(x, beta=100.0)
            
            try:
                result2 = minimize(
                    _objective2,
                    h_perturbed,
                    method='L-BFGS-B',
                    bounds=[(-1.0, 1.0)] * N,
                    options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                
                if time.time() < deadline - 5.0:
                    h_opt2 = result2.x
                    h_rounded2 = np.sign(h_opt2)
                    h_rounded2[h_rounded2 == 0] = 1.0
                    
                    score_rounded2 = _compute_score(h_rounded2)
                    if score_rounded2 < best_score:
                        best_score = score_rounded2
                        best_h = h_rounded2.copy()
            except Exception:
                pass
    
    # Final check: make sure we return a valid signed step function
    best_h = np.sign(best_h)
    best_h[best_h == 0] = 1.0
    
    return best_h
# EVOLVE-BLOCK-END
