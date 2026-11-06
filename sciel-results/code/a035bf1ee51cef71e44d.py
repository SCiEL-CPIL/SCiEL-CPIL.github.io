import time
import numpy as np
import math

_N = 24
_SQRN = math.sqrt(_N)
_T0 = time.time()
_LIMIT = 115.0 # Safety margin

def _time_left():
    return _LIMIT - (time.time() - _T0)

def _rs_poly():
    """Generate Rudin-Shapiro coefficients truncated to _N."""
    p, q = [1], [1]
    while len(p) < _N:
        p, q = p + q, p + [-c for c in q]
    return p[:_N]

def _eval(coeffs, M=4096):
    """
    Estimate max |P(z)| on unit circle.
    coeffs: list of 24 ints (+1/-1)
    M: number of grid points
    """
    # Precompute angles and exponentials
    # theta_j = 2*pi*j/M
    # z_j^k = exp(i * 2*pi*j*k/M)
    # We can use FFT if we pad, but direct matrix mult is fine for N=24, M=4096.
    # Actually, Z @ c is efficient.
    
    # To speed up, we can precompute the matrix Z once? 
    # No, M might change or we want flexibility.
    # But for local search, M is fixed.
    
    # Let's use a precomputed matrix if possible, or just compute.
    # Computing Z every time is expensive: M*N exp calls.
    # Better: Use FFT.
    # P(z) = sum c_k z^k.
    # If we define c_padded such that c_padded[k] = c_k for k<24, 0 otherwise.
    # Then P(e^{i 2pi j / M}) is NOT simply FFT(c_padded) unless M is power of 2 and we align.
    # FFT computes sum c_k exp(-i 2pi j k / M).
    # Our z = exp(i theta). So z^k = exp(i k theta).
    # This is the inverse FFT sign.
    # np.fft.ifft(c_padded) * M gives sum c_k exp(i 2pi j k / M).
    # So P(z_j) = M * np.fft.ifft(c_padded)[j].
    
    c_padded = np.zeros(M, dtype=complex)
    c_padded[:_N] = coeffs
    # ifft is fast
    vals = np.fft.ifft(c_padded) * M
    return np.max(np.abs(vals))

def _eval_precise(coeffs, M=16384):
    """
    More precise evaluation for final comparison.
    Uses larger grid and refinement.
    """
    c_padded = np.zeros(M, dtype=complex)
    c_padded[:_N] = coeffs
    vals = np.fft.ifft(c_padded) * M
    mags = np.abs(vals)
    
    # Find top 5 peaks
    top_idx = np.argsort(mags)[-5:]
    max_val = mags[top_idx[0]]
    
    # Refine peaks
    # Golden section search on |P(z)|
    # P(z) = sum c_k z^k
    def P_val(theta):
        z = np.exp(1j * theta)
        # Vectorized evaluation
        # k = 0..23
        k = np.arange(_N)
        # z^k
        zk = z ** k
        return np.sum(coeffs * zk)
        
    def P_mag(theta):
        return np.abs(P_val(theta))
        
    for idx in top_idx:
        theta0 = 2 * np.pi * idx / M
        # Search in [theta0 - pi/M, theta0 + pi/M]
        a = theta0 - np.pi / M
        b = theta0 + np.pi / M
        # Golden ratio
        gr = (math.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = P_mag(c)
        fd = P_mag(d)
        for _ in range(30): # 30 iterations for high precision
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = P_mag(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = P_mag(d)
        peak_val = max(fc, fd)
        if peak_val > max_val:
            max_val = peak_val
            
    return max_val

def _local_search(coeffs, M=2048, max_iters=5000, use_double_flip=False):
    """
    Local search with single or double flips.
    """
    current = coeffs[:]
    current_val = _eval(current, M)
    
    for it in range(max_iters):
        if _time_left() < 5.0:
            break
            
        improved = False
        if use_double_flip:
            # Try all pairs? 24*23/2 = 276 pairs.
            # This is heavier.
            # We can sample a subset or do it all if fast enough.
            # _eval is ~ O(M log M) + O(N). With M=2048, it's fast.
            # 276 * 5000 = 1.3M evals. Might be too slow in Python loops.
            # Let's stick to single flips for speed, or limited double flips.
            pass # Placeholder for double flip logic if needed
            
        # Single flip
        best_new_val = current_val
        best_new_idx = -1
        
        # Vectorized check for single flips is hard because eval is global.
        # We have to loop over k.
        for k in range(_N):
            # Flip k
            old_c = current[k]
            current[k] = -old_c
            
            new_val = _eval(current, M)
            
            if new_val < best_new_val:
                best_new_val = new_val
                best_new_idx = k
                # Don't break, check all to find the best improvement
            else:
                current[k] = old_c
        
        if best_new_idx != -1:
            # Apply the best flip
            current[best_new_idx] = -current[best_new_idx]
            current_val = best_new_val
            improved = True
        else:
            break # Local minimum
            
    return current, current_val

def construct():
    t_start = time.time()
    
    # Initial candidate: Rudin-Shapiro
    best_coeffs = _rs_poly()
    best_val = _eval_precise(best_coeffs)
    
    # Strategy:
    # 1. Run local search from RS with single flips.
    # 2. Run local search from random starts with single flips.
    # 3. If time permits, try double flips on the best candidate.
    
    # Local search 1: RS
    c1, v1 = _local_search(best_coeffs, M=2048, max_iters=10000)
    if v1 < best_val:
        best_coeffs = c1
        best_val = v1
    
    # Random restarts
    num_restarts = 0
    while _time_left() > 10.0:
        # Random start
        c_rand = [1 if np.random.rand() > 0.5 else -1 for _ in range(_N)]
        
        # Quick local search
        c2, v2 = _local_search(c_rand, M=1024, max_iters=2000)
        
        if v2 < best_val:
            # Verify with higher precision
            v2_precise = _eval_precise(c2)
            if v2_precise < best_val:
                best_coeffs = c2
                best_val = v2_precise
        
        num_restarts += 1
        if num_restarts % 5 == 0:
             # Periodic check with higher res on current best?
             pass

    # Final polish: Try to improve best_coeffs with a more careful search
    # Use higher M for the final local search
    if _time_left() > 5.0:
        c3, v3 = _local_search(best_coeffs, M=4096, max_iters=5000)
        if v3 < best_val:
             # Verify
             v3_precise = _eval_precise(c3)
             if v3_precise < best_val:
                 best_coeffs = c3
                 best_val = v3_precise

    return best_coeffs
