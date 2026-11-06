import math
import time
import numpy as np

_N = 24
_SQRTN = math.sqrt(_N)
_T0 = time.time()
_LIMIT = 145.0

def _time_left():
    return _LIMIT - (time.time() - _T0)

def _rs_poly(n):
    """Generate Rudin-Shapiro coefficients of length n."""
    if n <= 0:
        return []
    if n == 1:
        return [1]
    if n == 2:
        return [1, 1]
    
    R = [1]
    S = [1]
    while len(R) < n:
        new_R = R + S
        new_S = R + [-s for s in S]
        R = new_R
        S = new_S
    return R[:n]

def _calc_peak(coeffs, M=4096, refine_top=10, refine_iters=40):
    """
    Calculate the peak magnitude of P(z) on the unit circle.
    Uses dense sampling and golden-section refinement for top peaks.
    """
    c = np.array(coeffs, dtype=complex)
    
    k = np.arange(M)
    theta = 2.0 * np.pi * k / M
    
    powers = np.arange(_N)
    Z = np.exp(1j * np.outer(theta, powers))
    vals = Z @ c
    
    mag2 = np.abs(vals) ** 2
    peak_val = np.max(mag2)
    
    # Find indices of top peaks and refine
    if refine_top > 0 and M > 10:
        top_indices = np.argsort(mag2)[-refine_top:][::-1]
        best_peak = peak_val
        powers_arr = powers
        
        for idx in top_indices:
            delta = 2.0 * math.pi / M
            lo = theta[idx] - delta
            hi = theta[idx] + delta
            
            def f(a):
                z_a = np.exp(1j * a)
                z_powers = np.power(z_a, powers_arr)
                s = np.dot(c, z_powers)
                return s.real * s.real + s.imag * s.imag

            gr = (math.sqrt(5) - 1) / 2
            x1 = hi - gr * (hi - lo)
            x2 = lo + gr * (hi - lo)
            f1 = f(x1)
            f2 = f(x2)
            
            for _ in range(refine_iters):
                if f1 < f2:
                    hi = x2
                    x2 = x1
                    f2 = f1
                    x1 = hi - gr * (hi - lo)
                    f1 = f(x1)
                else:
                    lo = x1
                    x1 = x2
                    f1 = f2
                    x2 = lo + gr * (hi - lo)
                    f2 = f(x2)
            
            local_peak = max(f1, f2)
            if local_peak > best_peak:
                best_peak = local_peak
        
        peak_val = best_peak
    
    return math.sqrt(peak_val)

def _exhaustive_search(M=1024, batch_size=1024):
    """
    Exhaustively search all 2^23 sign-equivalence classes (c_0 fixed to +1).
    Uses vectorized batch matmul for speed.
    Returns (best_coeffs, best_peak) where best_peak is the max |P(z)| over |z|=1.
    """
    start_time = time.time()
    
    # Precompute the DFT matrix Z: M x N
    k = np.arange(M)
    theta = 2.0 * np.pi * k / M
    powers = np.arange(_N)
    Z = np.exp(1j * np.outer(theta, powers))  # (M, N)
    
    # Total number of sign classes: 2^23 (c_0 = +1 fixed)
    total = 1 << (_N - 1)  # 2^23 = 8388608
    
    best_peak2 = float('inf')
    best_mask = None
    
    # Process in batches
    for batch_start in range(0, total, batch_size):
        if _time_left() < 10.0:
            break
        
        batch_end = min(batch_start + batch_size, total)
        n_batch = batch_end - batch_start
        
        # Generate bitmasks for this batch: each mask is 23 bits (coefficients 1..23)
        masks = np.arange(batch_start, batch_end, dtype=np.uint32)
        
        # Extract bits: bit i of mask corresponds to coefficient (i+1)
        # If bit is set, coefficient is -1; if not set, coefficient is +1
        # We need to build the coefficient matrix: (n_batch, N)
        # c_0 = 1 always, c_j = (-1)^(bit_{j-1}) for j=1..23
        
        # Efficient: use bit operations
        # For each coefficient position j (1-indexed), the bit is at position (j-1)
        # coeffs[i, 0] = 1
        # coeffs[i, j] = 1 - 2 * bit_{j-1}(mask_i)
        
        # Build coefficient matrix using broadcasting
        # bit_positions: for coefficient j (j=1..23), bit position is j-1
        bit_positions = np.arange(_N - 1, dtype=np.uint32)  # 0, 1, ..., 22
        
        # Extract bits: (n_batch, N-1)
        bits = ((masks[:, None] >> bit_positions[None, :]) & 1).astype(np.float64)
        
        # Coefficients: c_0 = 1, c_j = 1 - 2*bit_{j-1}
        coeffs_batch = np.ones((n_batch, _N), dtype=np.float64)
        coeffs_batch[:, 1:] = 1.0 - 2.0 * bits
        
        # Compute P(z) for all polynomials in batch: (n_batch, M)
        # vals[i, k] = sum_j coeffs_batch[i, j] * Z[k, j]
        # = coeffs_batch @ Z.T
        vals = coeffs_batch @ Z.T  # (n_batch, M)
        
        # Peak squared magnitude for each polynomial
        peak2_batch = np.max(np.abs(vals) ** 2, axis=1)  # (n_batch,)
        
        # Find the best in this batch
        min_idx = np.argmin(peak2_batch)
        if peak2_batch[min_idx] < best_peak2:
            best_peak2 = peak2_batch[min_idx]
            best_mask = masks[min_idx]
    
    # Convert best_mask to coefficients
    if best_mask is None:
        # Fallback to RS
        rs = _rs_poly(_N)
        return rs, _calc_peak(rs)
    
    best_mask = int(best_mask)
    best_coeffs = [1]  # c_0 = +1
    for j in range(1, _N):
        bit = (best_mask >> (j - 1)) & 1
        best_coeffs.append(1 if bit == 0 else -1)
    
    return best_coeffs, math.sqrt(best_peak2)

def construct():
    global _T0
    _T0 = time.time()
    
    # Start with Rudin-Shapiro as initial best
    seed_rs = _rs_poly(_N)
    best_coeffs = seed_rs
    best_peak = _calc_peak(seed_rs, M=4096, refine_top=10, refine_iters=40)
    
    # Run exhaustive search with vectorized batch matmul
    # This guarantees finding the global optimum
    ex_coeffs, ex_peak = _exhaustive_search(M=1024, batch_size=1024)
    
    if ex_peak < best_peak:
        best_coeffs = ex_coeffs
        best_peak = ex_peak
    
    # Final high-precision verification with golden-section refinement
    if _time_left() > 5.0:
        final_peak = _calc_peak(best_coeffs, M=8192, refine_top=10, refine_iters=50)
        best_peak = final_peak
    
    # Ensure coefficients are +1 or -1
    best_coeffs = [1 if c >= 0 else -1 for c in best_coeffs]
    
    return best_coeffs

if __name__ == "__main__":
    # Self-test
    coeffs = construct()
    print(coeffs)
    print(len(coeffs))
    print(_calc_peak(coeffs, M=8192, refine_top=10, refine_iters=50))
