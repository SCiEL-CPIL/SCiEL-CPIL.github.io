import math
import time
import numpy as np

_N = 24
_SQRTN = math.sqrt(_N)
_T0 = time.time()
_LIMIT = 120.0
_TIME_MARGIN = 15.0

def _time_left():
    return _LIMIT - _TIME_MARGIN - (time.time() - _T0)

def _eval_poly_dense(coeffs, theta):
    """
    Evaluate polynomial P(z) = sum c_k z^k at angles theta.
    coeffs: (24,) array of +1/-1
    theta: (M,) array of angles
    Returns: (M,) complex values
    """
    z = np.exp(1j * theta)
    # Use Horner's method or direct matrix multiplication?
    # Direct matrix multiplication is faster for small N and large M.
    # Z[k, j] = z_k^j
    # We can compute powers iteratively to save memory/time if M is large, 
    # but for M=2048, precomputing Z is fine.
    # However, for the dense refinement, we only evaluate one polynomial at a time.
    # So we can just do:
    # val = c_0 + c_1 z + c_2 z^2 + ...
    # This is effectively a dot product with powers.
    
    # For a single polynomial, computing powers of z is O(M*N).
    # z_powers = np.power(z[:, None], np.arange(_N)[None, :]) # (M, N)
    # vals = z_powers @ coeffs # (M,)
    
    # Alternatively, use FFT if M is a power of 2 and we evaluate at roots of unity.
    # But theta is arbitrary in refinement.
    
    # Let's stick to the matrix approach for simplicity and speed in vectorized numpy.
    # Actually, for a single coeff vector, we can compute powers iteratively in a loop
    # but that's slow in Python. 
    # Vectorized:
    powers = np.arange(_N)
    # z is (M,)
    # We want (M, N) matrix of z_k^j
    # This creates a large temporary array.
    # For M=2048, N=24, it's 2048*24*16 bytes ~ 786 KB. Very fast.
    
    zk = np.power(z[:, None], powers[None, :])
    return zk @ coeffs

def _refine_peak(coeffs, peak_val_sq, phase, iters=25):
    """
    Refine the peak around a specific phase using golden section search.
    coeffs: array of 24 floats (+1/-1)
    peak_val_sq: current known peak magnitude squared (lower bound for return)
    phase: initial phase estimate in radians
    Returns: refined peak magnitude squared
    """
    def f(a):
        z = np.exp(1j * a)
        # Compute P(z)
        # Using iterative power to avoid creating arrays for each iteration?
        # No, numpy scalar operations are fast enough for 25 iterations.
        # Or just use the same logic as _eval_poly_dense but for single point.
        # P(z) = sum c_k z^k
        # z_powers = z ** np.arange(_N)
        # s = np.dot(coeffs, z_powers)
        
        # Optimized single point eval:
        # Horner's rule is O(N) and no extra memory.
        # P(z) = c_0 + z(c_1 + z(c_2 + ...))
        # But coeffs are in order c_0...c_23.
        # Let's use dot product with precomputed powers? No, z changes.
        
        # Simple loop in Python might be slower than numpy dot.
        # Let's use numpy dot with a small array.
        zk = z ** np.arange(_N)
        s = np.dot(coeffs, zk)
        return s.real * s.real + s.imag * s.imag

    # Window around phase
    # The grid spacing used in the previous step determines the window.
    # We assume the peak is within +/- 1 grid step.
    # Let's use a slightly larger window to be safe, e.g., 2 grid steps of the dense grid.
    # But we don't pass the grid size. We'll assume a standard small window.
    # If the coarse grid was 256, the error is small.
    # Let's use a window of 2*pi/256.
    d_phase = 2.0 * math.pi / 256.0
    lo = phase - d_phase
    hi = phase + d_phase
    
    f_lo = f(lo)
    f_hi = f(hi)
    
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    
    for _ in range(iters):
        m1 = hi - gr * (hi - lo)
        m2 = lo + gr * (hi - lo)
        f1 = f(m1)
        f2 = f(m2)
        if f1 < f2:
            hi = m2
            f_hi = f2
        else:
            lo = m1
            f_lo = f1
            
    peak2 = max(f_lo, f_hi)
    return max(peak2, peak_val_sq)

def _full_peak_eval(coeffs, M=2048):
    """
    Evaluate max |P(z)| over a dense grid and refine the top peak.
    Returns: refined peak magnitude squared
    """
    theta = 2.0 * np.pi * np.arange(M) / M
    vals = _eval_poly_dense(coeffs, theta)
    mag2 = np.abs(vals) ** 2
    peak_idx = np.argmax(mag2)
    peak_sq = mag2[peak_idx]
    phase = theta[peak_idx]
    
    # Refine around the peak
    refined_sq = _refine_peak(coeffs, peak_sq, phase)
    return refined_sq

def construct():
    global _T0
    _T0 = time.time()
    
    # Phase 1: Coarse Brute Force
    # We use a grid size of 256 for the initial scan.
    # This allows us to scan more candidates in the time limit.
    M_coarse = 256
    theta_coarse = 2.0 * np.pi * np.arange(M_coarse) / M_coarse
    z_coarse = np.exp(1j * theta_coarse)
    Z_coarse = np.power(z_coarse[:, None], np.arange(_N)[None, :])
    
    # We will keep the top K candidates from the coarse scan
    K = 10
    top_candidates = [] # List of (peak_sq, coeff_idx)
    
    B = 200000 # Larger batch size for efficiency
    total = 1 << _N
    
    best_coarse_peak_sq = float('inf')
    
    for start in range(0, total, B):
        if _time_left() < 20.0: # Leave 20s for refinement and local search
            break
        
        end = min(start + B, total)
        idxs = np.arange(start, end, dtype=np.uint32)
        
        # Convert indices to bit patterns
        bit_mask = 1 << np.arange(_N, dtype=np.uint32)
        bits = (idxs[:, None] & bit_mask[None, :]) > 0
        coeffs = (bits.astype(np.float64) * 2.0) - 1.0
        
        # Evaluate
        vals = coeffs @ Z_coarse.T
        mag2 = np.abs(vals) ** 2
        peaks = np.max(mag2, axis=1)
        
        # Find the best in this batch
        min_idx = np.argmin(peaks)
        min_peak_sq = peaks[min_idx]
        
        if min_peak_sq < best_coarse_peak_sq:
            best_coarse_peak_sq = min_peak_sq
            top_candidates.append((min_peak_sq, start + min_idx))
            
            # Keep only top K
            if len(top_candidates) > K:
                top_candidates.sort(key=lambda x: x[0])
                top_candidates = top_candidates[:K]
        else:
            # If the new candidate is better than the worst in our top K, replace
            if top_candidates and min_peak_sq < top_candidates[-1][0]:
                top_candidates.append((min_peak_sq, start + min_idx))
                top_candidates.sort(key=lambda x: x[0])
                top_candidates = top_candidates[:K]
            elif not top_candidates:
                top_candidates.append((min_peak_sq, start + min_idx))

    if not top_candidates:
        # Fallback: return random or all ones if no time was spent
        return [1] * _N

    # Phase 2: High Precision Refinement of Top Candidates
    best_refined_sq = float('inf')
    best_coeff_idx = top_candidates[0][1]
    
    for peak_sq, idx in top_candidates:
        if _time_left() < 10.0:
            break
            
        # Retrieve coefficients
        bits = (idx & (1 << np.arange(_N, dtype=np.uint32))) > 0
        coeffs = (bits.astype(np.float64) * 2.0) - 1.0
        
        # Refine
        refined_sq = _full_peak_eval(coeffs, M=2048)
        
        if refined_sq < best_refined_sq:
            best_refined_sq = refined_sq
            best_coeff_idx = idx

    # Phase 3: Local Search (Hill Climbing)
    # Start from the best candidate found
    bits = (best_coeff_idx & (1 << np.arange(_N, dtype=np.uint32))) > 0
    cur_coeffs = (bits.astype(np.float64) * 2.0) - 1.0
    cur_peak_sq = best_refined_sq
    
    improved = True
    # We can try multiple passes of local search
    while improved and _time_left() > 5.0:
        improved = False
        for j in range(_N):
            if _time_left() < 3.0:
                break
                
            # Flip bit j
            cur_coeffs[j] = -cur_coeffs[j]
            
            # Evaluate new peak
            # Use a slightly smaller grid for local search steps to save time?
            # No, we need consistent metric. Use M=1024 for speed, then final refine?
            # Let's use M=1024 for the local search step evaluation.
            new_peak_sq = _full_peak_eval(cur_coeffs, M=1024)
            
            if new_peak_sq < cur_peak_sq:
                cur_peak_sq = new_peak_sq
                improved = True
            else:
                # Revert
                cur_coeffs[j] = -cur_coeffs[j]

    # Final high-precision refinement of the local search result
    final_peak_sq = _full_peak_eval(cur_coeffs, M=4096)
    
    result = [int(1.0 if c > 0 else -1.0) for c in cur_coeffs]
    return result
