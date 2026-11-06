import math
import time
import numpy as np

_N = 24
_SQRTN = math.sqrt(_N)
_T0 = time.time()
_LIMIT = 120.0
_TIME_MARGIN = 10.0

def _time_left():
    return _LIMIT - _TIME_MARGIN - (time.time() - _T0)

def _refine_peak(coeffs, peak_val, phase, iters=20):
    """
    Refine the peak around a specific phase using golden section search.
    coeffs: list of 24 floats (+1/-1)
    peak_val: current known peak magnitude (squared for comparison)
    phase: initial phase estimate in radians
    """
    c_arr = np.array(coeffs, dtype=complex)
    powers = np.arange(_N)
    
    def f(a):
        z = np.exp(1j * a)
        zk = np.power(z, powers)
        s = np.dot(c_arr, zk)
        return (s.real * s.real + s.imag * s.imag)

    # We need a window around the phase.
    # Use a reasonable window based on the grid spacing
    M_grid = 128  # Coarser grid for initial search
    d_phase = 2.0 * math.pi / M_grid
    lo = phase - d_phase
    hi = phase + d_phase
    
    f_lo = f(lo)
    f_hi = f(hi)
    
    # Golden section parameters
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
    if peak2 > peak_val:
        return math.sqrt(peak2)
    return math.sqrt(peak_val)

def _eval_peak_fast(coeffs, Z, theta):
    """Evaluate the peak magnitude of a polynomial on the grid."""
    vals = coeffs @ Z.T
    mag2 = np.abs(vals) ** 2
    peak_idx = np.argmax(mag2)
    return mag2[peak_idx], theta[peak_idx]

def construct():
    global _T0
    _T0 = time.time()
    
    # Brute force approach with coarser grid for speed
    M = 128
    k = np.arange(M)
    theta = 2.0 * np.pi * k / M
    z = np.exp(1j * theta)
    Z = np.power(z[:, None], np.arange(_N)[None, :])
    
    best_peak_sq = float('inf')
    best_coeff_idx = 0
    best_peak_phase = 0.0
    
    # Keep track of top K candidates
    K = 50
    top_candidates = []  # List of (peak_sq, coeff_idx, peak_phase)
    
    # Batch size - larger for efficiency
    B = 200000
    total = 1 << _N
    
    bit_mask = 1 << np.arange(_N, dtype=np.uint32)
    
    for start in range(0, total, B):
        if _time_left() < 5.0:
            break
        
        end = min(start + B, total)
        idxs = np.arange(start, end)
        bits = (idxs[:, None].astype(np.uint32) & bit_mask[None, :]) > 0
        coeffs = (bits.astype(np.float64) * 2.0) - 1.0
        
        vals = coeffs @ Z.T
        mag2 = np.abs(vals) ** 2
        peaks = np.max(mag2, axis=1)
        
        # Find top K in this batch
        top_k_idx = np.argsort(peaks)[:K]
        for i in top_k_idx:
            peak_sq = peaks[i]
            coeff_idx = start + i
            peak_phase = theta[np.argmax(mag2[i])]
            
            # Add to top candidates if it's good enough
            if peak_sq < best_peak_sq:
                best_peak_sq = peak_sq
                best_coeff_idx = coeff_idx
                best_peak_phase = peak_phase
            
            # Maintain top K list
            if len(top_candidates) < K or peak_sq < top_candidates[-1][0]:
                top_candidates.append((peak_sq, coeff_idx, peak_phase))
                top_candidates.sort(key=lambda x: x[0])
                if len(top_candidates) > K:
                    top_candidates.pop()
    
    # After brute force, retrieve the best candidate
    bits = (best_coeff_idx & (1 << np.arange(_N, dtype=np.uint32))) > 0
    best_coeffs = (bits.astype(np.float64) * 2.0) - 1.0
    
    # Refine the peak of the best candidate
    true_peak = _refine_peak(best_coeffs, best_peak_sq, best_peak_phase)
    cur_peak_sq = true_peak * true_peak
    
    # Local search with multiple iterations and double-flip moves
    cur_coeffs = best_coeffs.copy()
    improved = True
    max_iters = 10
    
    for it in range(max_iters):
        if _time_left() < 3.0:
            break
        if not improved:
            break
        improved = False
        
        # Try single flips
        for j in range(_N):
            if _time_left() < 2.0:
                break
            cur_coeffs[j] = -cur_coeffs[j]
            new_peak_sq, new_phase = _eval_peak_fast(cur_coeffs, Z, theta)
            
            if new_peak_sq < cur_peak_sq:
                # Refine to get accurate peak
                refined_peak = _refine_peak(cur_coeffs, new_peak_sq, new_phase)
                if refined_peak * refined_peak < cur_peak_sq:
                    cur_peak_sq = refined_peak * refined_peak
                    best_coeffs = cur_coeffs.copy()
                    best_peak_sq = cur_peak_sq
                    best_peak_phase = new_phase
                    improved = True
                else:
                    cur_coeffs[j] = -cur_coeffs[j]
            else:
                cur_coeffs[j] = -cur_coeffs[j]
        
        # Try double flips
        for j1 in range(_N):
            if _time_left() < 2.0:
                break
            for j2 in range(j1 + 1, _N):
                if _time_left() < 1.0:
                    break
                cur_coeffs[j1] = -cur_coeffs[j1]
                cur_coeffs[j2] = -cur_coeffs[j2]
                new_peak_sq, new_phase = _eval_peak_fast(cur_coeffs, Z, theta)
                
                if new_peak_sq < cur_peak_sq:
                    refined_peak = _refine_peak(cur_coeffs, new_peak_sq, new_phase)
                    if refined_peak * refined_peak < cur_peak_sq:
                        cur_peak_sq = refined_peak * refined_peak
                        best_coeffs = cur_coeffs.copy()
                        best_peak_sq = cur_peak_sq
                        best_peak_phase = new_phase
                        improved = True
                    else:
                        cur_coeffs[j1] = -cur_coeffs[j1]
                        cur_coeffs[j2] = -cur_coeffs[j2]
                else:
                    cur_coeffs[j1] = -cur_coeffs[j1]
                    cur_coeffs[j2] = -cur_coeffs[j2]
    
    # Convert back to list of ints
    result = [int(1.0 if c > 0 else -1.0) for c in best_coeffs]
    
    return result
