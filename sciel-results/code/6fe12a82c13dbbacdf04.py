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

def _eval_poly(coeffs, theta):
    """Evaluate polynomial at multiple angles."""
    z = np.exp(1j * theta)
    powers = np.arange(_N)
    zk = np.power(z[:, None], powers[None, :])
    return zk @ coeffs

def _refine_peak(coeffs, peak_val_sq, phase, iters=30):
    """Refine the peak around a specific phase using golden section search."""
    def f(a):
        z = np.exp(1j * a)
        zk = z ** np.arange(_N)
        s = np.dot(coeffs, zk)
        return s.real * s.real + s.imag * s.imag

    M_grid = 256
    d_phase = 2.0 * math.pi / M_grid
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
    return math.sqrt(max(peak2, peak_val_sq))

def _full_peak_eval(coeffs, M=512):
    """Evaluate max |P(z)| over dense grid and refine top peaks."""
    theta = 2.0 * np.pi * np.arange(M) / M
    vals = _eval_poly(coeffs, theta)
    mag2 = np.abs(vals) ** 2
    peak_idx = np.argmax(mag2)
    peak_sq = mag2[peak_idx]
    phase = theta[peak_idx]
    refined = _refine_peak(coeffs, peak_sq, phase)
    return refined * refined

def construct():
    global _T0
    _T0 = time.time()
    
    # Phase 1: Coarse Brute Force
    M_coarse = 64
    theta_coarse = 2.0 * np.pi * np.arange(M_coarse) / M_coarse
    z_coarse = np.exp(1j * theta_coarse)
    Z_coarse = np.power(z_coarse[:, None], np.arange(_N)[None, :])
    
    # We want to find the top K candidates based on coarse peak
    # Since we can't hold all 2^24 peaks in memory easily if we want top K without sorting huge arrays,
    # we can just keep track of the best ones found so far.
    # But 2^24 is 16M. Storing 16M floats is 128MB. This is fine.
    # However, sorting 16M floats takes time.
    # Let's just find the single best in each batch and keep a list of top candidates?
    # Or, since the distribution is tight, we can use a threshold.
    
    # Better approach:
    # Process in batches, find the min peak in each batch, and keep a list of (peak, index) pairs.
    # At the end, sort these pairs.
    
    B = 100000
    total = 1 << _N
    
    best_pairs = [] # List of (peak_sq, index)
    
    for start in range(0, total, B):
        if _time_left() < 20.0:
            break
        
        end = min(start + B, total)
        idxs = np.arange(start, end, dtype=np.uint32)
        bit_mask = 1 << np.arange(_N, dtype=np.uint32)
        bits = (idxs[:, None] & bit_mask[None, :]) > 0
        coeffs = (bits.astype(np.float64) * 2.0) - 1.0
        
        vals = coeffs @ Z_coarse.T
        mag2 = np.abs(vals) ** 2
        peaks = np.max(mag2, axis=1)
        
        # Find the best in this batch
        min_idx_local = np.argmin(peaks)
        min_peak_sq = peaks[min_idx_local]
        min_idx_global = start + min_idx_local
        
        best_pairs.append((min_peak_sq, min_idx_global))
        
        # Optional: If we have time, we can also check for other low peaks in this batch?
        # No, just the best is enough for now.
        
    # Sort by peak
    best_pairs.sort(key=lambda x: x[0])
    
    # Take the top 10 candidates
    top_candidates = [idx for _, idx in best_pairs[:10]]
    
    if not top_candidates:
        # Fallback if no time left for even one batch
        return [1] * _N

    # Phase 2: Fine Evaluation of Top Candidates
    M_fine = 512
    theta_fine = 2.0 * np.pi * np.arange(M_fine) / M_fine
    z_fine = np.exp(1j * theta_fine)
    Z_fine = np.power(z_fine[:, None], np.arange(_N)[None, :])
    
    best_idx = top_candidates[0]
    best_peak_sq = float('inf')
    best_coeffs = None
    
    for idx in top_candidates:
        bits = (idx & (1 << np.arange(_N, dtype=np.uint32))) > 0
        coeffs = (bits.astype(np.float64) * 2.0) - 1.0
        
        vals = coeffs @ Z_fine.T
        mag2 = np.abs(vals) ** 2
        peak_sq = np.max(mag2)
        
        if peak_sq < best_peak_sq:
            best_peak_sq = peak_sq
            best_idx = idx
            best_coeffs = coeffs
    
    # Phase 3: Local Search (Hill Climbing)
    if best_coeffs is not None:
        cur_peak_sq = _full_peak_eval(best_coeffs, M=512)
        
        improved = True
        while improved and _time_left() > 5.0:
            improved = False
            for j in range(_N):
                if _time_left() < 3.0:
                    break
                best_coeffs[j] = -best_coeffs[j]
                new_peak_sq = _full_peak_eval(best_coeffs, M=256)
                if new_peak_sq < cur_peak_sq:
                    cur_peak_sq = new_peak_sq
                    improved = True
                else:
                    best_coeffs[j] = -best_coeffs[j]
        
        result = [int(1.0 if c > 0 else -1.0) for c in best_coeffs]
        return result
    else:
        # Should not happen
        return [1] * _N
