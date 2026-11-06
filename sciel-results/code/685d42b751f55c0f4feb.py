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

def _refine_peak(coeffs, peak_val_sq, phase, iters=30):
    """
    Refine the peak around a specific phase using golden section search.
    coeffs: list of 24 floats (+1/-1)
    peak_val_sq: current known peak magnitude squared
    phase: initial phase estimate in radians
    """
    c_arr = np.array(coeffs, dtype=complex)
    powers = np.arange(_N)
    
    def f(a):
        z = np.exp(1j * a)
        zk = np.power(z, powers)
        s = np.dot(c_arr, zk)
        return (s.real * s.real + s.imag * s.imag)

    # Window around the phase. 
    # For M=4096, d_phase is ~ 0.0015. 
    # Golden section will narrow this down.
    M_grid = 4096
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
    # The refined peak should be >= the grid peak, but due to numerical issues or 
    # if the grid peak was not the local max, we take the max.
    if peak2 > peak_val_sq:
        return math.sqrt(peak2)
    return math.sqrt(peak_val_sq)

def construct():
    global _T0
    _T0 = time.time()
    
    # Stage 1: Coarse Brute Force
    # Use M=512 for the full scan. This is fast enough.
    M_coarse = 512
    k = np.arange(M_coarse)
    theta_coarse = 2.0 * np.pi * k / M_coarse
    z_coarse = np.exp(1j * theta_coarse)
    Z_coarse = np.power(z_coarse[:, None], np.arange(_N)[None, :])
    
    best_peak_sq = float('inf')
    best_coeff_idx = 0
    best_peak_phase = 0.0
    
    # Keep top K candidates from coarse search
    K = 100
    top_candidates = []  # List of (peak_sq, coeff_idx, peak_phase)
    
    B = 200000
    total = 1 << _N
    
    bit_mask = 1 << np.arange(_N, dtype=np.uint32)
    
    for start in range(0, total, B):
        if _time_left() < 10.0:
            break
        
        end = min(start + B, total)
        idxs = np.arange(start, end)
        bits = (idxs[:, None].astype(np.uint32) & bit_mask[None, :]) > 0
        coeffs = (bits.astype(np.float64) * 2.0) - 1.0
        
        vals = coeffs @ Z_coarse.T
        mag2 = np.abs(vals) ** 2
        peaks = np.max(mag2, axis=1)
        
        # Find top K in this batch
        # We want the smallest peaks (minimize sup norm)
        top_k_idx = np.argsort(peaks)[:K]
        
        for i in top_k_idx:
            peak_sq = peaks[i]
            coeff_idx = start + i
            peak_phase = theta_coarse[np.argmax(mag2[i])]
            
            if peak_sq < best_peak_sq:
                best_peak_sq = peak_sq
                best_coeff_idx = coeff_idx
                best_peak_phase = peak_phase
            
            # Maintain top K list sorted by peak_sq
            if len(top_candidates) < K or peak_sq < top_candidates[-1][0]:
                top_candidates.append((peak_sq, coeff_idx, peak_phase))
                top_candidates.sort(key=lambda x: x[0])
                if len(top_candidates) > K:
                    top_candidates.pop()
    
    # Stage 2: Fine Evaluation of Top Candidates
    M_fine = 4096
    k_fine = np.arange(M_fine)
    theta_fine = 2.0 * np.pi * k_fine / M_fine
    z_fine = np.exp(1j * theta_fine)
    Z_fine = np.power(z_fine[:, None], np.arange(_N)[None, :])
    
    best_fine_peak_sq = float('inf')
    best_fine_coeff = None
    best_fine_phase = 0.0
    
    # Sort top candidates by coarse peak to check best ones first
    top_candidates.sort(key=lambda x: x[0])
    
    for peak_sq, coeff_idx, peak_phase in top_candidates:
        if _time_left() < 5.0:
            break
        
        # Reconstruct coefficients
        bits = (coeff_idx & (1 << np.arange(_N, dtype=np.uint32))) > 0
        coeffs = (bits.astype(np.float64) * 2.0) - 1.0
        
        # Evaluate on fine grid
        vals = coeffs @ Z_fine.T
        mag2 = np.abs(vals) ** 2
        fine_peak_sq = np.max(mag2)
        fine_peak_phase = theta_fine[np.argmax(mag2)]
        
        if fine_peak_sq < best_fine_peak_sq:
            best_fine_peak_sq = fine_peak_sq
            best_fine_coeff = coeffs.copy()
            best_fine_phase = fine_peak_phase
            
    if best_fine_coeff is None:
        # Fallback if no candidates were processed
        bits = (best_coeff_idx & (1 << np.arange(_N, dtype=np.uint32))) > 0
        best_fine_coeff = (bits.astype(np.float64) * 2.0) - 1.0
        best_fine_peak_sq = best_peak_sq
        best_fine_phase = best_peak_phase

    # Stage 3: Refinement
    true_peak = _refine_peak(best_fine_coeff, best_fine_peak_sq, best_fine_phase)
    cur_peak_sq = true_peak * true_peak
    cur_coeffs = best_fine_coeff.copy()
    
    # Stage 4: Local Search
    # We use a hybrid check: first check coarse peak, if it improves, check fine+refine.
    # This is much faster than refining every neighbor.
    
    improved = True
    max_iters = 5
    
    for it in range(max_iters):
        if _time_left() < 5.0:
            break
        if not improved:
            break
        improved = False
        
        # Try single flips
        for j in range(_N):
            if _time_left() < 2.0:
                break
            
            cur_coeffs[j] = -cur_coeffs[j]
            
            # Quick coarse check
            vals_c = cur_coeffs @ Z_coarse.T
            mag2_c = np.abs(vals_c) ** 2
            coarse_peak_sq = np.max(mag2_c)
            
            # Only proceed if coarse peak is lower than current best
            # (allow some slack for grid error)
            if coarse_peak_sq < cur_peak_sq * 1.001: 
                # Fine check
                vals_f = cur_coeffs @ Z_fine.T
                mag2_f = np.abs(vals_f) ** 2
                fine_peak_sq = np.max(mag2_f)
                fine_peak_phase = theta_fine[np.argmax(mag2_f)]
                
                if fine_peak_sq < cur_peak_sq:
                    refined_peak = _refine_peak(cur_coeffs, fine_peak_sq, fine_peak_phase)
                    if refined_peak * refined_peak < cur_peak_sq:
                        cur_peak_sq = refined_peak * refined_peak
                        best_fine_coeff = cur_coeffs.copy()
                        best_fine_phase = fine_peak_phase
                        improved = True
                    else:
                        cur_coeffs[j] = -cur_coeffs[j]
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
                
                vals_c = cur_coeffs @ Z_coarse.T
                mag2_c = np.abs(vals_c) ** 2
                coarse_peak_sq = np.max(mag2_c)
                
                if coarse_peak_sq < cur_peak_sq * 1.001:
                    vals_f = cur_coeffs @ Z_fine.T
                    mag2_f = np.abs(vals_f) ** 2
                    fine_peak_sq = np.max(mag2_f)
                    fine_peak_phase = theta_fine[np.argmax(mag2_f)]
                    
                    if fine_peak_sq < cur_peak_sq:
                        refined_peak = _refine_peak(cur_coeffs, fine_peak_sq, fine_peak_phase)
                        if refined_peak * refined_peak < cur_peak_sq:
                            cur_peak_sq = refined_peak * refined_peak
                            best_fine_coeff = cur_coeffs.copy()
                            best_fine_phase = fine_peak_phase
                            improved = True
                        else:
                            cur_coeffs[j1] = -cur_coeffs[j1]
                            cur_coeffs[j2] = -cur_coeffs[j2]
                    else:
                        cur_coeffs[j1] = -cur_coeffs[j1]
                        cur_coeffs[j2] = -cur_coeffs[j2]
                else:
                    cur_coeffs[j1] = -cur_coeffs[j1]
                    cur_coeffs[j2] = -cur_coeffs[j2]

    result = [int(1.0 if c > 0 else -1.0) for c in best_fine_coeff]
    return result
