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
    powers = np.arange(_N)
    zk = np.power(z[:, None], powers[None, :])
    return zk @ coeffs

def _refine_peak(coeffs, peak_val_sq, phase, iters=30):
    """
    Refine the peak around a specific phase using golden section search.
    coeffs: array of 24 floats (+1/-1)
    peak_val_sq: current known peak magnitude squared (lower bound for return)
    phase: initial phase estimate in radians
    Returns: refined peak magnitude squared
    """
    def f(a):
        z = np.exp(1j * a)
        zk = z ** np.arange(_N)
        s = np.dot(coeffs, zk)
        return s.real * s.real + s.imag * s.imag

    # Window around phase
    d_phase = 2.0 * math.pi / 512.0
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

def _golay_rudin_shapiro(n):
    """
    Generate Golay-Rudin-Shapiro sequence of length n.
    The GRS sequence is defined recursively:
    g(2n) = g(n)
    g(2n+1) = 1 - g(n)
    Starting with g(0) = 1, g(1) = 1.
    Returns array of +1/-1 of length n.
    """
    if n <= 0:
        return np.array([], dtype=np.float64)
    
    # Build the sequence
    seq = [1]  # g(0)
    for i in range(1, n):
        if i % 2 == 0:
            # g(2k) = g(k)
            seq.append(seq[i // 2])
        else:
            # g(2k+1) = 1 - g(k)
            seq.append(1 - seq[i // 2])
    
    # Convert to +1/-1
    return np.array([2.0 * s - 1.0 for s in seq], dtype=np.float64)

def construct():
    global _T0
    _T0 = time.time()
    
    # Phase 1: Start with Golay-Rudin-Shapiro sequence
    cur_coeffs = _golay_rudin_shapiro(_N)
    cur_peak_sq = _full_peak_eval(cur_coeffs, M=4096)
    
    best_coeffs = cur_coeffs.copy()
    best_peak_sq = cur_peak_sq
    
    # Phase 2: Single-flip local search (hill climbing)
    improved = True
    max_passes = 20
    passes = 0
    
    while improved and _time_left() > 30.0 and passes < max_passes:
        improved = False
        passes += 1
        
        # Try flipping each coefficient
        for j in range(_N):
            if _time_left() < 25.0:
                break
                
            # Flip bit j
            cur_coeffs[j] = -cur_coeffs[j]
            
            # Evaluate new peak using moderate grid for speed
            new_peak_sq = _full_peak_eval(cur_coeffs, M=1024)
            
            if new_peak_sq < cur_peak_sq:
                cur_peak_sq = new_peak_sq
                improved = True
                if cur_peak_sq < best_peak_sq:
                    best_peak_sq = cur_peak_sq
                    best_coeffs = cur_coeffs.copy()
            else:
                # Revert
                cur_coeffs[j] = -cur_coeffs[j]
    
    # Phase 3: Double-flip local search to escape local optima
    improved = True
    max_double_passes = 10
    double_passes = 0
    
    while improved and _time_left() > 20.0 and double_passes < max_double_passes:
        improved = False
        double_passes += 1
        
        # Try flipping pairs of coefficients
        for i in range(_N):
            if _time_left() < 15.0:
                break
            for j in range(i + 1, _N):
                if _time_left() < 15.0:
                    break
                    
                # Flip bits i and j
                cur_coeffs[i] = -cur_coeffs[i]
                cur_coeffs[j] = -cur_coeffs[j]
                
                # Evaluate new peak
                new_peak_sq = _full_peak_eval(cur_coeffs, M=1024)
                
                if new_peak_sq < cur_peak_sq:
                    cur_peak_sq = new_peak_sq
                    improved = True
                    if cur_peak_sq < best_peak_sq:
                        best_peak_sq = cur_peak_sq
                        best_coeffs = cur_coeffs.copy()
                else:
                    # Revert
                    cur_coeffs[i] = -cur_coeffs[i]
                    cur_coeffs[j] = -cur_coeffs[j]
    
    # Phase 4: Final high-precision refinement of the best result
    final_peak_sq = _full_peak_eval(best_coeffs, M=8192)
    
    result = [int(1.0 if c > 0 else -1.0) for c in best_coeffs]
    return result
