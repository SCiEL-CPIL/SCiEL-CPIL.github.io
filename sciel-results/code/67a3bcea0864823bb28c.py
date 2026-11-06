"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
_N = 24
_SQRTN = math.sqrt(_N)
_T0 = time.time()
_LIMIT = 120.0

def _time_left():
    return _LIMIT - (time.time() - _T0)

def _rs_poly():
    """Rudin-Shapiro coefficients truncated to _N."""
    p, q = [1], [1]
    while len(p) < _N:
        p, q = p + q, p + [-c for c in q]
    return p[:_N]

def _eval_grid(coeffs, n_pts):
    """Evaluate max |P(z)| on a uniform grid of n_pts points using FFT."""
    c = np.zeros(n_pts, dtype=complex)
    c[:len(coeffs)] = coeffs
    vals = np.fft.ifft(c) * n_pts
    return np.max(np.abs(vals))

def _eval_precise(coeffs, n_pts=4096):
    """
    Evaluate max |P(z)| with golden-section refinement around peaks.
    Returns the estimated sup norm.
    """
    # Dense sampling
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    c = np.zeros(n_pts, dtype=complex)
    c[:len(coeffs)] = coeffs
    vals = np.fft.ifft(c) * n_pts
    mag = np.abs(vals)
    
    # Find top peaks
    peak_indices = np.argsort(mag)[-12:]
    max_val = float(np.max(mag))
    
    # Golden section refinement around top peaks
    golden = (1 + math.sqrt(5)) / 2
    k_arr = np.arange(_N)
    
    for idx in peak_indices[:6]:
        a = angles[(idx - 1) % n_pts]
        b = angles[(idx + 1) % n_pts]
        # Ensure a < b
        if a > b:
            a, b = b, a
        
        # Golden section search
        for _ in range(40):
            m1 = b - (b - a) / golden
            m2 = a + (b - a) / golden
            f1 = np.abs(np.sum(coeffs * np.exp(1j * m1 * k_arr)))
            f2 = np.abs(np.sum(coeffs * np.exp(1j * m2 * k_arr)))
            if f1 < f2:
                b = m2
            else:
                a = m1
        f_mid = np.abs(np.sum(coeffs * np.exp(1j * (a + b) / 2 * k_arr)))
        if f_mid > max_val:
            max_val = f_mid
    
    return max_val

def _score_final(coeffs):
    """Compute the final score with golden-section refinement."""
    return _eval_precise(coeffs, n_pts=8192) / _SQRTN

def _local_search_single(coeffs, max_iter=200, n_pts=1024):
    """Local search: flip single coefficients to reduce peak."""
    coeffs = np.array(coeffs, dtype=float).copy()
    best_val = _eval_grid(coeffs, n_pts)
    k_arr = np.arange(_N)
    
    for _ in range(max_iter):
        if _time_left() < 10.0:
            break
        improved = False
        for k in range(_N):
            old = coeffs[k]
            coeffs[k] = -old
            val = _eval_grid(coeffs, n_pts)
            if val < best_val - 1e-9:
                best_val = val
                improved = True
            else:
                coeffs[k] = old
        
        if not improved:
            break
    
    return coeffs, best_val

def _local_search_double(coeffs, max_iter=100, n_pts=1024):
    """Local search: flip pairs of coefficients to reduce peak."""
    coeffs = np.array(coeffs, dtype=float).copy()
    best_val = _eval_grid(coeffs, n_pts)
    
    for _ in range(max_iter):
        if _time_left() < 10.0:
            break
        improved = False
        for i in range(_N):
            for j in range(i+1, _N):
                old_i, old_j = coeffs[i], coeffs[j]
                coeffs[i] = -old_i
                coeffs[j] = -old_j
                val = _eval_grid(coeffs, n_pts)
                if val < best_val - 1e-9:
                    best_val = val
                    improved = True
                else:
                    coeffs[i] = old_i
                    coeffs[j] = old_j
        
        if not improved:
            break
    
    return coeffs, best_val

def construct():
    # Start with Rudin-Shapiro
    coeffs = _rs_poly()
    coeffs_arr = np.array(coeffs, dtype=float)
    
    # Phase 1: Single-flip local search with moderate grid
    coeffs_arr, _ = _local_search_single(coeffs_arr, max_iter=100, n_pts=1024)
    
    # Phase 2: Double-flip local search to escape local optima
    coeffs_arr, _ = _local_search_double(coeffs_arr, max_iter=50, n_pts=1024)
    
    # Phase 3: More single-flip refinement with finer grid
    coeffs_arr, _ = _local_search_single(coeffs_arr, max_iter=50, n_pts=2048)
    
    # Phase 4: Another round of double-flip
    coeffs_arr, _ = _local_search_double(coeffs_arr, max_iter=30, n_pts=2048)
    
    # Phase 5: Final single-flip polish with even finer grid
    coeffs_arr, _ = _local_search_single(coeffs_arr, max_iter=30, n_pts=4096)
    
    # Convert to list of ints
    result = [1 if c > 0 else -1 for c in coeffs_arr]
    
    # Final check
    assert len(result) == _N
    return result
# EVOLVE-BLOCK-END
