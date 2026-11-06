"""Model-and-optimize for flattest plus-minus-one 24-coefficient polynomial.

P(z) = sum c_k z^k, c_k in {+1,-1}, |c| = 24.
Score = max_{|z|=1} |P(z)| / sqrt(24).  Optimum ~ 1.2583758 (max|P| ~ 6.1363).

Strategy: Simulated Annealing with FFT-based evaluation for speed.
1. Initialize with Rudin-Shapiro coefficients.
2. Run Simulated Annealing:
   - Flip random coefficient.
   - Evaluate peak using FFT (fast).
   - Accept based on Metropolis criterion.
3. Polish the best candidate with high-resolution local search.
"""
import math
import time
import numpy as np

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


def _peak_fft(coeffs, M):
    """Calculate peak magnitude using FFT on a grid of size M."""
    c = np.zeros(M)
    c[:_N] = coeffs
    vals = np.fft.fft(c)
    return np.max(np.abs(vals))


def _peak_refine(coeffs, M, phase, iters=30):
    """Golden section refinement of peak around phase."""
    lo = phase - math.pi / M
    hi = phase + math.pi / M
    c_arr = np.array(coeffs, dtype=complex)
    powers = np.arange(_N)

    def f(a):
        z = np.exp(1j * a)
        zk = np.power(z, powers)
        s = np.dot(c_arr, zk)
        return s.real * s.real + s.imag * s.imag

    f_lo = f(lo)
    f_hi = f(hi)
    for _ in range(iters):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        f1 = f(m1)
        f2 = f(m2)
        if f1 < f2:
            lo = m1
            f_lo = f1
        else:
            hi = m2
            f_hi = f2
    return math.sqrt(max(f_lo, f_hi))


def _score_final(coeffs, M=4096):
    """High precision score calculation."""
    k = np.arange(M)
    theta = 2.0 * np.pi * k / M
    z = np.exp(1j * theta)
    c = np.array(coeffs, dtype=complex)
    Z = np.power(z[:, None], np.arange(_N)[None, :])
    vals = Z @ c
    mag2 = np.abs(vals) ** 2
    best_idx = np.argmax(mag2)
    peak = math.sqrt(mag2[best_idx])
    # Refine the peak
    peak = _peak_refine(coeffs, M, theta[best_idx], iters=30)
    return peak / _SQRTN


def _simulate_annealing(seed, M, T_start, T_end, steps, rng):
    """Run simulated annealing."""
    cur = np.array(seed, dtype=np.int8)
    cur_peak = _peak_fft(cur, M)
    
    best = list(cur)
    best_peak = cur_peak
    
    # Cooling schedule: exponential
    ratio = T_end / T_start
    step_ratio = ratio ** (1.0 / steps)
    
    T = T_start
    
    for i in range(steps):
        if _time_left() < 10.0:
            break
            
        # Flip a random coefficient
        j = int(rng.integers(0, _N))
        cur[j] = -cur[j]
        
        new_peak = _peak_fft(cur, M)
        delta = new_peak - cur_peak
        
        if delta < 0 or (T > 1e-10 and rng.random() < math.exp(-delta / T)):
            cur_peak = new_peak
            if new_peak < best_peak:
                best_peak = new_peak
                best = list(cur)
        else:
            cur[j] = -cur[j]  # revert
            
        T *= step_ratio
        
    return best, best_peak


def _local_search_single_flip(coeffs, M, max_iters=100):
    """Single flip hill climbing until local minimum."""
    cur = np.array(coeffs, dtype=np.int8)
    cur_peak = _peak_fft(cur, M)
    
    improved = True
    iters = 0
    while improved and iters < max_iters and _time_left() > 2.0:
        improved = False
        for j in range(_N):
            cur[j] = -cur[j]
            p = _peak_fft(cur, M)
            if p < cur_peak:
                cur_peak = p
                improved = True
            else:
                cur[j] = -cur[j]
        iters += 1
    
    return list(cur), cur_peak


def _local_search_double_flip(coeffs, M, max_iters=50):
    """Double flip hill climbing to escape local minima."""
    cur = np.array(coeffs, dtype=np.int8)
    cur_peak = _peak_fft(cur, M)
    
    improved = True
    iters = 0
    while improved and iters < max_iters and _time_left() > 2.0:
        improved = False
        for j in range(_N):
            for k in range(j + 1, _N):
                cur[j] = -cur[j]
                cur[k] = -cur[k]
                p = _peak_fft(cur, M)
                if p < cur_peak:
                    cur_peak = p
                    improved = True
                else:
                    cur[j] = -cur[j]
                    cur[k] = -cur[k]
        iters += 1
    
    return list(cur), cur_peak


def _random_restart(base_coeffs, rng, M, T_start, T_end, steps):
    """Create a perturbed version and run SA."""
    cur = np.array(base_coeffs, dtype=np.int8)
    # Randomly flip 3-5 coefficients to escape local optimum
    n_flips = int(rng.integers(3, 6))
    for _ in range(n_flips):
        j = int(rng.integers(0, _N))
        cur[j] = -cur[j]
    
    return _simulate_annealing(list(cur), M, T_start, T_end, steps, rng)


def construct():
    global _T0
    _T0 = time.time()

    # Seed
    seed = _rs_poly()
    
    # Phase 1: Simulated Annealing with multiple restarts
    M_sa = 1024
    T_start = 0.5
    T_end = 0.01
    
    rng = np.random.default_rng(42)
    
    # First SA run
    best, best_peak = _simulate_annealing(seed, M_sa, T_start, T_end, 10**9, rng)
    
    # Try a few random restarts if time permits
    restart_count = 0
    while _time_left() > 30.0 and restart_count < 5:
        candidate, cand_peak = _random_restart(best, rng, M_sa, T_start * 0.5, T_end, 10**8)
        if cand_peak < best_peak:
            best, best_peak = candidate, cand_peak
        restart_count += 1
    
    # Phase 2: Polish with single flip local search
    if _time_left() > 15.0:
        M_polish = 2048
        best, best_peak = _local_search_single_flip(best, M_polish, max_iters=50)
    
    # Phase 3: Double flip to escape local minima
    if _time_left() > 10.0:
        M_polish = 2048
        best, best_peak = _local_search_double_flip(best, M_polish, max_iters=30)
    
    # Phase 4: Another single flip pass after double flip
    if _time_left() > 5.0:
        M_polish = 2048
        best, best_peak = _local_search_single_flip(best, M_polish, max_iters=30)
    
    # Final score calculation with high precision
    # The evaluator does this, but we return the coefficients.
    # We just need to return the best coefficients.
    
    return [1 if v > 0 else -1 for v in best]
