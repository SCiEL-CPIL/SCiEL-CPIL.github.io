"""Flattest plus-minus-one polynomials with 24 coefficients (minimum sup norm)."""
import time
import math
import numpy as np

# Deadline guard: evaluator kills after 150s wall-clock; keep a safety margin.
_DEADLINE = 130.0  # seconds
_START_TIME = time.time()
_GR = (math.sqrt(5.0) - 1.0) / 2.0  # golden ratio conjugate ~0.618

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _eval_p(coeffs, theta):
    """Evaluate |P(exp(i*theta))| using Horner's method (fast)."""
    # coeffs[0] + coeffs[1]*z + coeffs[2]*z^2 + ...
    # Horner: c0 + z*(c1 + z*(c2 + ...))
    # But our P(z) = sum c_k z^k, so P(z) = c0 + c1*z + c2*z^2 + ...
    # Horner from highest degree: c_{n-1} + z*(c_{n-2} + z*(...))
    n = len(coeffs)
    result = coeffs[n-1]
    z = np.exp(1j * theta)
    for k in range(n-2, -1, -1):
        result = result * z + coeffs[k]
    return abs(result)

def _eval_p_vec(coeffs, thetas):
    """Evaluate |P(exp(i*theta))| for multiple thetas using vectorized Horner."""
    n = len(coeffs)
    z = np.exp(1j * thetas)
    result = np.full(len(thetas), coeffs[n-1], dtype=complex)
    for k in range(n-2, -1, -1):
        result = result * z + coeffs[k]
    return np.abs(result)

def _max_norm_full(coeffs):
    """Compute max |P(z)| over |z|=1 using dense sampling + golden-section refinement (high accuracy)."""
    n = len(coeffs)
    m = 4096
    c_padded = np.zeros(m, dtype=complex)
    c_padded[:n] = coeffs
    X = np.fft.fft(c_padded)
    vals = np.abs(X)
    top_k = 8
    idx = np.argsort(-vals)[:top_k]
    
    best = 0.0
    for i in idx:
        theta0 = 2.0 * np.pi * i / m
        a = theta0 - 2.0 * np.pi / m
        b = theta0 + 2.0 * np.pi / m
        c = b - _GR * (b - a)
        d = a + _GR * (b - a)
        fc = _eval_p(coeffs, c)
        fd = _eval_p(coeffs, d)
        for _ in range(45):
            if fc > fd:
                b = d
                d = c
                fd = fc
                c = b - _GR * (b - a)
                fc = _eval_p(coeffs, c)
            else:
                a = c
                c = d
                fc = fd
                d = a + _GR * (b - a)
                fd = _eval_p(coeffs, d)
        mid = (a + b) / 2.0
        fm = _eval_p(coeffs, mid)
        if fm > best:
            best = fm
    return best

def _max_norm_fast(coeffs):
    """Fast screening: max over FFT grid only (no golden refinement)."""
    n = len(coeffs)
    m = 4096
    c_padded = np.zeros(m, dtype=complex)
    c_padded[:n] = coeffs
    X = np.fft.fft(c_padded)
    return np.max(np.abs(X))

def _score(coeffs):
    """Score = max_norm / sqrt(24) using full refinement."""
    return _max_norm_full(coeffs) / math.sqrt(24.0)

def _score_fast(coeffs):
    """Fast screening score (no golden refinement)."""
    return _max_norm_fast(coeffs) / math.sqrt(24.0)

def _flip_bit(coeffs, pos):
    """Return a new list with bit at pos flipped."""
    c = list(coeffs)
    c[pos] = -c[pos]
    return c

def _hill_climb_fast(coeffs, time_limit):
    """Hill-climb using fast screening score, return best coeffs found."""
    n = len(coeffs)
    curr = list(coeffs)
    curr_score = _score_fast(curr)
    improved = True
    while improved and _time_left() > time_limit:
        improved = False
        for pos in range(n):
            if _time_left() < time_limit:
                break
            flipped = _flip_bit(curr, pos)
            s = _score_fast(flipped)
            if s < curr_score - 1e-10:
                curr_score = s
                curr = flipped
                improved = True
                break
    return curr, curr_score

def construct():
    """Return 24 coefficients in {+1, -1} minimizing max |P(z)| / sqrt(24)."""
    n = 24
    
    # Start from Golay-Rudin-Shapiro as a good initial guess
    p, q = [1], [1]
    while len(p) < n:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = p[:n]
    
    # Also try the "other" GRS sequence
    p3, q3 = [-1], [1]
    while len(p3) < n:
        p3, q3 = p3 + q3, p3 + [-c for c in q3]
    cand3 = p3[:n]
    
    # Use fast score for initial comparison
    s1 = _score_fast(best_coeffs)
    s3 = _score_fast(cand3)
    if s3 < s1:
        best_coeffs = cand3
    
    # Phase 1: Fast hill-climbing (single-bit flips with screening score)
    best_coeffs, _ = _hill_climb_fast(best_coeffs, 5.0)
    
    # Phase 2: Fast double-bit flip search
    if _time_left() > 10.0:
        improved = True
        while improved and _time_left() > 5.0:
            improved = False
            curr_score = _score_fast(best_coeffs)
            for i in range(n):
                if _time_left() < 5.0:
                    break
                for j in range(i + 1, n):
                    if _time_left() < 5.0:
                        break
                    flipped = _flip_bit(best_coeffs, i)
                    flipped = _flip_bit(flipped, j)
                    s = _score_fast(flipped)
                    if s < curr_score - 1e-10:
                        curr_score = s
                        best_coeffs = flipped
                        improved = True
                        break
                if improved:
                    break
    
    # Phase 3: Multi-start with random perturbations (using fast score)
    if _time_left() > 20.0:
        rng = np.random.RandomState(42)
        best_fast_score = _score_fast(best_coeffs)
        for restart in range(500):
            if _time_left() < 20.0:
                break
            num_flips = rng.randint(2, 6)
            positions = rng.choice(n, size=num_flips, replace=False)
            curr = list(best_coeffs)
            for pos in positions:
                curr[pos] = -curr[pos]
            
            curr, curr_score = _hill_climb_fast(curr, 20.0)
            if curr_score < best_fast_score - 1e-10:
                best_fast_score = curr_score
                best_coeffs = curr
    
    # Phase 4: Triple-bit flips around the best (fast screening)
    if _time_left() > 10.0:
        improved = True
        while improved and _time_left() > 5.0:
            improved = False
            curr_score = _score_fast(best_coeffs)
            for i in range(n):
                if _time_left() < 5.0:
                    break
                for j in range(i + 1, n):
                    if _time_left() < 5.0:
                        break
                    for k in range(j + 1, n):
                        if _time_left() < 5.0:
                            break
                        flipped = _flip_bit(best_coeffs, i)
                        flipped = _flip_bit(flipped, j)
                        flipped = _flip_bit(flipped, k)
                        s = _score_fast(flipped)
                        if s < curr_score - 1e-10:
                            curr_score = s
                            best_coeffs = flipped
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    
    # Final: one more round of single-bit polish with full score
    if _time_left() > 5.0:
        improved = True
        while improved and _time_left() > 3.0:
            improved = False
            full_score = _score(best_coeffs)
            for pos in range(n):
                if _time_left() < 3.0:
                    break
                flipped = _flip_bit(best_coeffs, pos)
                s = _score(flipped)
                if s < full_score - 1e-12:
                    full_score = s
                    best_coeffs = flipped
                    improved = True
                    break
    
    # Return as a list of ints
    return [int(c) for c in best_coeffs]
