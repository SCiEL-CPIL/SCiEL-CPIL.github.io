"""Flattest plus-minus-one polynomials with 24 coefficients (minimum sup norm)."""
import time
import math
import numpy as np

# Deadline guard: evaluator kills after 150s wall-clock; keep a safety margin.
_DEADLINE = 130.0  # seconds
_START_TIME = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _max_norm(coeffs):
    """Compute max |P(z)| over |z|=1 using dense sampling + golden-section refinement."""
    n = len(coeffs)
    # Dense initial sampling
    m = 4096
    theta = 2.0 * np.pi * np.arange(m) / m
    z = np.exp(1j * theta)
    # Evaluate P(z) = sum c_k z^k via FFT-based approach
    # Pad coeffs to length m, take FFT, which gives P(z_j) for z_j = exp(-2pi i j/m)
    # But we need z_j = exp(2pi i j/m), so use conjugate
    c_padded = np.zeros(m, dtype=complex)
    c_padded[:n] = coeffs
    # FFT: X_j = sum_k c_k exp(-2pi i j k / m) = P(exp(-2pi i j/m))
    # We want P(exp(2pi i j/m)) = conj(X_j) if c_k are real
    X = np.fft.fft(c_padded)
    vals = np.abs(X)  # |P(exp(-2pi i j/m))| = |P(exp(2pi i j/m))| since coeffs are real
    # Find top peaks
    top_k = 8
    idx = np.argsort(-vals)[:top_k]
    
    best = 0.0
    for i in idx:
        # Golden-section refinement around theta_i
        theta0 = 2.0 * np.pi * i / m
        a = theta0 - 2.0 * np.pi / m
        b = theta0 + 2.0 * np.pi / m
        # Golden-section search for max of |P(z)| in [a, b]
        gr = (math.sqrt(5.0) - 1.0) / 2.0  # golden ratio conjugate ~0.618
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = _eval_p(coeffs, c)
        fd = _eval_p(coeffs, d)
        for _ in range(60):
            if fc > fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = _eval_p(coeffs, c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = _eval_p(coeffs, d)
        mid = (a + b) / 2.0
        fm = _eval_p(coeffs, mid)
        if fm > best:
            best = fm
    return best

def _eval_p(coeffs, theta):
    """Evaluate P(exp(i*theta)) and return |P|."""
    n = len(coeffs)
    # Direct evaluation for small n
    k = np.arange(n)
    z = np.exp(1j * theta * k)
    return abs(np.dot(coeffs, z))

def _score(coeffs):
    """Score = max_norm / sqrt(24)."""
    return _max_norm(coeffs) / math.sqrt(24.0)

def _flip_bit(coeffs, pos):
    """Return a new list with bit at pos flipped."""
    c = list(coeffs)
    c[pos] = -c[pos]
    return c

def construct():
    """Return 24 coefficients in {+1, -1} minimizing max |P(z)| / sqrt(24)."""
    n = 24
    
    # Start from Golay-Rudin-Shapiro as a good initial guess
    p, q = [1], [1]
    while len(p) < n:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = p[:n]
    best_score = _score(best_coeffs)
    
    # Also try the conjugate (negate all) - same norm, skip
    # Try the "other" GRS sequence
    p2, q2 = [1], [1]
    while len(p2) < n:
        p2, q2 = p2 + q2, p2 + [-c for c in q2]
    # The standard GRS gives one; let's also try starting with [-1]
    p3, q3 = [-1], [1]
    while len(p3) < n:
        p3, q3 = p3 + q3, p3 + [-c for c in q3]
    cand3 = p3[:n]
    s3 = _score(cand3)
    if s3 < best_score:
        best_score = s3
        best_coeffs = cand3
    
    # Local search: try single-bit flips and double-bit flips
    # Use a greedy hill-climbing with restarts
    t0 = time.time()
    
    # Phase 1: Single-bit flip hill climbing
    improved = True
    while improved and _time_left() > 5.0:
        improved = False
        for pos in range(n):
            if _time_left() < 5.0:
                break
            flipped = _flip_bit(best_coeffs, pos)
            s = _score(flipped)
            if s < best_score - 1e-12:
                best_score = s
                best_coeffs = flipped
                improved = True
                break  # restart scan from beginning after improvement
    
    # Phase 2: Double-bit flip search (pairs)
    if _time_left() > 10.0:
        improved = True
        while improved and _time_left() > 5.0:
            improved = False
            for i in range(n):
                if _time_left() < 5.0:
                    break
                for j in range(i + 1, n):
                    if _time_left() < 5.0:
                        break
                    flipped = _flip_bit(best_coeffs, i)
                    flipped = _flip_bit(flipped, j)
                    s = _score(flipped)
                    if s < best_score - 1e-12:
                        best_score = s
                        best_coeffs = flipped
                        improved = True
                        break
                if improved:
                    break
    
    # Phase 3: Multi-start with random perturbations
    if _time_left() > 15.0:
        rng = np.random.RandomState(42)
        for restart in range(200):
            if _time_left() < 15.0:
                break
            # Start from best_coeffs and flip 2-5 random bits
            num_flips = rng.randint(2, 6)
            positions = rng.choice(n, size=num_flips, replace=False)
            curr = list(best_coeffs)
            for pos in positions:
                curr[pos] = -curr[pos]
            
            # Local hill-climb from this perturbation
            curr_score = _score(curr)
            c_improved = True
            while c_improved and _time_left() > 15.0:
                c_improved = False
                for pos in range(n):
                    if _time_left() < 15.0:
                        break
                    flipped = _flip_bit(curr, pos)
                    s = _score(flipped)
                    if s < curr_score - 1e-12:
                        curr_score = s
                        curr = flipped
                        c_improved = True
                        break
            
            if curr_score < best_score - 1e-12:
                best_score = curr_score
                best_coeffs = curr
    
    # Final: try triple-bit flips around the best
    if _time_left() > 8.0:
        improved = True
        while improved and _time_left() > 3.0:
            improved = False
            for i in range(n):
                if _time_left() < 3.0:
                    break
                for j in range(i + 1, n):
                    if _time_left() < 3.0:
                        break
                    for k in range(j + 1, n):
                        if _time_left() < 3.0:
                            break
                        flipped = _flip_bit(best_coeffs, i)
                        flipped = _flip_bit(flipped, j)
                        flipped = _flip_bit(flipped, k)
                        s = _score(flipped)
                        if s < best_score - 1e-12:
                            best_score = s
                            best_coeffs = flipped
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    
    # Return as a list of ints
    return [int(c) for c in best_coeffs]
