"""Search for the flattest ±1 polynomial of 24 coefficients (min sup norm / sqrt(24))."""
import time
import math
import numpy as np

_N = 24
_SQRTN = math.sqrt(_N)
_T0 = time.time()
_LIMIT = 110.0  # safety margin under the 120 s wall-clock kill

def _time_left():
    return _LIMIT - (time.time() - _T0)

def _rs_poly(n=_N):
    """Rudin-Shapiro ±1 sequence truncated to n coefficients."""
    p, q = [1], [1]
    while len(p) < n:
        p, q = p + q, p + [-c for c in q]
    return p[:n]

# Precompute FFT machinery for fast peak detection
_NFFT = 1 << 15  # 32768 sample points on the unit circle
_ANGLES = np.linspace(0.0, 2.0 * np.pi, _NFFT, endpoint=False)
_Z = np.exp(1j * _ANGLES)

def _peak_estimate(coeffs):
    """Estimate max |P(z)| on |z|=1 via dense FFT sampling + golden-section refinement."""
    c = np.zeros(_NFFT, dtype=complex)
    c[:len(coeffs)] = coeffs
    # P(z_j) = sum_k c_k exp(i 2pi j k / NFFT)  =>  this is the DFT of c (with sign convention)
    vals = np.fft.fft(c)  # vals[j] = sum_k c_k exp(+i 2pi j k / NFFT)  (numpy uses exp(-i...) for fft? check)
    # numpy fft: a_k = sum_n x_n exp(+2pi i k n / N)? Actually numpy uses exp(-2pi i k n / N).
    # We want P(z_j)=sum c_k exp(i 2pi j k/N). That's the *inverse* convention up to scaling.
    # Use ifft: ifft(c)[j] = (1/N) sum_k c_k exp(+2pi i j k/N). So N*ifft(c) = P(z_j).
    vals = _NFFT * np.fft.ifft(c)
    mag = np.abs(vals)
    # Find top peaks
    k = min(8, _NFFT)
    idx = np.argpartition(mag, -k)[-k:]
    best = 0.0
    for j in idx:
        # refine around angle _ANGLES[j]
        a0 = _ANGLES[j]
        best = max(best, _refine_peak(coeffs, a0))
    return best

def _refine_peak(coeffs, a0):
    """Golden-section maximize |P(a)| near a0 within +/- 2*pi/_NFFT."""
    def f(a):
        z = np.exp(1j * a)
        # Horner / direct
        v = 0j
        for ck in reversed(coeffs):
            v = v * z + ck
        return abs(v)
    h = 2.0 * np.pi / _NFFT
    lo, hi = a0 - h, a0 + h
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = hi - gr * (hi - lo)
    x2 = lo + gr * (hi - lo)
    f1, f2 = f(x1), f(x2)
    for _ in range(40):
        if f1 < f2:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + gr * (hi - lo)
            f2 = f(x2)
        else:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - gr * (hi - lo)
            f1 = f(x1)
    return max(f1, f2)

def _score(coeffs):
    return _peak_estimate(coeffs) / _SQRTN

def _local_search(coeffs, max_rounds=200):
    """Hill-climbing with single-flip then double-flip moves."""
    cur = list(coeffs)
    cur_val = _score(cur)
    improved = True
    while improved and _time_left() > 5.0:
        improved = False
        # single flip
        best_val = cur_val
        best_move = None
        for i in range(_N):
            trial = cur[:]
            trial[i] = -trial[i]
            v = _score(trial)
            if v < best_val - 1e-12:
                best_val = v
                best_move = (i,)
        if best_move is not None:
            cur[best_move[0]] = -cur[best_move[0]]
            cur_val = best_val
            improved = True
            continue
        # double flip
        best_val = cur_val
        best_move = None
        for i in range(_N):
            for j in range(i + 1, _N):
                trial = cur[:]
                trial[i] = -trial[i]
                trial[j] = -trial[j]
                v = _score(trial)
                if v < best_val - 1e-12:
                    best_val = v
                    best_move = (i, j)
                if _time_left() < 5.0:
                    break
            if _time_left() < 5.0:
                break
        if best_move is not None:
            cur[best_move[0]] = -cur[best_move[0]]
            cur[best_move[1]] = -cur[best_move[1]]
            cur_val = best_val
            improved = True
    return cur, cur_val

def construct():
    best = None
    best_val = float('inf')

    # Start from Rudin-Shapiro
    start = _rs_poly()
    c, v = _local_search(start)
    if v < best_val:
        best, best_val = c, v

    # Try a few other well-known starting points / random restarts if time allows
    rng = np.random.default_rng(12345)
    attempts = 0
    while _time_left() > 15.0 and attempts < 40:
        attempts += 1
        # random ±1 start
        s = [1 if rng.random() < 0.5 else -1 for _ in range(_N)]
        c, v = _local_search(s, max_rounds=30)
        if v < best_val:
            best, best_val = c, v

    # Also try the known-good candidates: all-ones with a few flips, and sign-flipped RS
    for variant in [list(start), [-x for x in start]]:
        if _time_left() > 15.0:
            c, v = _local_search(variant)
            if v < best_val:
                best, best_val = c, v

    if best is None:
        best = start

    # Final high-precision polish: re-evaluate with more refinement points
    # (ensure we return a valid list of 24 ±1 ints)
    return [1 if x > 0 else -1 for x in best]
