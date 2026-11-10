"""Flattest ±1 polynomial (24 coeffs) via A + z^12 B decomposition + fast DFT scoring."""
import time
import numpy as np

_T0 = time.time()
_HARD = 140.0
_N = 24
_HALF = 12
_M = 1 << _HALF          # 4096
_SQRT24 = float(np.sqrt(24))
_W = 2 * np.pi / _N


def _score(c):
    """max_{|z|=1} |P(z)| / sqrt(24), using DFT + dense golden-section refinement."""
    c = np.asarray(c, dtype=np.float64)
    dft = np.fft.fft(c, n=_N)
    mag = np.abs(dft)
    i0 = int(np.argmax(mag))
    best = mag[i0]
    for j in range(_N):
        a = (j - 0.5) * _W
        b = (j + 0.5) * _W
        # golden-section search on |P| over [a, b]
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        x1 = b - gr * (b - a)
        x2 = a + gr * (b - a)
        f1 = np.abs(np.dot(c, np.exp(1j * x1 * np.arange(_N))))
        f2 = np.abs(np.dot(c, np.exp(1j * x2 * np.arange(_N))))
        for _ in range(50):
            if f1 < f2:
                b, x2, f2 = x2, x1, f1
                x1 = b - gr * (b - a)
                f1 = np.abs(np.dot(c, np.exp(1j * x1 * np.arange(_N))))
            else:
                a, x1, f1 = x1, x2, f2
                x2 = a + gr * (b - a)
                f2 = np.abs(np.dot(c, np.exp(1j * x2 * np.arange(_N))))
        fm = np.abs(np.dot(c, np.exp(1j * 0.5 * (a + b) * np.arange(_N))))
        if fm > best:
            best = fm
    return best / _SQRT24


def construct():
    # ---- precompute DFT of all 2^12 half-vectors (±1) ----
    signs = np.array([1.0, -1.0], dtype=np.float64)
    idx = np.arange(_M)
    bits = (idx[:, None] >> np.arange(_HALF)[None, :]) & 1
    sv = 1.0 - 2.0 * bits                       # (4096,12) in {-1,1}
    F = np.fft.fft(sv, n=_N, axis=1)            # (4096,24)
    # P = A + z^12 B  =>  DFT(P)[k] = F_A[k] + F_B[k] * exp(-2j pi k/24)
    w12 = np.exp(-2j * np.pi * np.arange(_N) / _N)
    F2 = F * w12[None, :]

    def combo(a, b):
        return np.abs(F[a] + F2[b]).max()

    # ---- greedy bit-flip local search (fast, on the DFT) ----
    def greedy(a, b, iters):
        best = combo(a, b)
        for _ in range(iters):
            improved = False
            # flip a bit in A (index a)
            for bit in range(_HALF):
                a2 = a ^ (1 << bit)
                v = combo(a2, b)
                if v < best:
                    a, best = a2, v
                    improved = True
            # flip a bit in B (index b)
            for bit in range(_HALF):
                b2 = b ^ (1 << bit)
                v = combo(a, b2)
                if v < best:
                    b, best = b2, v
                    improved = True
            if not improved:
                break
        return a, b, best

    # ---- seed initial (A,B) from Golay-Rudin-Shapiro ----
    p, q = [1], [1]
    while len(p) < _N:
        p, q = p + q, p + [-c for c in q]
    p = p[:_N]
    A0 = int(sum((1 if p[k] > 0 else 0) << k for k in range(_HALF)))
    B0 = int(sum((1 if p[_HALF + k] > 0 else 0) << k for k in range(_HALF)))

    # ---- many fast restarts to escape local minima ----
    best_a, best_b, best_raw = A0, B0, combo(A0, B0)
    rng = np.random.default_rng(20240517)
    iters = 60
    while time.time() - _T0 < _HARD:
        a = int(rng.integers(0, _M))
        b = int(rng.integers(0, _M))
        a, b, raw = greedy(a, b, iters)
        if raw < best_raw:
            best_a, best_b, best_raw = a, b, raw

    # ---- convert best (A,B) indices back to ±1 coefficients ----
    def idx_to_coeffs(i):
        return [1.0 if ((i >> k) & 1) == 0 else -1.0 for k in range(_HALF)]

    A = idx_to_coeffs(best_a)
    B = idx_to_coeffs(best_b)
    coeffs = A + B                       # length 24, in {-1,1}
    return [int(round(x)) for x in coeffs]
