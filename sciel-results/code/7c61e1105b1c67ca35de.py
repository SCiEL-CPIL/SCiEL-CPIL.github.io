"""Spectral-shape autoconvolution minimizer for the C3 upper-bound task."""
import time

N = 400

# Fixed DCT-II orthonormal basis (mode k), precomputed once.
_dct_modes = None


def _basis():
    global _dct_modes
    if _dct_modes is None:
        import numpy as np
        n = np.arange(N)
        _dct_modes = np.sqrt(2.0 / N) * np.cos(np.pi * (n[None, :] + 0.5) * np.arange(N)[:, None])
    return _dct_modes


def _score(h):
    import numpy as np
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return float("inf")
    return abs(2.0 * N * float(np.max(np.convolve(h, h))) / (s * s))


def _value_from_parts(maxc, s):
    return abs(2.0 * N * maxc / (s * s))


def construct():
    import numpy as np
    import scipy.signal as sig

    deadline = time.time() + 140.0  # safety margin under the 150 s wall-clock kill
    base = np.ones(N, dtype=float)
    M = _basis()

    best_h = base.copy()
    best_val = _score(base)  # = 2.0

    # For mode k (k>=1) the perturbed central-lag numerator is (1 + 2*amp^2) * s^2/N.
    # Off-peak entries depend on the cross-correlation of the mode with itself.
    # We scan (mode, amplitude) and keep the best.

    modes = M[1:]  # exclude the DC (k=0) mode

    # Precompute the self autoconvolution of each candidate mode (full).
    # auto_k[l] = sum_i mode_k[i]*mode_k[i+l-...]; use np.convolve.
    autos = np.array([np.convolve(m, m, mode="full") for m in modes])  # shape (K, 2N-1)

    # central index for full convolve of length N with itself is N-1
    central = N - 1

    # Coarse-to-fine amplitude grid (per mode) to find the suppression sweet spot.
    amp_coarse = [0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.18, 0.22, 0.28, 0.35, 0.45, 0.60, 0.80]

    # Evaluate a candidate (mode idx, amp) quickly using the precomputed auto.
    def cand_value(k_idx, amp):
        # h = base + amp*mode ;  sum(h) = N (mode is zero-mean, orthogonal to DC)
        # conv(h,h) = conv(base,base) + amp*conv(base,mode) + amp*conv(mode,base) + amp^2*conv(mode,mode)
        # conv(base,base) is the flat triangular auto of ones.
        # conv(base,mode) and conv(mode,base): base=ones, conv(ones,mode)=cumsum-ish; compute once per mode.
        return _cand_value_cached(k_idx, amp)

    # Precompute conv(ones, mode) for each mode to avoid recompute.
    ones_auto_cross = np.array([np.convolve(base, m, mode="full") for m in modes])

    # flat auto of ones
    ones_auto = np.convolve(base, base, mode="full")

    # cache of (k,amp)->value not needed; just compute.
    def evaluate(k_idx, amp):
        a = autos[k_idx]
        x = ones_auto_cross[k_idx]
        c = ones_auto + amp * (x + x[::-1]) + amp * amp * a
        return _value_from_parts(float(np.max(c)), float(N))  # sum(h)=N since mode zero-mean

    improved = False
    # Stage 1: coarse scan over all modes and coarse amplitudes.
    for k_idx in range(modes.shape[0]):
        if time.time() > deadline:
            break
        for amp in amp_coarse:
            v = evaluate(k_idx, amp)
            if v < best_val - 1e-12:
                best_val = v
                best_h = base + amp * modes[k_idx]
                improved = True

    # Stage 2: refine around the best (mode, amp) with a fine local 1-D scan.
    if improved:
        # identify the best coarse (mode,amp) by re-scan of coarse grid
        bk, ba = 0, amp_coarse[0]
        for k_idx in range(modes.shape[0]):
            for amp in amp_coarse:
                v = evaluate(k_idx, amp)
                if v < best_val - 1e-12:
                    best_val = v
                    bk, ba = k_idx, amp
        # fine refinement over amplitude near ba for that mode
        lo, hi = max(0.0, ba - 0.06), ba + 0.06
        fine = np.linspace(lo, hi, 61)
        for amp in fine:
            if time.time() > deadline:
                break
            v = evaluate(bk, float(amp))
            if v < best_val - 1e-12:
                best_val = v
                best_h = base + float(amp) * modes[bk]

    # Final safety: ensure finiteness and nonzero sum.
    if (not np.all(np.isfinite(best_h))) or (abs(float(np.sum(best_h))) < 1e-12):
        return base.copy()
    return best_h
