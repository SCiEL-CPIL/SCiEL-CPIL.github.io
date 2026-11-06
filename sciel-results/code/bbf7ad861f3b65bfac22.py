"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_t0 = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill


def _value(h):
    """Compute the exact objective value (uses direct convolve)."""
    h = np.asarray(h, dtype=float)
    if len(h) != N or not np.all(np.isfinite(h)):
        return np.inf
    s = float(np.sum(h))
    if s == 0.0:
        return np.inf
    conv = np.convolve(h, h)
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))


def _fft_conv_max(h):
    """Max of the full autocorrelation via FFT (fast, for optimization)."""
    size = 2 * N
    f = np.fft.rfft(h, size)
    conv = np.fft.irfft(f * np.conj(f), size)
    return float(np.max(conv))


def _fast_value(h):
    """Fast objective using FFT for the autocorrelation peak."""
    h = np.asarray(h, dtype=float)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return 1e12
    m = _fft_conv_max(h)
    return abs(2.0 * N * m / (s * s))


def construct():
    global _t0
    _t0 = time.time()

    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE

    best_h = np.ones(N)
    best_v = _value(best_h)

    def update_best(h):
        nonlocal best_h, best_v
        h = np.asarray(h, dtype=float)
        if len(h) != N or not np.all(np.isfinite(h)):
            return
        s = float(np.sum(h))
        if abs(s) < 1e-12:
            return
        v = _value(h)
        if v < best_v - 1e-12:
            best_v = v
            best_h = h.copy()

    # ---- Structural family 1: polynomial + low-frequency Fourier ----
    # h[i] = sum_k p[k] * (i/N)^k  +  sum_m (a[m] cos + b[m] sin)(2 pi m i / N)
    # This is a genuinely different modelling family (smooth, low-frequency)
    # from the localized Hamming atoms, and includes the constant (p[0]=1).
    i = np.arange(N)
    t = i / float(N)
    poly = np.stack([t ** k for k in range(5)], axis=1)          # (N,5)
    harm = np.arange(1, 5)
    cosb = np.stack([np.cos(2 * np.pi * harm * i / N) for _ in [0]], axis=0)
    cosb = np.stack([np.cos(2 * np.pi * m * i / N) for m in harm], axis=1)
    sinb = np.stack([np.sin(2 * np.pi * m * i / N) for m in harm], axis=1)
    # Build full design matrix: 5 poly + 4 cos + 4 sin = 13 params
    design = np.hstack([poly, cosb, sinb])                        # (N, 13)

    def _recon13(p):
        return p @ design

    def _obj13(p):
        h = _recon13(p)
        return _fast_value(h)

    rng = np.random.default_rng(12345)
    inits13 = [np.zeros(13)]
    inits13[0][0] = 1.0
    inits13.append(np.array([1.0, 0.0, 0.2, 0.0, 0.0, 0.1, 0.0, -0.1, 0.0, 0.0, 0.05, 0.0, -0.05]))
    inits13.append(np.array([1.0, 0.0, -0.15, 0.0, 0.0, -0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    for _ in range(6):
        init = np.zeros(13)
        init[0] = 1.0
        init[1:] = rng.standard_normal(12) * 0.15
        inits13.append(init)

    for init in inits13:
        if not deadline_ok():
            break
        try:
            res = minimize(
                _obj13, init, method='L-BFGS-B',
                bounds=[(-10, 10)] * 13,
                options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.x is not None:
                update_best(_recon13(res.x))
        except Exception:
            pass

    # ---- Structural family 2: localized Hamming atoms (current idea) ----
    atom_len = 10
    num_offsets = 15
    offsets = [o for o in range(0, N, N // num_offsets) if o + atom_len <= N]
    atom = np.hamming(atom_len)
    s_atom = atom.sum()
    if s_atom != 0:
        atom = atom / s_atom
    basis = []
    for sign in (1.0, -1.0):
        for off in offsets:
            v = np.zeros(N)
            v[off:off + atom_len] = sign * atom
            basis.append(v)
    basis_arr = np.array(basis)
    num_params = len(basis_arr)

    if num_params > 0:
        def _recon_atom(p):
            return p @ basis_arr

        def _obj_atom(p):
            return _fast_value(_recon_atom(p))

        inits_atom = [np.ones(num_params) / num_params]
        for _ in range(4):
            inits_atom.append(rng.standard_normal(num_params) * 0.1)
        for _ in range(3):
            init = np.zeros(num_params)
            idx = rng.choice(num_params, size=max(1, num_params // 4), replace=False)
            init[idx] = rng.standard_normal(len(idx)) * 0.5
            inits_atom.append(init)

        for init in inits_atom:
            if not deadline_ok():
                break
            try:
                res = minimize(
                    _obj_atom, init, method='L-BFGS-B',
                    bounds=[(-5, 5)] * num_params,
                    options={'maxiter': 250, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.x is not None:
                    update_best(_recon_atom(res.x))
            except Exception:
                pass

    # ---- Refinement: coordinate descent on the full 400-dim vector ----
    # Uses the fast FFT objective; a different search method (steepest-ish
    # coordinate descent) to polish whatever structural family found best.
    if deadline_ok():
        h = best_h.copy()
        cur = _fast_value(h)
        deltas = [0.05, -0.05, 0.2, -0.2, 0.8, -0.8]
        for _ in range(60):
            if not deadline_ok():
                break
            improved = False
            for idx in range(N):
                if not deadline_ok():
                    break
                old = h[idx]
                for d in deltas:
                    h[idx] = old + d
                    v = _fast_value(h)
                    if v < cur - 1e-12:
                        cur = v
                        improved = True
                        break
                h[idx] = old
            if not improved:
                break
        update_best(h)

    # ---- Final validation with the exact evaluator objective ----
    s = float(np.sum(best_h))
    if s == 0.0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    v_final = _value(best_h)
    if v_final < 2.0 - 1e-12:
        return best_h
    return np.ones(N)
# EVOLVE-BLOCK-END
