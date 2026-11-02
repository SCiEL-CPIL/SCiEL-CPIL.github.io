"""Tuned Fourier-invariant Hermite-Gaussian test function.

Minimizes the upper bound on A(f)*A(Fourier(f)) over three free Hermite
coefficients (degrees 0, 4, 8). The degree-12 weight is fixed so g(0)=0.
Uses critical-point analysis plus multi-phase optimization with tighter
surrogates and a high-precision local polish.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize

_SEED = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
_START = time.time()
_DEADLINE = 205.0
_MARGIN = 6.0


def _remaining():
    return _DEADLINE - _MARGIN - (time.time() - _START)


def _poly_coeffs(a):
    """Monomial coefficients of P(λ) = a0·H0 + a4·H4 + a8·H8 + a12·H12."""
    a0, a4, a8 = a
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 798336.0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0

    c = np.zeros(13)
    c[0] = a12 * 4096
    c[2] = a12 * (-245760)
    c[4] = a12 * 5529600
    c[6] = a12 * (-45158400)
    c[8] = a12 * 164430720
    c[10] = a12 * (-222772500)
    c[12] = a12 * 798336
    c[4] += a8 * 256
    c[6] += a8 * (-4480)
    c[8] += a8 * 26880
    c[10] += a8 * (-6720)
    c[12] += a8 * 1680
    c[8] += a4 * 16
    c[10] += a4 * (-48)
    c[12] += a4 * 12
    c[12] += a0
    return c


def _g(a, lam):
    """g(λ) = P(λ)·exp(-λ²/2) evaluated via Horner."""
    c = _poly_coeffs(a)
    lam = np.asarray(lam, dtype=float)
    p = np.zeros_like(lam)
    for coeff in c:
        p = p * lam + coeff
    return p * np.exp(-0.5 * lam * lam)


def _crit_pts(a):
    """Critical points of g: P'(λ) - λ·P(λ) = 0."""
    c = _poly_coeffs(a)
    # Q(λ) = P'(λ) - λ·P(λ), degree 13
    Q = np.zeros(14)
    for i in range(13):
        Q[i] = -c[i]
    for i in range(12):
        Q[i + 2] += (12 - i) * c[i]
    try:
        roots = np.roots(Q)
    except Exception:
        return np.array([])
    real = []
    for r in roots:
        if abs(r.imag) < 1e-10:
            re = float(r.real)
            if re >= -1e-10:
                real.append(max(0.0, re))
    return np.array(real)


def _bound(a, n_grid=20001, lam_max=25.0, local_n=5001, local_w=0.02):
    """Max of (1-g)/g over λ≥0 using critical points + grid + local refine."""
    crits = _crit_pts(a)
    grid = np.linspace(0.0, lam_max, n_grid)
    pts = np.concatenate([grid, crits]) if len(crits) else grid
    g = _g(a, pts)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-14, ratio, 1e18)
    ratio = np.where(np.isfinite(ratio), ratio, 1e18)
    ratio = np.maximum(ratio, 0.0)
    val = float(np.max(ratio))
    x0 = float(pts[int(np.argmax(ratio))])

    # Local refinement
    if 0.0 < x0 < lam_max:
        lo = max(0.0, x0 - local_w)
        hi = x0 + local_w
        loc = np.linspace(lo, hi, local_n)
        gl = _g(a, loc)
        with np.errstate(divide="ignore", invalid="ignore"):
            rl = (1.0 - gl) / gl
        rl = np.where(gl > 1e-14, rl, 1e18)
        rl = np.where(np.isfinite(rl), rl, 1e18)
        rl = np.maximum(rl, 0.0)
        val = max(val, float(np.max(rl)))
    return val


def _soft_max(a, temp=0.005, n=25001, lam_max=22.0):
    """Smooth surrogate: soft-max of (1-g)/g via log-sum-exp."""
    grid = np.linspace(0.0, lam_max, n)
    g = _g(a, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-14, ratio, 1e18)
    ratio = np.where(np.isfinite(ratio), ratio, 1e18)
    ratio = np.clip(ratio, 1e-12, 1e14)
    m = float(np.max(ratio))
    s = np.exp((ratio - m) / temp)
    return m + temp * math.log(float(np.mean(s)))


def _soft_max_tight(a, temp=0.001, n=40001, lam_max=20.0):
    """Tighter soft-max for final polish."""
    grid = np.linspace(0.0, lam_max, n)
    g = _g(a, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-14, ratio, 1e18)
    ratio = np.where(np.isfinite(ratio), ratio, 1e18)
    ratio = np.clip(ratio, 1e-12, 1e14)
    m = float(np.max(ratio))
    s = np.exp((ratio - m) / temp)
    return m + temp * math.log(float(np.mean(s)))


def construct():
    global _START
    _START = time.time()
    best_a = _SEED.copy()
    best_val = _bound(_SEED)

    # ---- Phase 1: Global multi-start on soft surrogate ----
    rng = np.random.default_rng(12345)
    starts = [_SEED.copy()]
    for sc in (0.05, 0.1, 0.2, 0.4, 0.8):
        starts.append(_SEED + sc * 0.02 * rng.standard_normal(3))
    for _ in range(12):
        starts.append(rng.standard_normal(3) * 0.05)
    # Structured starts
    for a0 in (-0.2, -0.5, -0.8, -1.13, -1.5, -2.0):
        for a4 in (0.005, 0.02, 0.04, 0.08):
            starts.append(np.array([a0, a4, a4 * 0.008]))
    starts.append(np.array([0.0, 0.0, 0.0]))

    for s in starts:
        if _remaining() < 10.0:
            break
        try:
            res = minimize(
                _soft_max, s, method="L-BFGS-B",
                options={"maxiter": 150, "ftol": 1e-14, "gtol": 1e-10},
            )
        except Exception:
            continue
        cand = np.asarray(res.x, dtype=float)
        if not np.all(np.isfinite(cand)):
            continue
        v = _bound(cand)
        if v < best_val:
            best_val = v
            best_a = cand.copy()

    # ---- Phase 2: Tighter soft-max polish ----
    if _remaining() > 10.0:
        try:
            res = minimize(
                _soft_max_tight, best_a, method="L-BFGS-B",
                options={"maxiter": 300, "ftol": 1e-15, "gtol": 1e-12},
            )
            cand = np.asarray(res.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _bound(cand)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass

    # ---- Phase 3: Nelder-Mead on exact bound ----
    if _remaining() > 8.0:
        try:
            res = minimize(
                _bound, best_a, method="Nelder-Mead",
                options={"maxiter": 600, "xatol": 1e-11, "fatol": 1e-13,
                         "adaptive": True, "maxfev": 2000},
            )
            cand = np.asarray(res.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _bound(cand)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass

    # ---- Phase 4: Coordinate descent polish ----
    if _remaining() > 4.0:
        step_sizes = [0.001, 0.0003, 0.0001, 0.00003, 0.00001]
        for ss in step_sizes:
            if _remaining() < 1.0:
                break
            improved = True
            while improved and _remaining() > 1.0:
                improved = False
                for axis in range(3):
                    for direction in (-1, 1):
                        cand = best_a.copy()
                        cand[axis] += direction * ss
                        v = _bound(cand)
                        if v < best_val - 1e-14:
                            best_val = v
                            best_a = cand.copy()
                            improved = True

    # ---- Phase 5: Fine grid around best ----
    if _remaining() > 3.0:
        step = 0.0002
        for i in (-4, -3, -2, -1, 0, 1, 2, 3, 4):
            for j in (-4, -3, -2, -1, 0, 1, 2, 3, 4):
                for k in (-4, -3, -2, -1, 0, 1, 2, 3, 4):
                    if _remaining() < 0.5:
                        break
                    cand = best_a + step * np.array([i, j, k], dtype=float)
                    v = _bound(cand)
                    if v < best_val:
                        best_val = v
                        best_a = cand.copy()

    # ---- Phase 6: Ultra-fine local search ----
    if _remaining() > 2.0:
        step = 0.00005
        for i in (-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6):
            for j in (-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6):
                for k in (-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6):
                    if _remaining() < 0.3:
                        break
                    cand = best_a + step * np.array([i, j, k], dtype=float)
                    v = _bound(cand)
                    if v < best_val:
                        best_val = v
                        best_a = cand.copy()

    # ---- Phase 7: Final high-precision bound evaluation ----
    if _remaining() > 1.0:
        final_val = _bound(best_a, n_grid=50001, lam_max=30.0,
                           local_n=20001, local_w=0.05)
        # If the high-res evaluation is better, keep it (it should be <=)

    return np.array(best_a, dtype=float)

# EVOLVE-BLOCK-END
