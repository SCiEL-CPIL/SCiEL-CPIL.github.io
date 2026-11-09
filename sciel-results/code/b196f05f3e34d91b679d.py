# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
from scipy.optimize import minimize
from math import lgamma, pi

_K = 10
_NPHI = 130
_NTH = 70
_NPTS = _NTH * _NPHI
_NC = 2 * _K + 1
_NZ = _K + 1  # 11 complex amplitudes: z_0, z_1, ..., z_10
_NZ_REAL = 2 * _NZ  # 22 real dimensions


def _build_quadrature():
    x, w = np.polynomial.legendre.leggauss(_NTH)
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI
    W = np.outer(w, np.full(_NPHI, dphi))

    Y = np.zeros((_NTH, _NPHI, _NC))
    for idx, m in enumerate(range(-_K, _K + 1)):
        am = abs(m)
        log_ratio = lgamma(_K - am + 1) - lgamma(_K + am + 1)
        fact_ratio = np.exp(log_ratio)
        Nm = np.sqrt((2 * _K + 1) / (4.0 * np.pi) * fact_ratio)
        P = lpmv(am, _K, x)
        if m == 0:
            Y[:, :, idx] = Nm * P[:, None]
        elif m > 0:
            Y[:, :, idx] = np.sqrt(2.0) * Nm * P[:, None] * np.cos(m * phi)[None, :]
        else:
            Y[:, :, idx] = np.sqrt(2.0) * Nm * P[:, None] * np.sin(am * phi)[None, :]
    return Y, W


_Y, _W = _build_quadrature()
_Wflat = _W.ravel()
_Yflat = _Y.reshape(_NPTS, _NC)


def _z_to_c(z):
    """z is complex array of length _NZ. z[0]=c_0, z[m]=c_{-m}+i*c_m for m>=1."""
    c = np.zeros(_NC)
    c[_K] = z[0].real
    for m in range(1, _K + 1):
        c[_K - m] = z[m].real
        c[_K + m] = z[m].imag
    return c


def _c_to_z(c):
    """c is 21-vector. Returns complex array of length _NZ."""
    z = np.zeros(_NZ, dtype=np.complex128)
    z[0] = c[_K]
    for m in range(1, _K + 1):
        z[m] = c[_K - m] + 1j * c[_K + m]
    return z


def _zr_to_c(zr):
    """zr is real array of length _NZ_REAL. zr[0]=c_0, zr[2m-1]=Re(z_m), zr[2m]=Im(z_m)."""
    c = np.zeros(_NC)
    c[_K] = zr[0]
    for m in range(1, _K + 1):
        c[_K - m] = zr[2 * m - 1]
        c[_K + m] = zr[2 * m]
    return c


def _c_to_zr(c):
    """c is 21-vector. Returns real array of length _NZ_REAL."""
    zr = np.zeros(_NZ_REAL)
    zr[0] = c[_K]
    for m in range(1, _K + 1):
        zr[2 * m - 1] = c[_K - m]
        zr[2 * m] = c[_K + m]
    return zr


def _objective_c(c):
    """R^4 = <f^4> / <f^2>^2, scale invariant."""
    f = _Yflat @ c
    f2 = f * f
    l2sq = float(np.dot(_Wflat, f2))
    if l2sq <= 1e-30:
        return 1e30
    f4 = f2 * f2
    l4sq = float(np.dot(_Wflat, f4))
    return l4sq / (l2sq * l2sq)


def _gradient_c(c):
    """Gradient of R^4 w.r.t. c."""
    f = _Yflat @ c
    f2 = f * f
    f3 = f2 * f
    l2sq = float(np.dot(_Wflat, f2))
    if l2sq <= 1e-30:
        return np.zeros_like(c)
    l4sq = float(np.dot(_Wflat, f2 * f2))
    wf3 = _Wflat * f3
    wf = _Wflat * f
    grad_f3 = _Yflat.T @ wf3
    grad_f = _Yflat.T @ wf
    B = l2sq
    A = l4sq
    grad = (4.0 / (B * B)) * grad_f3 - (4.0 * A / (B * B * B)) * grad_f
    return grad


def _objective_zr(zr):
    c = _zr_to_c(zr)
    return _objective_c(c)


def _gradient_zr(zr):
    c = _zr_to_c(zr)
    gc = _gradient_c(c)
    gzr = np.zeros(_NZ_REAL)
    gzr[0] = gc[_K]
    for m in range(1, _K + 1):
        gzr[2 * m - 1] = gc[_K - m]
        gzr[2 * m] = gc[_K + m]
    return gzr


def _coordinate_descent(zr, max_sweeps=50, deadline=None):
    """Coordinate descent on the 22 real components of the complex amplitude vector.
    For each coordinate, do a 1D line search (golden section or parabolic)."""
    best_val = _objective_zr(zr)
    best_zr = zr.copy()
    n = len(zr)

    for sweep in range(max_sweeps):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        improved = False
        for i in range(n):
            if deadline is not None and time.time() > deadline - 3.0:
                break
            # Try small perturbations in coordinate i
            x = best_zr.copy()
            step = 0.1 * (1.0 + abs(x[i]))
            best_local = best_val
            best_xi = x[i]
            # Scan a few steps in both directions
            for sign in [1, -1]:
                for mult in [1.0, 0.5, 0.25, 0.1, 2.0, 4.0]:
                    x[i] = x[i] + sign * step * mult
                    v = _objective_zr(x)
                    if v < best_local:
                        best_local = v
                        best_xi = x[i]
            if best_xi != x[i]:
                x[i] = best_xi
                # Refine with smaller steps around best_xi
                x0 = x[i]
                small_step = step * 0.1
                for sign in [1, -1]:
                    for mult in [1.0, 0.5, 0.25, 0.1]:
                        x[i] = x0 + sign * small_step * mult
                        v = _objective_zr(x)
                        if v < best_local:
                            best_local = v
                            best_xi = x[i]
                x[i] = best_xi
                if best_local < best_val - 1e-15:
                    best_val = best_local
                    best_zr = x.copy()
                    improved = True
        if not improved:
            break

    return best_zr, best_val


def _local_optimize(zr0, deadline):
    """Local optimization from a starting point in the 22D zr space."""
    zr = np.asarray(zr0, dtype=np.float64).copy()
    nrm = np.linalg.norm(zr)
    if nrm < 1e-14:
        return None
    zr = zr / nrm

    best_zr = zr.copy()
    best_val = _objective_zr(zr)

    # Phase 1: Coordinate descent
    try:
        zr_cd, val_cd = _coordinate_descent(zr, max_sweeps=30, deadline=deadline)
        if val_cd < best_val:
            best_val = val_cd
            best_zr = zr_cd.copy()
    except Exception:
        pass

    if time.time() > deadline - 30.0:
        c = _zr_to_c(best_zr)
        nrm = np.linalg.norm(c)
        if nrm < 1e-14:
            return None
        c = c / nrm
        return c, _objective_c(c)

    # Phase 2: L-BFGS-B on the 22D zr space (unconstrained, scale invariant)
    try:
        res = minimize(_objective_zr, best_zr, jac=_gradient_zr, method="L-BFGS-B",
                       options={"maxiter": 300, "maxfun": 2000,
                                "ftol": 1e-15, "gtol": 1e-10})
        if res.fun < best_val:
            best_val = res.fun
            best_zr = res.x.copy()
    except Exception:
        pass

    # Re-normalize
    nrm = np.linalg.norm(best_zr)
    if nrm < 1e-14:
        return None
    best_zr = best_zr / nrm
    best_val = _objective_zr(best_zr)

    if time.time() > deadline - 15.0:
        c = _zr_to_c(best_zr)
        nrm = np.linalg.norm(c)
        if nrm < 1e-14:
            return None
        c = c / nrm
        return c, _objective_c(c)

    # Phase 3: Nelder-Mead polish
    try:
        res2 = minimize(_objective_zr, best_zr, method="Nelder-Mead",
                        options={"maxiter": 3000, "xatol": 1e-12,
                                 "fatol": 1e-14, "adaptive": True})
        if res2.fun < best_val:
            best_val = res2.fun
            best_zr = res2.x.copy()
    except Exception:
        pass

    nrm = np.linalg.norm(best_zr)
    if nrm < 1e-14:
        return None
    best_zr = best_zr / nrm
    c = _zr_to_c(best_zr)
    nrm_c = np.linalg.norm(c)
    if nrm_c < 1e-14:
        return None
    c = c / nrm_c
    return c, _objective_c(c)


def _make_structured_seeds():
    """Generate diverse structured seeds in the 22D zr space.
    Each seed is a real vector of length 22 representing [c0, Re(z1), Im(z1), ..., Re(z10), Im(z10)]."""
    seeds = []

    # Seed 0: m=0 only
    zr = np.zeros(_NZ_REAL)
    zr[0] = 1.0
    seeds.append(zr)

    # Seed 1: m=10 cos only
    zr = np.zeros(_NZ_REAL)
    zr[2 * 10] = 1.0  # Im(z_10) = c_10
    seeds.append(zr)

    # Seed 2: m=10 sin only
    zr = np.zeros(_NZ_REAL)
    zr[2 * 10 - 1] = 1.0  # Re(z_10) = c_{-10}
    seeds.append(zr)

    # Seed 3: all m equal amplitude, zero phase
    zr = np.zeros(_NZ_REAL)
    zr[0] = 1.0
    for m in range(1, _K + 1):
        zr[2 * m] = 1.0  # Im(z_m) = c_m
    seeds.append(zr)

    # Seed 4: all m equal amplitude, 45-degree phase
    zr = np.zeros(_NZ_REAL)
    zr[0] = 1.0
    for m in range(1, _K + 1):
        zr[2 * m - 1] = 0.70710678
        zr[2 * m] = 0.70710678
    seeds.append(zr)

    # Seed 5: only even m
    zr = np.zeros(_NZ_REAL)
    zr[0] = 1.0
    for m in range(2, _K + 1, 2):
        zr[2 * m] = 1.0
    seeds.append(zr)

    # Seed 6: only odd m
    zr = np.zeros(_NZ_REAL)
    zr[0] = 1.0
    for m in range(1, _K + 1, 2):
        zr[2 * m] = 1.0
    seeds.append(zr)

    # Seed 7: m=0 and m=10
    zr = np.zeros(_NZ_REAL)
    zr[0] = 1.0
    zr[2 * 10] = 1.0
    seeds.append(zr)

    # Seed 8: m=0 and m=5
    zr = np.zeros(_NZ_REAL)
    zr[0] = 1.0
    zr[2 * 5] = 1.0
    seeds.append(zr)

    # Seed 9: alternating signs on m
    zr = np.zeros(_NZ_REAL)
    zr[0] = 1.0
    for m in range(1, _K + 1):
        zr[2 * m] = (-1) ** m
    seeds.append(zr)

    # Seed 10: linearly increasing amplitude
    zr = np.zeros(_NZ_REAL)
    zr[0] = 1.0
    for m in range(1, _K + 1):
        zr[2 * m] = m / _K
    seeds.append(zr)

    # Seed 11: linearly decreasing amplitude
    zr = np.zeros(_NZ_REAL)
    zr[0] = 1.0
    for m in range(1, _K + 1):
        zr[2 * m] = (_K - m + 1) / _K
    seeds.append(zr)

    # Seed 12: Gaussian random
    rng = np.random.default_rng(123)
    zr = rng.standard_normal(_NZ_REAL)
    seeds.append(zr)

    # Seed 13: random with emphasis on low m
    zr = np.zeros(_NZ_REAL)
    for i in range(_NZ_REAL):
        zr[i] = rng.standard_normal() * (1.0 + 1.0 / (1.0 + i))
    seeds.append(zr)

    # Seed 14: random with emphasis on high m
    zr = np.zeros(_NZ_REAL)
    for i in range(_NZ_REAL):
        zr[i] = rng.standard_normal() * (0.5 + i / _NZ_REAL)
    seeds.append(zr)

    # Seed 15: cos(0.3*m) pattern
    zr = np.zeros(_NZ_REAL)
    zr[0] = 1.0
    for m in range(1, _K + 1):
        zr[2 * m] = np.cos(0.3 * m)
    seeds.append(zr)

    # Seed 16: sin(0.5*m) pattern
    zr = np.zeros(_NZ_REAL)
    zr[0] = 0.5
    for m in range(1, _K + 1):
        zr[2 * m] = np.sin(0.5 * m)
    seeds.append(zr)

    # Seed 17: only m=±10 with phase
    zr = np.zeros(_NZ_REAL)
    zr[2 * 10 - 1] = 0.70710678
    zr[2 * 10] = 0.70710678
    seeds.append(zr)

    # Seed 18: m=0, ±5, ±10
    zr = np.zeros(_NZ_REAL)
    zr[0] = 1.0
    zr[2 * 5] = 0.7
    zr[2 * 10] = 0.5
    seeds.append(zr)

    # Seed 19: m=0, ±2, ±4, ±6, ±8, ±10 (even m, decreasing)
    zr = np.zeros(_NZ_REAL)
    zr[0] = 1.0
    for m in range(2, _K + 1, 2):
        zr[2 * m] = 1.0 - 0.1 * m
    seeds.append(zr)

    # Seed 20: random phases, equal amplitudes
    zr = np.zeros(_NZ_REAL)
    zr[0] = 1.0
    for m in range(1, _K + 1):
        phase = rng.uniform(0, 2 * np.pi)
        zr[2 * m - 1] = np.cos(phase)
        zr[2 * m] = np.sin(phase)
    seeds.append(zr)

    # Seed 21: random, seeded differently
    rng2 = np.random.default_rng(456)
    zr = rng2.standard_normal(_NZ_REAL)
    seeds.append(zr)

    # Seed 22: sparse - only m=0, ±3, ±7
    zr = np.zeros(_NZ_REAL)
    zr[0] = 1.0
    zr[2 * 3] = 0.8
    zr[2 * 7] = 0.6
    seeds.append(zr)

    # Seed 23: m=0 and all m with 1/sqrt(m) weighting
    zr = np.zeros(_NZ_REAL)
    zr[0] = 1.0
    for m in range(1, _K + 1):
        zr[2 * m] = 1.0 / np.sqrt(m)
    seeds.append(zr)

    # Normalize all seeds
    normalized_seeds = []
    for zr in seeds:
        nrm = np.linalg.norm(zr)
        if nrm > 1e-14:
            normalized_seeds.append(zr / nrm)
    return normalized_seeds


def rudin_problem10() -> np.ndarray:
    start = time.time()
    deadline = start + 585.0  # safety margin for 630s kill

    seeds = _make_structured_seeds()
    best_c = None
    best_val = np.inf

    # Optimize from each structured seed
    for i, zr0 in enumerate(seeds):
        if time.time() > deadline - 40.0:
            break
        try:
            out = _local_optimize(zr0, deadline)
        except Exception:
            continue
        if out is None:
            continue
        c, val = out
        if val < best_val:
            best_val = val
            best_c = c.copy()

    # Use remaining time for additional random restarts in zr space
    rng = np.random.default_rng(789)
    n_extra = 0
    while time.time() < deadline - 30.0:
        zr0 = rng.standard_normal(_NZ_REAL)
        nrm = np.linalg.norm(zr0)
        if nrm < 1e-14:
            continue
        zr0 /= nrm
        try:
            out = _local_optimize(zr0, deadline)
        except Exception:
            continue
        if out is None:
            continue
        c, val = out
        if val < best_val:
            best_val = val
            best_c = c.copy()
        n_extra += 1

    if best_c is None:
        best_c = np.ones(_NC, dtype=np.float64)
        best_c /= np.linalg.norm(best_c)

    best_c = np.asarray(best_c, dtype=np.float64)
    if not np.all(np.isfinite(best_c)):
        best_c = np.ones(_NC, dtype=np.float64)
        best_c /= np.linalg.norm(best_c)
    mx = float(np.max(np.abs(best_c)))
    if mx < 1e-3:
        best_c = best_c * (1.0 / mx)
    if mx > 1e5:
        best_c = best_c * (1e5 / mx)
    return best_c


# EVOLVE-BLOCK-END
