# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
from scipy.optimize import minimize

_K = 10
_NPHI = 130
_NTH = 70
_NPTS = _NTH * _NPHI
_NC = 2 * _K + 1


def _build_quadrature():
    x, w = np.polynomial.legendre.leggauss(_NTH)
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI
    W = np.outer(w, np.full(_NPHI, dphi))

    Y = np.zeros((_NTH, _NPHI, _NC))
    for idx, m in enumerate(range(-_K, _K + 1)):
        am = abs(m)
        # N_m = sqrt((2K+1)/(4pi) * (K-am)!/(K+am)!)
        # compute (K-am)!/(K+am)! = 1 / prod_{n=K-am+1}^{K+am} n
        denom = 1.0
        for n in range(_K - am + 1, _K + am + 1):
            denom *= n
        fact_ratio = 1.0 / denom
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


def _ratio_from_coeffs(c):
    f = _Yflat @ c
    f2 = f * f
    l2sq = float(np.dot(_Wflat, f2))
    if l2sq <= 0.0:
        return np.inf
    l4sq = float(np.dot(_Wflat, f2 * f2))
    if l4sq <= 0.0:
        return 1.0
    return np.sqrt(l4sq) / np.sqrt(l2sq)


def _objective(c):
    """Minimize R = ||f||_4/||f||_2.  We minimize R^2 for smoothness.
    R^2 = <f^4> / <f^2>^2  (since ||f||_4^2 = <f^4>^{1/2}... wait)
    Actually ||f||_4 = (<f^4>)^{1/4}, ||f||_2 = (<f^2>)^{1/2}
    R = (<f^4>)^{1/4} / (<f^2>)^{1/2}
    R^2 = (<f^4>)^{1/2} / <f^2>
    R^4 = <f^4> / <f^2>^2
    Minimize R^4 = <f^4> / <f^2>^2.  This is homogeneous of degree 0 in c.
    """
    f = _Yflat @ c
    f2 = f * f
    l2sq = float(np.dot(_Wflat, f2))
    if l2sq <= 1e-30:
        return 1e30
    f4 = f2 * f2
    l4sq = float(np.dot(_Wflat, f4))
    return l4sq / (l2sq * l2sq)


def _gradient(c):
    """Analytic gradient of R^4 = <f^4>/<f^2>^2 w.r.t. c.
    Let A = <f^4>, B = <f^2>.  Objective = A / B^2.
    dA/dc_i = 4 <f^3 * Y_i>
    dB/dc_i = 2 <f * Y_i>
    d(A/B^2)/dc_i = (4<f^3 Y_i> * B^2 - A * 2*B * 2<f Y_i>) / B^4
                  = 4<f^3 Y_i> / B^2 - 4 A <f Y_i> / B^3
                  = 4/B^2 * <f^3 Y_i> - 4A/B^3 * <f Y_i>
    """
    f = _Yflat @ c
    f2 = f * f
    f3 = f2 * f
    l2sq = float(np.dot(_Wflat, f2))
    if l2sq <= 1e-30:
        return np.zeros_like(c)
    l4sq = float(np.dot(_Wflat, f2 * f2))
    # <f^3 Y_i> for all i: _Yflat.T @ (W * f^3)
    wf3 = _Wflat * f3
    wf = _Wflat * f
    grad_f3 = _Yflat.T @ wf3   # shape (NC,)
    grad_f = _Yflat.T @ wf     # shape (NC,)
    B = l2sq
    A = l4sq
    grad = (4.0 / (B * B)) * grad_f3 - (4.0 * A / (B * B * B)) * grad_f
    return grad


def _optimize_from(c0, deadline):
    c = np.asarray(c0, dtype=np.float64).copy()
    nrm = np.linalg.norm(c)
    if nrm < 1e-14:
        return None
    c = c / nrm

    def fun(x):
        return _objective(x)

    def jac(x):
        return _gradient(x)

    best_c = c.copy()
    best_val = fun(c)

    # L-BFGS-B
    try:
        res = minimize(fun, c, jac=jac, method="L-BFGS-B",
                       options={"maxiter": 200, "maxfun": 1000,
                                "ftol": 1e-14, "gtol": 1e-8})
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x.copy()
    except Exception:
        pass

    if time.time() > deadline - 20.0:
        return best_c, best_val

    # Nelder-Mead polish
    try:
        res2 = minimize(fun, best_c, method="Nelder-Mead",
                        options={"maxiter": 800, "xatol": 1e-10,
                                 "fatol": 1e-12, "adaptive": True})
        if res2.fun < best_val:
            best_val = res2.fun
            best_c = res2.x.copy()
    except Exception:
        pass

    nrm = np.linalg.norm(best_c)
    if nrm < 1e-14:
        return None
    best_c = best_c / nrm
    return best_c, float(_objective(best_c))


def _make_seeds(rng, n_seeds):
    seeds = []
    # Seed 0: all zeros (will be rejected, but handle)
    # Use diverse structured + random seeds
    for i in range(n_seeds):
        c = np.zeros(_NC)
        if i == 0:
            # m=0 only
            c[_K] = 1.0
        elif i == 1:
            # m=10 cos
            c[_K + 10] = 1.0
        elif i == 2:
            # m=10 sin
            c[_K - 10] = 1.0
        elif i == 3:
            # all m with cos pattern
            for m in range(-_K, _K + 1):
                c[m + _K] = np.cos(0.5 * m)
        elif i == 4:
            # alternating
            for m in range(-_K, _K + 1):
                c[m + _K] = (-1) ** m
        elif i == 5:
            # Gaussian random
            c[:] = rng.standard_normal(_NC)
        elif i == 6:
            # Only even m
            for m in range(-_K, _K + 1, 2):
                c[m + _K] = rng.standard_normal()
        elif i == 7:
            # Only odd m
            for m in range(-_K + 1, _K + 1, 2):
                c[m + _K] = rng.standard_normal()
        elif i == 8:
            # Only m=0, ±10
            c[_K] = 1.0
            c[_K + 10] = rng.standard_normal()
            c[_K - 10] = rng.standard_normal()
        elif i == 9:
            # Only m=±5
            c[_K + 5] = rng.standard_normal()
            c[_K - 5] = rng.standard_normal()
        elif i == 10:
            # Symmetric: c_m = c_-m
            for m in range(0, _K + 1):
                v = rng.standard_normal()
                c[m + _K] = v
                c[-m + _K] = v
        elif i == 11:
            # Antisymmetric: c_m = -c_-m, c_0=0
            for m in range(1, _K + 1):
                v = rng.standard_normal()
                c[m + _K] = v
                c[-m + _K] = -v
        else:
            # General random with some structure
            c[:] = rng.standard_normal(_NC)
            # Add a bit of low-order structure
            c[_K] += 0.5
            if i % 2 == 0:
                c[_K + 10] += 0.3
                c[_K - 10] += 0.3
        nrm = np.linalg.norm(c)
        if nrm > 1e-14:
            c /= nrm
        seeds.append(c)
    return seeds


def rudin_problem10() -> np.ndarray:
    start = time.time()
    deadline = start + 590.0  # safety margin
    rng = np.random.default_rng(42)

    seeds = _make_seeds(rng, 24)
    best_c = None
    best_val = np.inf

    for i, c0 in enumerate(seeds):
        if time.time() > deadline - 30.0:
            break
        try:
            out = _optimize_from(c0, deadline)
        except Exception:
            continue
        if out is None:
            continue
        c, val = out
        if val < best_val:
            best_val = val
            best_c = c.copy()

    # Use remaining time for more random restarts
    n_extra = 0
    while time.time() < deadline - 40.0:
        c0 = rng.standard_normal(_NC)
        nrm = np.linalg.norm(c0)
        if nrm < 1e-14:
            continue
        c0 /= nrm
        try:
            out = _optimize_from(c0, deadline)
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
