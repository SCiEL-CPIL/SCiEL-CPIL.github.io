# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
from scipy.optimize import minimize

_K = 10
_NPHI = 130
_NTH = 70


def _build_quadrature():
    """Build Gauss-Legendre nodes/weights in x=cos(theta) and phi grid,
    plus the real orthonormal degree-K harmonics evaluated on the grid."""
    x, w = np.polynomial.legendre.leggauss(_NTH)          # x in (-1,1), weights sum to 2
    # surface element dOmega = dphi dx (since dOmega = sin(theta) dtheta dphi = -dx dphi)
    # total mass = 4*pi, and sum_i w_i = 2, sum_j (dphi) = 2*pi  -> product = 4*pi. Good.
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI

    # 2D weights: W[i,j] = w_i * dphi
    W = np.outer(w, np.full(_NPHI, dphi))

    # Evaluate real orthonormal harmonics Y_m for m = -K..K on the grid.
    # Y_0 = N_0 P^K(cos theta)
    # Y_m = sqrt(2) N_m P^|m|(cos theta) cos(m phi)  for m>0
    # Y_m = sqrt(2) N_m P^|m|(cos theta) sin(|m| phi) for m<0
    # N_m = sqrt( (2K+1)/(4 pi) * (K-m)!/(K+m)! )
    Y = np.zeros((_NTH, _NPHI, 2 * _K + 1))
    for idx, m in enumerate(range(-_K, _K + 1)):
        am = abs(m)
        fact_ratio = 1.0
        for n in range(am):
            fact_ratio *= (1.0 / (n + 1.0))
        # (K-am)!/(K+am)! = product over n=am..K of 1/(n+1)  ... careful
        # (K-am)!/(K+am)! = 1 / [ (K-am+1)(K-am+2)...(K+am) ]
        denom = 1.0
        for n in range(_K - am + 1, _K + am + 1):
            denom *= n
        fact_ratio = 1.0 / denom
        Nm = np.sqrt((2 * _K + 1) / (4.0 * np.pi) * fact_ratio)
        P = lpmv(am, _K, x)  # includes Condon-Shortley phase
        if m == 0:
            Y[:, :, idx] = Nm * P[:, None]
        elif m > 0:
            Y[:, :, idx] = np.sqrt(2.0) * Nm * P[:, None] * np.cos(m * phi)[None, :]
        else:
            Y[:, :, idx] = np.sqrt(2.0) * Nm * P[:, None] * np.sin(am * phi)[None, :]
    return Y, W


_Y, _W = _build_quadrature()
_W2 = _W * _W  # not needed; keep W
_Wflat = _W.ravel()
_Yflat = _Y.reshape(-1, 2 * _K + 1)  # (N, 21)


def _ratio_from_coeffs(c):
    """Compute R = ||f||4/||f||2 for coefficients c (length 21)."""
    f = _Yflat @ c                       # (N,)
    f2 = f * f
    f4 = f2 * f2
    l2sq = float(np.dot(_Wflat, f2))
    if l2sq <= 0.0:
        return np.inf
    l4sq = float(np.dot(_Wflat, f4))
    if l4sq <= 0.0:
        return 1.0
    l2 = np.sqrt(l2sq)
    l4 = np.sqrt(l4sq)
    return l4 / l2


def _objective(c):
    c = np.asarray(c, dtype=np.float64)
    nrm = np.linalg.norm(c)
    if nrm < 1e-14:
        return 1e30
    c = c / nrm
    return _ratio_from_coeffs(c)


def _gradient(c):
    """Analytic gradient of R w.r.t. c (with c normalized to unit norm handled by
    projecting; simpler: compute gradient of R(c) for the normalized c via chain rule).
    We optimize over the sphere; use gradient of R(c/||c||). For BFGS we can just
    provide gradient of R(c) and rely on normalization being near-identity at optimum
    since R is scale-invariant (R(tc)=R(c)), so gradient is orthogonal to c at any
    point; we can use the raw gradient of R(c) and it will be tangent automatically.
    """
    c = np.asarray(c, dtype=np.float64)
    nrm = np.linalg.norm(c)
    if nrm < 1e-14:
        return np.zeros_like(c)
    cn = c / nrm
    f = _Yflat @ cn                      # (N,)
    f2 = f * f
    l2sq = float(np.dot(_Wflat, f2))
    l4sq = float(np.dot(_Wflat, f2 * f2))
    l2 = np.sqrt(l2sq)
    l4 = np.sqrt(l4sq)
    # d l2 / d c = (1/l2) * 2 * W . f * Y  (since l2 = sqrt(sum W f^2), d/dc = (1/(2 l2))*2 sum W f Y)
    # d l4 / d c = (1/l4) * 2 * sum W f^3 Y
    Yt_f = _Yflat.T @ (f * _Wflat)       # (21,)
    Yt_f3 = _Yflat.T @ (f * f2 * _Wflat) # (21,)
    dl2_dc = (2.0 / l2) * Yt_f
    dl4_dc = (2.0 / l4) * Yt_f3
    dR_dc = (dl4_dc * l2 - dl2_dc * l4) / (l2 * l2)
    # Project to tangent of sphere (remove radial component) to respect normalization
    dR_dc = dR_dc - (np.dot(dR_dc, cn) / nrm) * c
    return dR_dc


def _optimize_from(c0, deadline):
    c0 = np.asarray(c0, dtype=np.float64)
    nrm = np.linalg.norm(c0)
    if nrm < 1e-14:
        c0 = np.random.randn(2 * _K + 1)
        nrm = np.linalg.norm(c0)
    c0 = c0 / nrm
    res = minimize(_objective, c0, jac=_gradient, method="L-BFGS-B",
                   options={"maxiter": 400, "maxfun": 800, "gtol": 1e-10,
                            "ftol": 1e-14})
    if res.success is False and not np.isfinite(res.fun):
        return None
    c = np.asarray(res.x, dtype=np.float64)
    nrm = np.linalg.norm(c)
    if nrm < 1e-14:
        return None
    c = c / nrm
    return c, float(_ratio_from_coeffs(c))


def rudin_problem10() -> np.ndarray:
    start = time.time()
    deadline = start + 600.0  # safety margin before 630s kill
    rng = np.random.default_rng(2024)

    best_c = None
    best_val = np.inf

    # Deterministic-ish set of restarts: mix of structured and random.
    def _candidate(i):
        c = np.zeros(2 * _K + 1)
        # spread energy across m to encourage flatness
        for m in range(-_K, _K + 1):
            idx = m + _K
            c[idx] = np.cos(i * 0.7 + m * 0.3) + 0.3 * rng.standard_normal()
        return c

    n_restarts = 40
    for i in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        c0 = _candidate(i)
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
        if time.time() > deadline - 30.0:
            break

    if best_c is None:
        # Fallback: all-ones (valid, matches seed behavior)
        best_c = np.ones(2 * _K + 1, dtype=np.float64)

    # Ensure validity: finite, in [1e-3, 1e5] range for max|c|
    best_c = np.asarray(best_c, dtype=np.float64)
    if not np.all(np.isfinite(best_c)):
        best_c = np.ones(2 * _K + 1, dtype=np.float64)
    mx = float(np.max(np.abs(best_c)))
    if mx < 1e-3:
        best_c = best_c * (1.0 / mx)
    if mx > 1e5:
        best_c = best_c * (1e5 / mx)
    return best_c


# EVOLVE-BLOCK-END
