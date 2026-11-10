# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize

_DEADLINE = time.time() + 615.0
_K = 10
_NTH = 70
_NPHI = 130


def _build_harmonics():
    """Return Y_flat (NTH*NPHI, 2k+1) and w_flat (NTH*NPHI,) for quadrature."""
    k = _K
    mu, w = np.polynomial.legendre.leggauss(_NTH)
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI

    Y = np.zeros((_NTH, _NPHI, 2 * k + 1), dtype=np.float64)
    for idx, m in enumerate(range(-k, k + 1)):
        am = abs(m)
        N = np.sqrt((2 * k + 1) / (4.0 * np.pi)
                    * float(factorial(k - am, exact=True))
                    / float(factorial(k + am, exact=True)))
        P = lpmv(am, k, mu)
        if m == 0:
            Y[:, :, idx] = N * P[:, None]
        elif m > 0:
            Y[:, :, idx] = np.sqrt(2.0) * N * P[:, None] * np.cos(m * phi)[None, :]
        else:
            Y[:, :, idx] = np.sqrt(2.0) * N * P[:, None] * np.sin(am * phi)[None, :]

    Y_flat = Y.reshape(_NTH * _NPHI, 2 * k + 1)
    w_flat = np.outer(w, np.full(_NPHI, dphi)).ravel()

    return Y_flat, w_flat


def _objective_and_grad(c, Y, w):
    """
    Compute R^4 = S4/S2^2 and its gradient. Minimizing R^4 is equivalent to minimizing R.
    """
    f = Y @ c
    f2 = f * f
    f3 = f2 * f
    f4 = f2 * f2

    S2 = np.dot(w, f2)
    S4 = np.dot(w, f4)

    if S2 < 1e-15:
        return 1e12, np.zeros_like(c)

    val = S4 / (S2 * S2)

    Yt_wf3 = Y.T @ (w * f3)
    Yt_wf = Y.T @ (w * f)

    grad = 4.0 * (S2 * Yt_wf3 - S4 * Yt_wf) / (S2 * S2 * S2)

    return val, grad


def _ratio_val(c, Y, w):
    """Compute R = ||f||_4 / ||f||_2."""
    f = Y @ c
    f2 = f * f
    S2 = np.dot(w, f2)
    S4 = np.dot(w, f2 * f2)
    if S2 < 1e-15:
        return np.inf
    return (S4 ** 0.25) / np.sqrt(S2)


def _make_inits(rng, k):
    """Diverse initial directions on the unit sphere."""
    inits = []
    n = 2 * k + 1
    # Random
    for _ in range(20):
        v = rng.standard_normal(n)
        inits.append(v / np.linalg.norm(v))
    # Single dominant m
    for m in range(-k, k + 1):
        v = np.zeros(n)
        v[m + k] = 1.0
        inits.append(v)
    # Paired cos/sin
    for m in range(1, k + 1):
        v = np.zeros(n)
        v[m + k] = 1.0
        v[-m + k] = 1.0
        inits.append(v / np.linalg.norm(v))
    # Alternating signs
    v = np.ones(n)
    v[::2] = -1.0
    inits.append(v / np.linalg.norm(v))
    # Random sparse
    for _ in range(8):
        v = np.zeros(n)
        idx = rng.choice(n, size=rng.integers(2, 9), replace=False)
        v[idx] = rng.standard_normal(len(idx))
        if np.linalg.norm(v) > 1e-12:
            inits.append(v / np.linalg.norm(v))
    # High-m dominance (m=10, 9)
    for m in [10, 9, 8]:
        v = np.zeros(n)
        v[m + k] = 1.0
        v[-m + k] = 0.5
        inits.append(v / np.linalg.norm(v))
    return inits


def _optimize_from(x0, Y, w, maxiter, tol):
    """Run L-BFGS-B with analytic gradient from x0."""
    def fun(c):
        val, grad = _objective_and_grad(c, Y, w)
        return val, grad

    try:
        res = minimize(fun, x0, method="L-BFGS-B", jac=True,
                       options={"maxiter": maxiter, "maxls": 40,
                                "ftol": tol, "gtol": tol * 0.1})
        c_opt = res.x
        val_opt = _ratio_val(c_opt, Y, w)
        return c_opt, val_opt
    except Exception:
        return x0, _ratio_val(x0, Y, w)


def rudin_problem10() -> np.ndarray:
    Y, w = _build_harmonics()
    k = _K
    n = 2 * k + 1
    rng = np.random.default_rng(42)
    inits = _make_inits(rng, k)

    best_c = inits[0].copy()
    best_val = np.inf

    # Phase 1: Short runs from diverse starts
    scored = []
    for i, x0 in enumerate(inits):
        if time.time() > _DEADLINE - 180.0:
            break
        c_opt, val_opt = _optimize_from(x0, Y, w, maxiter=150, tol=1e-14)
        if np.isfinite(val_opt):
            scored.append((val_opt, c_opt))
            if val_opt < best_val:
                best_val = val_opt
                best_c = c_opt.copy()

    scored.sort(key=lambda t: t[0])
    top = [c for _, c in scored[:10]]
    if not top:
        top = [best_c]

    # Phase 2: Refine top candidates
    for x0 in top:
        if time.time() > _DEADLINE - 90.0:
            break
        c_opt, val_opt = _optimize_from(x0, Y, w, maxiter=500, tol=1e-16)
        if np.isfinite(val_opt) and val_opt < best_val:
            best_val = val_opt
            best_c = c_opt.copy()

    # Phase 3: Random restarts from perturbed best
    while time.time() < _DEADLINE - 30.0:
        x0 = best_c + 0.2 * rng.standard_normal(n)
        x0 = x0 / np.linalg.norm(x0)
        c_opt, val_opt = _optimize_from(x0, Y, w, maxiter=200, tol=1e-15)
        if np.isfinite(val_opt) and val_opt < best_val:
            best_val = val_opt
            best_c = c_opt.copy()

    # Final normalization
    out = best_c.astype(np.float64)
    nrm = np.linalg.norm(out)
    if nrm < 1e-12:
        out = np.ones(n, dtype=np.float64)
    else:
        out = out / nrm

    # Validation
    if out.shape != (21,):
        out = np.ones(21, dtype=np.float64)
    if np.any(np.isnan(out)) or np.any(np.isinf(out)):
        out = np.ones(21, dtype=np.float64)

    max_abs = np.max(np.abs(out))
    if max_abs < 1e-3:
        out = out / max_abs * 1.0
    elif max_abs > 1e5:
        out = out / max_abs * 1.0

    return out


# EVOLVE-BLOCK-END
