# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import lpmv, factorial

_DEADLINE = time.time() + 620.0
_K = 10
_NTH = 70
_NPHI = 130


def _build_harmonics():
    """Return Y (NPHI, NTH, 2K+1), weights (NTH,), and dphi."""
    k = _K
    mu, w = np.polynomial.legendre.leggauss(_NTH)
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI

    Y = np.zeros((_NPHI, _NTH, 2 * k + 1), dtype=np.float64)
    for idx, m in enumerate(range(-k, k + 1)):
        am = abs(m)
        N = np.sqrt((2 * k + 1) / (4.0 * np.pi)
                    * float(factorial(k - am, exact=True))
                    / float(factorial(k + am, exact=True)))
        P = lpmv(am, k, mu)
        if m == 0:
            Y[:, :, idx] = N * P
        elif m > 0:
            Y[:, :, idx] = np.sqrt(2.0) * N * P * np.cos(m * phi)[:, None]
        else:
            Y[:, :, idx] = np.sqrt(2.0) * N * P * np.sin(am * phi)[:, None]
    return Y, w, dphi


def _norm4_sq(Y, w, dphi, c):
    """Return ||f||_4^4 = ∫ f^4 dOmega for f = Y @ c."""
    f = Y @ c
    f2 = f * f
    f4 = f2 * f2
    return np.sum(f4 * w, axis=1) * dphi


def _ratio(Y, w, dphi, c):
    """Compute R = ||f||_4 / ||f||_2."""
    f = Y @ c
    f2 = f * f
    f4 = f2 * f2
    n2_sq = np.sum(f2 * w, axis=1) * dphi
    n4_4 = np.sum(f4 * w, axis=1) * dphi
    if n2_sq < 1e-15 or n4_4 < 1e-15:
        return np.inf
    return (n4_4 ** 0.25) / np.sqrt(n2_sq)


def _obj_sphere(c, Y, w, dphi):
    """Minimize ||f||_4^4 subject to ||c||_2 = 1 (projected)."""
    nrm = np.linalg.norm(c)
    if nrm < 1e-14:
        return 1e12
    c_n = c / nrm
    return _norm4_sq(Y, w, dphi, c_n)


def _grad_sphere(c, Y, w, dphi):
    """Gradient of ||f||_4^4 projected onto the unit sphere."""
    nrm = np.linalg.norm(c)
    if nrm < 1e-14:
        return np.zeros_like(c)
    c_n = c / nrm
    f = Y @ c_n
    f2 = f * f
    # d(||f||_4^4)/dc_n = 4 * sum W * f^3 * Y_i
    Wf3 = f * f2  # f^3
    # Gradient w.r.t. c_n: Y^T * (W * 4*f^3)
    # Then project: grad_projected = grad - (grad . c_n) * c_n
    grad = np.zeros(2 * _K + 1)
    for i in range(2 * _K + 1):
        grad[i] = 4.0 * np.sum(Wf3 * Y[:, :, i] * w[:, None], axis=1) * dphi
    # Project onto tangent space of unit sphere
    dot = np.dot(grad, c_n)
    grad = grad - dot * c_n
    # Scale back: since c_n = c/nrm, dc/dc_n = 1/nrm
    # Actually for L-BFGS-B we work in the full space, so just return grad
    return grad


def _projected_minimize(x0, Y, w, dphi, maxiter, ftol=1e-14, gtol=1e-12):
    """L-BFGS-B with sphere projection tracking."""
    best_c = x0 / np.linalg.norm(x0)
    best_val = _ratio(Y, w, dphi, best_c)
    state = {"c": best_c.copy(), "val": best_val}

    def fun(c):
        nrm = np.linalg.norm(c)
        if nrm < 1e-14:
            return 1e12
        c_n = c / nrm
        val4 = _norm4_sq(Y, w, dphi, c_n)
        # Also compute ratio to track
        r = _ratio(Y, w, dphi, c_n)
        if r < state["val"]:
            state["c"] = c_n.copy()
            state["val"] = r
        return val4

    def jac(c):
        nrm = np.linalg.norm(c)
        if nrm < 1e-14:
            return np.zeros_like(c)
        c_n = c / nrm
        f = Y @ c_n
        f2 = f * f
        Wf3 = f * f2
        grad = np.zeros(2 * _K + 1)
        for i in range(2 * _K + 1):
            grad[i] = 4.0 * np.sum(Wf3 * Y[:, :, i] * w[:, None], axis=1) * dphi
        # Project to tangent
        dot = np.dot(grad, c_n)
        grad = grad - dot * c_n
        return grad

    try:
        minimize(fun, x0, jac=jac, method="L-BFGS-B",
                 options={"maxiter": maxiter, "maxls": 50, "ftol": ftol, "gtol": gtol})
    except Exception:
        pass

    # Ensure best is tracked
    final_c = x0 / np.linalg.norm(x0)
    final_r = _ratio(Y, w, dphi, final_c)
    if final_r < state["val"]:
        state["c"] = final_c.copy()
        state["val"] = final_r

    return state["c"], state["val"]


def _make_inits(rng, k):
    """Diverse initial directions on the unit sphere."""
    inits = []
    n = 2 * k + 1
    # Random
    for _ in range(30):
        v = rng.standard_normal(n)
        inits.append(v / np.linalg.norm(v))
    # Single dominant m
    for m in range(-k, k + 1):
        v = np.zeros(n)
        v[m + k] = 1.0
        inits.append(v)
    # Paired cos/sin equal weight
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
    for _ in range(12):
        v = np.zeros(n)
        idx = rng.choice(n, size=rng.integers(2, 9), replace=False)
        v[idx] = rng.standard_normal(len(idx))
        if np.linalg.norm(v) > 1e-10:
            inits.append(v / np.linalg.norm(v))
    # Gaussian-like over m
    m_vals = np.arange(-k, k + 1, dtype=float)
    for sigma in [2.0, 3.0, 5.0]:
        v = np.exp(-m_vals**2 / (2 * sigma**2))
        inits.append(v / np.linalg.norm(v))
    # High-frequency emphasis
    v = np.abs(m_vals) / k
    inits.append(v / np.linalg.norm(v))
    # Low-frequency emphasis
    v = 1.0 - np.abs(m_vals) / k
    inits.append(v / np.linalg.norm(v))
    return inits


def rudin_problem10() -> np.ndarray:
    Y, w, dphi = _build_harmonics()
    k = _K
    n = 2 * k + 1
    rng = np.random.default_rng(42)
    inits = _make_inits(rng, k)

    best_c = inits[0].copy()
    best_val = _ratio(Y, w, dphi, best_c)

    # Phase 1: Short runs from all initializations, keep top 8
    scored = []
    for i, x0 in enumerate(inits):
        if time.time() > _DEADLINE - 150.0:
            break
        c, val = _projected_minimize(x0, Y, w, dphi, maxiter=80, ftol=1e-12, gtol=1e-10)
        scored.append((val, c))
        if val < best_val:
            best_val = val
            best_c = c.copy()

    scored.sort(key=lambda t: t[0])
    top = [c for _, c in scored[:8]]
    if not top:
        top = [best_c]

    # Phase 2: Refine top candidates with more iterations
    for x0 in top:
        if time.time() > _DEADLINE - 60.0:
            break
        c, val = _projected_minimize(x0, Y, w, dphi, maxiter=500, ftol=1e-14, gtol=1e-13)
        if val < best_val:
            best_val = val
            best_c = c.copy()

    # Phase 3: Random restarts from perturbed best
    while time.time() < _DEADLINE - 45.0:
        x0 = best_c + 0.25 * rng.standard_normal(n)
        x0 = x0 / np.linalg.norm(x0)
        c, val = _projected_minimize(x0, Y, w, dphi, maxiter=200, ftol=1e-14, gtol=1e-13)
        if val < best_val:
            best_val = val
            best_c = c.copy()

    # Phase 4: Final high-precision polish
    if time.time() < _DEADLINE - 20.0:
        c, val = _projected_minimize(best_c, Y, w, dphi, maxiter=1000, ftol=1e-16, gtol=1e-15)
        if val < best_val:
            best_val = val
            best_c = c.copy()

    out = best_c.astype(np.float64)
    # Validate
    if len(out) != 21:
        out = np.ones(21, dtype=np.float64)
    if np.any(np.isnan(out)) or np.any(np.isinf(out)):
        out = np.ones(21, dtype=np.float64)
    max_abs = np.max(np.abs(out))
    if max_abs < 1e-3:
        out = out / max_abs * 1.0 if max_abs > 0 else np.ones(21, dtype=np.float64)
    elif max_abs > 1e5:
        out = out / max_abs * 1.0
    return out

# EVOLVE-BLOCK-END
