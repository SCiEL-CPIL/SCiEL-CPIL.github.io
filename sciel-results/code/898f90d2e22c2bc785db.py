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
    """Return Y (nphi, nth, 2k+1), weights (nth,), and dphi."""
    k = _K
    mu, w = np.polynomial.legendre.leggauss(_NTH)
    theta = np.arccos(mu)
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


def _norms(Y, w, dphi, c):
    """Return (||f||_2, ||f||_4) for f = Y @ c on the grid."""
    f = Y @ c
    f2 = f * f
    f4 = f2 * f2
    n2 = np.sum(f2 * w, axis=1) * dphi
    n4 = np.sum(f4 * w, axis=1) * dphi
    return np.sqrt(np.sum(n2)), np.sum(n4)


def _ratio(Y, w, dphi, c):
    n2, n4 = _norms(Y, w, dphi, c)
    if n2 < 1e-12:
        return np.inf
    return (n4 ** 0.25) / np.sqrt(n2)


def _projected_minimize(x0, Y, w, dphi, maxiter):
    """L-BFGS-B with projection onto the unit sphere at each step."""
    x0 = x0 / np.linalg.norm(x0)
    best_c = x0.copy()
    best_val = _ratio(Y, w, dphi, best_c)
    state = {"c": best_c, "val": best_val}

    def fun(c):
        nrm = np.linalg.norm(c)
        if nrm < 1e-12:
            return 1e12
        c = c / nrm
        _, n4 = _norms(Y, w, dphi, c)
        if n4 < state["val"]:
            state["c"] = c
            state["val"] = n4
        return n4

    try:
        minimize(fun, x0, method="L-BFGS-B",
                 options={"maxiter": maxiter, "maxls": 30, "ftol": 1e-15, "gtol": 1e-12})
    except Exception:
        pass
    return state["c"], state["val"]


def _make_inits(rng, k):
    """Diverse initial directions with emphasis on highest-order harmonics."""
    inits = []
    # Highest order m = ±k (most uniform magnitude)
    for m in [k, -k]:
        v = np.zeros(2 * k + 1)
        v[m + k] = 1.0
        inits.append(v)
    # Pair of m = ±k
    v = np.zeros(2 * k + 1)
    v[k + k] = 1.0
    v[-k + k] = 1.0
    inits.append(v / np.linalg.norm(v))
    
    # Next highest m = ±(k-1)
    for m in [k-1, -(k-1)]:
        v = np.zeros(2 * k + 1)
        v[m + k] = 1.0
        inits.append(v)
    v = np.zeros(2 * k + 1)
    v[k-1 + k] = 1.0
    v[-(k-1) + k] = 1.0
    inits.append(v / np.linalg.norm(v))
    
    # m = ±(k-2)
    for m in [k-2, -(k-2)]:
        v = np.zeros(2 * k + 1)
        v[m + k] = 1.0
        inits.append(v)
    v = np.zeros(2 * k + 1)
    v[k-2 + k] = 1.0
    v[-(k-2) + k] = 1.0
    inits.append(v / np.linalg.norm(v))
    
    # m=0 mixed with high m
    for m in [k, k-1, k-2]:
        for weight in [0.1, 0.5, 0.9]:
            v = np.zeros(2 * k + 1)
            v[0 + k] = weight
            v[m + k] = np.sqrt(1 - weight**2)
            v[-m + k] = np.sqrt(1 - weight**2)
            inits.append(v / np.linalg.norm(v))
    
    # Random
    for _ in range(20):
        v = rng.standard_normal(2 * k + 1)
        inits.append(v / np.linalg.norm(v))
    
    # Single dominant m (other values)
    for m in range(-k + 3, k - 2):
        v = np.zeros(2 * k + 1)
        v[m + k] = 1.0
        inits.append(v)
    
    # Paired cos/sin (m>0)
    for m in range(1, k + 1):
        v = np.zeros(2 * k + 1)
        v[m + k] = 1.0
        v[-m + k] = 1.0
        inits.append(v / np.linalg.norm(v))
    
    # Alternating signs
    v = np.ones(2 * k + 1)
    v[::2] = -1.0
    inits.append(v / np.linalg.norm(v))
    
    # Random sparse
    for _ in range(8):
        v = np.zeros(2 * k + 1)
        idx = rng.choice(2 * k + 1, size=rng.integers(2, 8), replace=False)
        v[idx] = rng.standard_normal(len(idx))
        if np.linalg.norm(v) > 0:
            inits.append(v / np.linalg.norm(v))
    return inits


def _polish(x0, Y, w, dphi, maxiter=2000):
    """High-precision polish pass with very tight tolerances."""
    x0 = x0 / np.linalg.norm(x0)
    best_c = x0.copy()
    best_val = _ratio(Y, w, dphi, best_c)
    state = {"c": best_c, "val": best_val}

    def fun(c):
        nrm = np.linalg.norm(c)
        if nrm < 1e-12:
            return 1e12
        c = c / nrm
        _, n4 = _norms(Y, w, dphi, c)
        if n4 < state["val"]:
            state["c"] = c
            state["val"] = n4
        return n4

    try:
        minimize(fun, x0, method="L-BFGS-B",
                 options={"maxiter": maxiter, "maxls": 50, "ftol": 1e-18, "gtol": 1e-14})
    except Exception:
        pass
    return state["c"], state["val"]


def rudin_problem10() -> np.ndarray:
    Y, w, dphi = _build_harmonics()
    k = _K
    rng = np.random.default_rng(12345)
    inits = _make_inits(rng, k)

    best_c = inits[0].copy()
    best_val = np.inf

    # Phase 1: short runs from many diverse starts, keep top 8
    scored = []
    for i, x0 in enumerate(inits):
        if time.time() > _DEADLINE - 150.0:
            break
        c, val = _projected_minimize(x0, Y, w, dphi, maxiter=150)
        scored.append((val, c))
    scored.sort(key=lambda t: t[0])
    top = [c for _, c in scored[:8]]
    if not top:
        top = [best_c]

    # Phase 2: refine top candidates with more iterations
    for x0 in top:
        if time.time() > _DEADLINE - 90.0:
            break
        c, val = _projected_minimize(x0, Y, w, dphi, maxiter=800)
        if val < best_val:
            best_val = val
            best_c = c

    # Phase 3: random restarts from perturbed best until deadline
    while time.time() < _DEADLINE - 60.0:
        x0 = best_c + 0.5 * rng.standard_normal(2 * k + 1)
        x0 = x0 / np.linalg.norm(x0)
        c, val = _projected_minimize(x0, Y, w, dphi, maxiter=300)
        if val < best_val:
            best_val = val
            best_c = c

    # Phase 4: High-precision polish on the best candidate
    if time.time() < _DEADLINE - 15.0:
        c, val = _polish(best_c, Y, w, dphi, maxiter=2000)
        if val < best_val:
            best_val = val
            best_c = c

    out = best_c.astype(np.float64)
    out = out / np.linalg.norm(out)
    return out


# EVOLVE-BLOCK-END
