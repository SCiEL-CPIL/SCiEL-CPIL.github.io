# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import gauss_legendre
from scipy.special import lpmv
from scipy.optimize import differential_evolution, minimize

_DEADLINE = time.time() + 600.0
_EPS = 1e-12

# ---- Precompute quadrature grid and the 21 real orthonormal degree-10 harmonics ----
_NTH = 70
_NPHI = 130
_K = 10

_x, _w = gauss_legendre(_NTH)                 # x = cos(theta) in [-1,1]
_w = _w * 0.5                                   # integrate over cos(theta) in [-1,1]
_phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)

# Normalization N_m = sqrt(21/(4 pi) * (10-m)!/(10+m)!)
def _norm(m):
    m = abs(int(m))
    return np.sqrt(21.0 / (4.0 * np.pi) * float(np.math.factorial(_K - m)) /
                                float(np.math.factorial(_K + m)))

def _build_harmonics():
    M = np.arange(-_K, _K + 1)
    N = _NTH * _NPHI
    H = np.empty((N, 2 * _K + 1), dtype=np.float64)
    for j, m in enumerate(M):
        am = abs(m)
        # Condon-Shortley phase is included in lpmv
        P = lpmv(am, _K, _x)                   # shape (NTH,)
        if m == 0:
            base = _norm(0) * P                # (NTH,)
            H[:, j] = np.repeat(base, _NPHI)
        elif m > 0:
            base = np.sqrt(2.0) * _norm(m) * P * np.cos(m * _phi)   # (NTH, NPHI)
            H[:, j] = base.ravel()
        else:
            base = np.sqrt(2.0) * _norm(am) * P * np.sin(am * _phi) # (NTH, NPHI)
            H[:, j] = base.ravel()
    return H

_H = _build_harmonics()

# Quadrature weights for the sphere (total mass 4*pi).  dOmega = dphi * dcos(theta).
_wphi = 2.0 * np.pi / _NPHI
_W = np.tile(_wphi * _w, _NPHI)                # (N,)
_W2 = _W * _W


def _ratio(c):
    f = _H @ c                                  # (N,)
    n2 = np.dot(f, f)
    if n2 <= _EPS:
        return 1e12
    n4 = np.dot(f * f, f * f)
    if n4 <= _EPS:
        return 1e12
    return np.sqrt(n4 / n2)


def _ratio_grad(c):
    f = _H @ c
    n2 = np.dot(f, f)
    if n2 <= _EPS:
        return 1e12, np.zeros_like(c)
    n4 = np.dot(f * f, f * f)
    if n4 <= _EPS:
        return 1e12, np.zeros_like(c)
    r = np.sqrt(n4 / n2)
    # d r / d c = (r / (2 n2)) * (2 f * (2 n2) - 4 f * n4)
    g = (r / (2.0 * n2)) * (4.0 * f * n2 - 4.0 * f * n4)
    g = _H.T @ g
    return r, g


def _valid(c):
    if c.shape != (21,):
        return False
    if not np.all(np.isfinite(c)):
        return False
    mx = np.max(np.abs(c))
    return (mx >= 1e-3) and (mx <= 1e5)


def _polish(c0):
    try:
        res = minimize(_ratio, c0, jac=_ratio_grad, method='L-BFGS-B',
                       options={'maxiter': 300, 'gtol': 1e-10, 'ftol': 1e-12})
        if res.success or res.fun < _ratio(c0):
            return res.x
    except Exception:
        pass
    return c0


def rudin_problem10() -> np.ndarray:
    rng = np.random.default_rng(12345)
    best = np.ones(2 * _K + 1, dtype=np.float64)
    best_r = _ratio(best)

    def _consider(c):
        nonlocal best, best_r
        c = np.asarray(c, dtype=np.float64)
        if not _valid(c):
            return
        r = _ratio(c)
        if r < best_r:
            best_r = r
            best = c.copy()

    # ---- Global differential evolution over diverse starts ----
    bounds = [(-1.0, 1.0)] * (2 * _K + 1)
    if time.time() < _DEADLINE - 45.0:
        try:
            de = differential_evolution(_ratio, bounds, seed=12345,
                                        maxiter=25, popsize=12, tol=1e-8,
                                        init='sobol', polish=True,
                                        updating='deferred', seed=12345)
            if de.x is not None and _valid(de.x):
                _consider(de.x)
        except Exception:
            pass

    # ---- Local refinement from many diverse starting points ----
    starts = []
    # symmetric / antisymmetric / single-mode / random patterns
    starts.append(np.ones(2 * _K + 1))
    starts.append(np.zeros(2 * _K + 1)); starts[-1][10] = 1.0
    starts.append(np.array([1.0 if (i - _K) % 2 == 0 else -1.0 for i in range(2 * _K + 1)]))
    for _ in range(40):
        s = rng.normal(0.0, 1.0, 2 * _K + 1)
        s = s / (np.linalg.norm(s) + _EPS)
        starts.append(s)
    for _ in range(20):
        s = np.zeros(2 * _K + 1)
        s[rng.integers(0, 2 * _K + 1)] = 1.0
        s[rng.integers(0, 2 * _K + 1)] = rng.choice([-1.0, 1.0])
        starts.append(s)

    for s in starts:
        if time.time() >= _DEADLINE - 30.0:
            break
        c = _polish(s)
        _consider(c)
        # a second polish from the polished point (cheap, often helps)
        if time.time() >= _DEADLINE - 30.0:
            break
        c2 = _polish(c)
        _consider(c2)

    # ---- Final safety validation ----
    if not _valid(best):
        best = np.ones(2 * _K + 1, dtype=np.float64)
    return best


# EVOLVE-BLOCK-END
