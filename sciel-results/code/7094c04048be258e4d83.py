# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_SEED = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])

# analytic values at 0 for the P(0)=0 constraint
_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 1680.0
_H12_0 = 665280.0

def _hermite(n, x):
    x = np.asarray(x, dtype=float)
    H0 = np.ones_like(x)
    if n == 0:
        return H0
    H1 = 2.0 * x
    if n == 1:
        return H1
    Hp2 = H0  # H_{k-2}
    Hp1 = H1  # H_{k-1}
    for k in range(1, n):
        Hk1 = 2.0 * x * Hp1 - 2.0 * k * Hp2
        Hp2, Hp1 = Hp1, Hk1
    return Hp1

def _basis(L, N):
    x = np.linspace(-L, L, N)
    dx = x[1] - x[0]
    g = np.exp(-0.5 * x * x)
    B0 = _hermite(0, x) * g
    B4 = _hermite(4, x) * g
    B8 = _hermite(8, x) * g
    B12 = _hermite(12, x) * g
    return B0, B4, B8, B12, dx

_B0s, _B4s, _B8s, _B12s, _DXs = _basis(14.0, 30000)
_B0f, _B4f, _B8f, _B12f, _DXf = _basis(16.0, 100000)

def _value(c, fine=False):
    a0, a4, a8 = c
    a12 = -(a0 * _H0_0 + a4 * _H4_0 + a8 * _H8_0) / _H12_0
    if fine:
        g = a0 * _B0f + a4 * _B4f + a8 * _B8f + a12 * _B12f
        dx = _DXf
    else:
        g = a0 * _B0s + a4 * _B4s + a8 * _B8s + a12 * _B12s
        dx = _DXs
    n2 = float(dx * np.dot(g, g))
    if n2 <= 0.0 or not np.isfinite(n2):
        return 1e30
    g = g / np.sqrt(n2)
    L1 = float(dx * np.sum(np.abs(g)))
    return L1 * L1

def construct():
    t0 = time.time()
    deadline = t0 + 150.0
    best = _SEED.copy()
    best_val = _value(best)

    rng = np.random.default_rng(12345)
    starts = [_SEED.copy()]
    # targeted starts around seed
    for scale in (0.5, 0.1, 0.01):
        starts.append(_SEED + scale * rng.normal(0, 1.0, 3))
    for _ in range(20):
        starts.append(rng.normal(0, 1.5, 3))

    for s in starts:
        if time.time() > deadline - 40.0:
            break
        try:
            res = minimize(_value, s, method='L-BFGS-B',
                           options={'maxiter': 1500, 'maxfun': 3000, 'ftol': 1e-14})
            if res.fun < best_val:
                best_val = float(res.fun)
                best = np.asarray(res.x, dtype=float).copy()
        except Exception:
            pass
        # derivative-free polish from current best occasionally
        if time.time() > deadline - 30.0:
            break
        try:
            res2 = minimize(_value, best, method='Nelder-Mead',
                            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-15})
            if res2.fun < best_val:
                best_val = float(res2.fun)
                best = np.asarray(res2.x, dtype=float).copy()
        except Exception:
            pass

    # final high-precision polish on the fine grid
    if time.time() < deadline - 10.0:
        try:
            res = minimize(lambda c: _value(c, fine=True), best, method='L-BFGS-B',
                           options={'maxiter': 5000, 'maxfun': 50000, 'ftol': 1e-16, 'gtol': 1e-14})
            if np.isfinite(res.fun) and res.fun < best_val:
                best = np.asarray(res.x, dtype=float).copy()
        except Exception:
            pass

    if not np.all(np.isfinite(best)):
        return _SEED.copy()
    return best
# EVOLVE-BLOCK-END
