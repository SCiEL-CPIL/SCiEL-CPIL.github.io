"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 200.0   # safety margin below the 210s wall-clock kill
_EPS = 1e-9

# Exact values of the (physicists') Hermite polynomials at 0:
# H0(0)=1, H4(0)=12, H8(0)=1344, H12(0)=42528
_H0_0, _H4_0, _H8_0, _H12_0 = 1.0, 12.0, 1344.0, 42528.0

# Precompute the leading coefficients of each H_{2k} (k = 0..6):
#   H_{2k}(x) = 2^{2k} * sum_{j=0..k} (-1)^j * (2k)! / (j! (2k-2j)!) * x^{2k-2j}
# Leading coefficient (x^{2k}) = 2^{2k} * (-1)^k * (2k)! / (k! k!)
def _lead_coeff(k):
    import math
    n = 2 * k
    c = (2.0 ** n) * ((-1.0) ** k) * math.factorial(n) / (math.factorial(k) * math.factorial(k))
    return c

_LEAD = np.array([_lead_coeff(k) for k in range(7)])  # index k -> leading coeff of H_{2k}


def _c3_of(c0, c1, c2):
    return -(_H0_0 * c0 + _H4_0 * c1 + _H8_0 * c2) / _H12_0


def _value_from(c0, c1, c2, _x=None):
    """Return r^2/(2*pi) for the given (c0,c1,c2), or inf if g is invalid."""
    c3 = _c3_of(c0, c1, c2)
    coefs = np.array([c0, c1, c2, c3]) * _LEAD[0::2]      # leading coeffs of H0,H4,H8,H12
    total_lead = coefs.sum()
    if abs(total_lead) < _EPS:
        return np.inf                                       # degree dropped -> unbounded
    a = coefs / total_lead                                  # monic: g ~ x^12
    if _x is None:
        _x = np.linspace(0.0, 15.0, 4000)
    q = a[0] * _x ** 12 + a[1] * _x ** 8 + a[2] * _x ** 4 + a[3]
    # g must be non-negative everywhere (else the evaluator flips it to -g,
    # which we already account for via the monic leading-coefficient normalization).
    if q.min() < -_EPS:
        return np.inf
    pos = q > _EPS
    if not pos.any():
        return 0.0
    # largest sign-changing root: last index where q goes from positive to negative
    idx = np.where(pos[:-1] & (~pos[1:]))[0]
    if idx.size == 0:
        return np.inf
    r = _x[idx[-1]]
    return r * r / (2.0 * np.pi)


def _objective(z):
    c0, c1, c2 = z
    return _value_from(c0, c1, c2)


def construct():
    import numpy as np
    from scipy.optimize import differential_evolution, minimize

    t0 = time.time()
    _X = np.linspace(0.0, 15.0, 4000)

    def obj(z):
        return _value_from(z[0], z[1], z[2], _X)

    best = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_v = obj(best)

    # ---- 1. Coarse 2D grid over the projective ratios (c0,c1) with c2 = 1 ----
    def _grid_search(n0, n1, lo, hi):
        nonlocal best, best_v
        for c0 in np.linspace(lo, hi, n0):
            for c1 in np.linspace(lo, hi, n1):
                if time.time() > _DEADLINE:
                    return
                v = obj([c0, c1, 1.0])
                if v < best_v:
                    best_v = v
                    best = np.array([c0, c1, 1.0])

    _grid_search(60, 60, -1.0, 1.0)
    _grid_search(120, 120, -0.5, 0.5)

    # ---- 2. Global refinement with differential evolution (multi-start) ----
    starts = [best.copy()]
    rng = np.random.default_rng(12345)
    for _ in range(3):
        starts.append(rng.uniform(-1.0, 1.0, size=3))
    for s in starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = differential_evolution(
                obj, bounds=[(-1.0, 1.0)] * 3, seed=7, tol=1e-8,
                maxiter=60, popsize=12, init='sobol', polish=False,
                x0=s, updating='deferred', workers=1,
            )
            if res.fun < best_v:
                best_v = float(res.fun)
                best = np.array(res.x)
        except Exception:
            pass

    # ---- 3. High-precision local polish (tighter tolerance) ----
    if time.time() < _DEADLINE - 10 and best_v < np.inf:
        try:
            res = minimize(obj, best, method='L-BFGS-B',
                           bounds=[(-1.0, 1.0)] * 3,
                           options={'ftol': 1e-14, 'gtol': 1e-12, 'maxiter': 2000})
            if res.fun < best_v:
                best_v = float(res.fun)
                best = np.array(res.x)
        except Exception:
            pass

    # ---- 4. Final exact evaluation at the returned point ----
    v_final = obj(best)
    if v_final < best_v:
        best_v = v_final
    if time.time() > _DEADLINE - 3:
        # return the best we have so far
        pass

    return np.array(best, dtype=float)
# EVOLVE-BLOCK-END
