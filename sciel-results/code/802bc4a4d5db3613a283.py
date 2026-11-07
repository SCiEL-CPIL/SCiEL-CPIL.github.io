"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import brentq, differential_evolution, minimize

_DEADLINE = time.time() + 200.0   # safety margin below the 210s wall-clock kill
_EPS = 1e-9

# Exact values of the (physicists') Hermite polynomials at 0:
# H0(0)=1, H4(0)=12, H8(0)=1344, H12(0)=42528
_H0_0, _H4_0, _H8_0, _H12_0 = 1.0, 12.0, 1344.0, 42528.0

# Precompute the leading coefficients of each H_{2k} (k = 0..6):
#   H_{2k}(x) = 2^{2k} * sum_{j=0..k} (-1)^j * (2k)! / (j! (2k-2j)!) * x^{2k-2j}
# Leading coefficient (x^{2k}) = 2^{2k} * (-1)^k * (2k)! / (k! k!)
def _lead_coeff(k):
    n = 2 * k
    c = (2.0 ** n) * ((-1.0) ** k) * math.factorial(n) / (math.factorial(k) * math.factorial(k))
    return c

_LEAD = np.array([_lead_coeff(k) for k in range(7)])  # index k -> leading coeff of H_{2k}


def _c3_of(c0, c1, c2):
    return -(_H0_0 * c0 + _H4_0 * c1 + _H8_0 * c2) / _H12_0


def _q_coefficients(c0, c1, c2):
    """Return monic coefficients [a3, a2, a1, a0] for q(t) = a3*t^3 + a2*t^2 + a1*t + a0 where t=x^2."""
    c3 = _c3_of(c0, c1, c2)
    coefs = np.array([c0, c1, c2, c3]) * _LEAD[0::2]
    total_lead = coefs.sum()
    if abs(total_lead) < _EPS:
        return None
    a = coefs / total_lead  # [a0, a1, a2, a3] for powers x^0, x^4, x^8, x^12
    # q(t) = a3*t^3 + a2*t^2 + a1*t + a0
    return a  # stored as [a0, a1, a2, a3]


def _eval_q(a, t):
    """Evaluate q(t) = a[3]*t^3 + a[2]*t^2 + a[1]*t + a[0]."""
    return ((a[3] * t + a[2]) * t + a[1]) * t + a[0]


def _value_from(c0, c1, c2, _t_grid=None):
    """Return r^2/(2*pi) for the given (c0,c1,c2), or inf if g is invalid."""
    a = _q_coefficients(c0, c1, c2)
    if a is None:
        return np.inf
    # Use np.roots to find all roots of the cubic q(t) = a[3]t^3 + a[2]t^2 + a[1]t + a[0]
    # Coefficients for np.roots: [a3, a2, a1, a0]
    try:
        roots = np.roots([a[3], a[2], a[1], a[0]])
    except Exception:
        return np.inf
    # Filter to real roots
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    # We need q(t) >= 0 for all t >= 0, and find the largest sign-changing root
    # A sign-changing root is where q goes from positive to negative (or vice versa)
    # Since we need q >= 0, valid roots are those where q changes from + to -
    # For a cubic with positive leading coefficient, q -> +inf as t -> +inf
    # So sign-changing roots come in pairs: + to - then - to +
    # The largest sign-changing root where q goes from + to - is what we want
    # Actually, we need the largest t where q(t) = 0 and q changes sign
    # Let's check each real root >= 0
    valid_roots = []
    for rt in real_roots:
        if rt < -_EPS:
            continue
        # Check if this is a sign-changing root by evaluating q slightly before and after
        delta = max(1e-8, abs(rt) * 1e-8)
        q_before = _eval_q(a, max(0.0, rt - delta))
        q_after = _eval_q(a, rt + delta)
        # Sign change: one positive, one negative (or zero at the root)
        if q_before * q_after < 0:
            valid_roots.append(rt)
    if not valid_roots:
        # Check if q is non-negative everywhere on [0, inf)
        # Sample some points
        t_test = np.linspace(0.0, 500.0, 1000)
        q_test = _eval_q(a, t_test)
        if q_test.min() < -_EPS:
            return np.inf
        # If q >= 0 everywhere, r = 0
        return 0.0
    # The largest sign-changing root
    r_t = max(valid_roots)
    # Verify q >= 0 on [0, r_t]
    if r_t > 0:
        t_check = np.linspace(0.0, r_t, 500)
        q_check = _eval_q(a, t_check)
        if q_check.min() < -_EPS:
            return np.inf
    r = math.sqrt(max(0.0, r_t))
    return r * r / (2.0 * np.pi)


def _objective(z):
    c0, c1, c2 = z
    return _value_from(c0, c1, c2)

def _smooth_objective(z, penalty_weight=100.0):
    """Smooth surrogate: add penalty for negative q values to make objective differentiable."""
    c0, c1, c2 = z
    a = _q_coefficients(c0, c1, c2)
    if a is None:
        return np.inf
    # Evaluate q on a grid
    t_grid = np.linspace(0.0, 225.0, 200)
    q_vals = _eval_q(a, t_grid)
    # Penalty for negative values: sum of max(0, -q)^2
    neg_mask = q_vals < 0
    if neg_mask.any():
        penalty = penalty_weight * np.sum(q_vals[neg_mask] ** 2)
    else:
        penalty = 0.0
    base_val = _value_from(c0, c1, c2)
    if base_val == np.inf:
        return penalty + 1000.0
    return base_val + penalty


def construct():
    import numpy as np
    from scipy.optimize import differential_evolution, minimize

    t0 = time.time()
    _T_GRID = np.linspace(0.0, 225.0, 2000)

    def obj(z):
        return _value_from(z[0], z[1], z[2], _T_GRID)

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
    for _ in range(5):
        starts.append(rng.uniform(-1.0, 1.0, size=3))
    for s in starts:
        if time.time() > _DEADLINE - 60:
            break
        try:
            res = differential_evolution(
                obj, bounds=[(-1.0, 1.0)] * 3, seed=7, tol=1e-8,
                maxiter=80, popsize=15, init='sobol', polish=False,
                x0=s, updating='deferred', workers=1,
            )
            if res.fun < best_v:
                best_v = float(res.fun)
                best = np.array(res.x)
        except Exception:
            pass

    # ---- 3. Local polish with Nelder-Mead (handles nonsmoothness better) ----
    if time.time() < _DEADLINE - 20 and best_v < np.inf:
        try:
            res = minimize(obj, best, method='Nelder-Mead',
                           options={'fatol': 1e-12, 'xtol': 1e-12, 'maxiter': 5000, 'maxfev': 5000})
            if res.fun < best_v:
                best_v = float(res.fun)
                best = np.array(res.x)
        except Exception:
            pass

    # ---- 4. Smooth surrogate optimization with L-BFGS-B ----
    if time.time() < _DEADLINE - 15 and best_v < np.inf:
        try:
            def smooth_obj(z):
                return _smooth_objective(z, penalty_weight=50.0)
            res = minimize(smooth_obj, best, method='L-BFGS-B',
                           bounds=[(-1.0, 1.0)] * 3,
                           options={'ftol': 1e-16, 'gtol': 1e-14, 'maxiter': 5000, 'maxls': 50})
            if res.fun < best_v:
                best_v = float(res.fun)
                best = np.array(res.x)
        except Exception:
            pass

    # ---- 4b. Final high-precision polish with Nelder-Mead (tighter) ----
    if time.time() < _DEADLINE - 10 and best_v < np.inf:
        try:
            res = minimize(obj, best, method='Nelder-Mead',
                           options={'fatol': 1e-15, 'xtol': 1e-15, 'maxiter': 10000, 'maxfev': 10000,
                                    'adaptive': True})
            if res.fun < best_v:
                best_v = float(res.fun)
                best = np.array(res.x)
        except Exception:
            pass

    # ---- 4c. Multi-start local polish from nearby points ----
    if time.time() < _DEADLINE - 5 and best_v < np.inf:
        rng2 = np.random.default_rng(999)
        for _ in range(5):
            if time.time() > _DEADLINE - 5:
                break
            perturbed = best + rng2.normal(0, 0.01, size=3)
            try:
                res = minimize(obj, perturbed, method='Nelder-Mead',
                               options={'fatol': 1e-14, 'xtol': 1e-14, 'maxiter': 3000, 'maxfev': 3000})
                if res.fun < best_v:
                    best_v = float(res.fun)
                    best = np.array(res.x)
            except Exception:
                pass

    # ---- 5. Final exact evaluation at the returned point ----
    v_final = obj(best)
    if v_final < best_v:
        best_v = v_final

    # ---- 6. One more round of fine-tuning around the best point ----
    if time.time() < _DEADLINE - 3 and best_v < np.inf:
        rng3 = np.random.default_rng(42)
        for scale in [0.005, 0.001, 0.0001]:
            if time.time() > _DEADLINE - 3:
                break
            for _ in range(10):
                if time.time() > _DEADLINE - 3:
                    break
                candidate = best + rng3.normal(0, scale, size=3)
                try:
                    res = minimize(obj, candidate, method='Nelder-Mead',
                                   options={'fatol': 1e-15, 'xtol': 1e-15, 'maxiter': 2000, 'maxfev': 2000})
                    if res.fun < best_v:
                        best_v = float(res.fun)
                        best = np.array(res.x)
                except Exception:
                    pass

    return np.array(best, dtype=float)
# EVOLVE-BLOCK-END
