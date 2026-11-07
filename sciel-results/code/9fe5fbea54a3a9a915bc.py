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
    # a[3] is the leading coefficient (coefficient of t^3)
    # We normalize so a[3] = 1
    if abs(a[3]) < _EPS:
        return None
    a = a / a[3]
    return a  # stored as [a0, a1, a2, 1.0]


def _eval_q(a, t):
    """Evaluate q(t) = a[3]*t^3 + a[2]*t^2 + a[1]*t + a[0]."""
    return ((a[3] * t + a[2]) * t + a[1]) * t + a[0]


def _value_from(c0, c1, c2, _t_grid=None):
    """Return r^2/(2*pi) for the given (c0,c1,c2), or inf if g is invalid."""
    a = _q_coefficients(c0, c1, c2)
    if a is None:
        return np.inf
    
    # Use np.roots for accuracy and speed
    # q(t) = t^3 + a[2]t^2 + a[1]t + a[0]
    roots = np.roots([1.0, a[2], a[1], a[0]])
    
    # Filter real roots
    real_roots = roots[np.isreal(roots)].real
    # Filter positive roots (with a small tolerance to avoid 0)
    positive_roots = real_roots[real_roots > 1e-7]
    
    if len(positive_roots) == 0:
        # No positive real roots implies q(t) > 0 for all t > 0 (since leading coeff is 1)
        # Thus, no sign-changing positive roots.
        return 0.0
        
    # Sort positive roots
    positive_roots.sort()
    
    # Check the largest positive root for sign change
    r_t = positive_roots[-1]
    
    # Verify sign change
    # Use a small epsilon relative to the root magnitude
    eps = max(1e-8, 1e-8 * abs(r_t))
    q_left = _eval_q(a, r_t - eps)
    q_right = _eval_q(a, r_t + eps)
    
    if q_left * q_right < 0:
        r = math.sqrt(max(0.0, r_t))
        return r * r / (2.0 * np.pi)
    else:
        # If the largest root is not sign-changing (e.g., double root),
        # check if any smaller positive root is sign-changing.
        # However, if q(t) >= 0 for all t, there are no sign-changing roots.
        # If there are sign-changing roots, the largest one is the one where q goes from - to +
        # (since q -> +inf).
        # If the largest one is a double root (touches 0), it doesn't change sign.
        # But if it's a double root, q(t) >= 0 locally.
        # If q(t) dips below 0, it must have two simple roots or one simple and one double?
        # If q(t) has a double root at t_d > 0 and a simple root at t_s < 0.
        # q(t) = (t-t_d)^2 (t-t_s).
        # For t > t_d, q > 0.
        # For 0 < t < t_d, (t-t_d)^2 > 0, (t-t_s) > 0 (if t_s < 0). So q > 0.
        # So q(t) > 0 for all t > 0. No sign change.
        # So if the largest positive root is not sign-changing, likely no positive sign-changing roots exist.
        return 0.0


def _objective(z):
    c0, c1, c2 = z
    return _value_from(c0, c1, c2)


def construct():
    import numpy as np
    from scipy.optimize import differential_evolution, minimize

    t0 = time.time()
    
    def obj(z):
        return _value_from(z[0], z[1], z[2])

    best = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_v = obj(best)

    # If the seed is already 0.0, we are done.
    if best_v <= 1e-12:
        return np.array(best, dtype=float)

    # Global search to find a point where value is 0.0
    # This corresponds to finding coefficients where q(t) >= 0 for all t >= 0.
    
    # Use differential evolution
    # Bounds: The coefficients can be scaled, but the ratios matter.
    # Let's bound them in [-10, 10] to cover reasonable ratios.
    
    try:
        res = differential_evolution(
            obj, bounds=[(-10.0, 10.0)] * 3, seed=42, tol=1e-10,
            maxiter=200, popsize=20, init='sobol', polish=False,
            updating='deferred', workers=1,
        )
        if res.fun < best_v:
            best_v = float(res.fun)
            best = np.array(res.x)
    except Exception:
        pass

    if best_v <= 1e-12:
        return np.array(best, dtype=float)

    # If DE didn't find 0, try local refinement from the best point found
    if time.time() < _DEADLINE - 20 and best_v < np.inf:
        try:
            res = minimize(obj, best, method='Nelder-Mead',
                           options={'fatol': 1e-14, 'xtol': 1e-14, 'maxiter': 5000, 'maxfev': 5000})
            if res.fun < best_v:
                best_v = float(res.fun)
                best = np.array(res.x)
        except Exception:
            pass

    # Final check
    v_final = obj(best)
    if v_final < best_v:
        best_v = v_final

    return np.array(best, dtype=float)
# EVOLVE-BLOCK-END
