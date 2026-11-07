import time
import numpy as np
from fractions import Fraction
from scipy.optimize import minimize, differential_evolution

# ---------- exact Hermite coefficients (physicists' H_n) ----------
def _hermite_coeffs(n):
    if n == 0:
        return [Fraction(1)]
    if n == 1:
        return [Fraction(0), Fraction(2)]
    c1 = _hermite_coeffs(n - 1)
    c2 = _hermite_coeffs(n - 2)
    a = [Fraction(0)] + [Fraction(2) * v for v in c1]
    b = [Fraction(-2 * (n - 1)) * v for v in c2]
    m = max(len(a), len(b))
    res = [Fraction(0)] * m
    for i, v in enumerate(a):
        res[i] += v
    for i, v in enumerate(b):
        res[i] += v
    while len(res) > 1 and res[-1] == 0:
        res.pop()
    return res

_HC = {n: _hermite_coeffs(n) for n in (0, 4, 8, 12)}
_HF = {n: np.array([float(v) for v in c], dtype=np.float64) for n, c in _HC.items()}
_H0_0 = float(_HC[0][0])
_H4_0 = float(_HC[4][0])
_H8_0 = float(_HC[8][0])
_H12_0 = float(_HC[12][0])

# ---------- coefficient extraction ----------
def _build_q_coeffs(c1, c2):
    """q(x)=g(x)/x^2 with c0=1. Returns ascending coeffs of q in powers x^{2k}."""
    c3 = -(_H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    g = 1.0 * _HF[0] + c1 * _HF[4] + c2 * _HF[8] + c3 * _HF[12]
    # g is even, so g[2], g[4], ... are the coefficients of x^2, x^4, etc.
    # q(x) = g(x)/x^2. If g(x) = sum a_i x^i, q(x) = sum a_{i+2} x^{i-2}.
    # Since g is even, only even powers.
    # g[0] is coeff of x^0 (must be 0).
    # g[2] is coeff of x^2. q[0] = g[2].
    # g[4] is coeff of x^4. q[1] = g[4].
    # ...
    # g[12] is coeff of x^12. q[5] = g[12].
    q = g[2:13:2]
    while len(q) > 1 and q[-1] == 0.0:
        q = q[:-1]
    return np.asarray(q, dtype=np.float64)

def _value_from(c1, c2):
    try:
        q = _build_q_coeffs(c1, c2)
        if len(q) < 2:
            return np.inf
        # Find roots of q(x) = 0. q is a polynomial in x^2? No, q is in x.
        # Wait, _build_q_coeffs returns coeffs for x^0, x^1, ...?
        # Let's check indices.
        # g[2] -> x^2 term in g -> x^0 term in q.
        # g[4] -> x^4 term in g -> x^2 term in q.
        # So q(y) where y=x^2 is a polynomial in y.
        # The array q contains coeffs for y^0, y^1, ..., y^5.
        # So we need roots of Q(y) = 0.
        try:
            roots_y = np.roots(q[::-1])
        except Exception:
            return np.inf
        
        real_pos = []
        for r in roots_y:
            if abs(r.imag) <= 1e-10 * max(1.0, abs(r.real)) and r.real > 1e-12:
                real_pos.append(float(r.real))
        
        if not real_pos:
            return 1e10
            
        def Qval(y):
            return np.polyval(q[::-1], y)
        
        best_y = None
        # We want the LARGEST real root at which q changes sign.
        # q(y) = g(sqrt(y))/y.
        # Sign change of q at y means g(x) changes sign at x=sqrt(y).
        for y in sorted(real_pos, reverse=True):
            h = 1e-7 * max(1.0, abs(y))
            if Qval(y - h) * Qval(y + h) < 0:
                best_y = y
                break
        
        if best_y is None:
            # If no sign change found with tight tolerance, try looser
            for y in sorted(real_pos, reverse=True):
                for h in [1e-6, 1e-5, 1e-4, 1e-3]:
                    h = h * max(1.0, abs(y))
                    if Qval(y - h) * Qval(y + h) < 0:
                        best_y = y
                        break
                if best_y is not None:
                    break
                    
        if best_y is None:
            return 1e10
            
        return best_y / (2.0 * np.pi)
    except Exception:
        return np.inf

def construct():
    start = time.time()
    DEADLINE = 200.0 # Safety margin for 210s limit

    seed_c0 = -113.0 / 100.0
    seed_c1 = 1.0 / 25.0
    seed_c2 = 1.0 / 3240.0
    seed_c1r = seed_c1 / seed_c0
    seed_c2r = seed_c2 / seed_c0

    best_c1 = seed_c1r
    best_c2 = seed_c2r
    best_val = _value_from(best_c1, best_c2)

    def time_left():
        return DEADLINE - (time.time() - start)

    # Phase 1: Global Search using Differential Evolution
    # Bounds for c1/c0 and c2/c0
    # The seed is around c1r ~ -0.035, c2r ~ -0.00027
    # Let's try a reasonably wide box
    bounds = [(-1.0, 1.0), (-0.1, 0.1)]
    
    if time_left() > 60:
        try:
            def obj_de(params):
                if time_left() < 10:
                    return best_val
                return _value_from(params[0], params[1])
            
            res_de = differential_evolution(obj_de, bounds, maxiter=100, popsize=10, 
                                             tol=1e-6, seed=42, polish=False)
            if res_de.success or (np.isfinite(res_de.fun) and res_de.fun < best_val):
                best_c1, best_c2 = res_de.x
                best_val = _value_from(best_c1, best_c2)
        except Exception:
            pass

    # Phase 2: Local Optimization from multiple starts
    def obj_exact(params):
        if time_left() < 5:
            return best_val
        return _value_from(params[0], params[1])

    rng = np.random.RandomState(42)
    starts = [[best_c1, best_c2], [seed_c1r, seed_c2r]]
    
    # Add some random starts
    for _ in range(5):
        starts.append([best_c1 + rng.randn() * 0.1, best_c2 + rng.randn() * 0.01])
        
    for s in starts:
        if time_left() < 15:
            break
        try:
            res = minimize(obj_exact, s, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = float(res.fun)
                best_c1, best_c2 = float(res.x[0]), float(res.x[1])
        except Exception:
            pass

    # Phase 3: Final Polish
    if time_left() > 10:
        try:
            res = minimize(obj_exact, [best_c1, best_c2], method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = float(res.fun)
                best_c1, best_c2 = float(res.x[0]), float(res.x[1])
        except Exception:
            pass

    if not np.isfinite(best_val):
        return np.array([seed_c0, seed_c1, seed_c2])

    return np.array([1.0, float(best_c1), float(best_c2)])
