"""Optimized search for uncertainty inequality coefficients."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import hermite
from scipy import interpolate

# Precompute Hermite polynomials (physicists' convention) up to degree 12
_H = [hermite(n) for n in range(13)]
_H0 = _H[0].coeffs
_H4 = _H[4].coeffs
_H8 = _H[8].coeffs
_H12 = _H[12].coeffs

# Pad to same length (13 coefficients, degree 12)
def _pad(c, n=13):
    c = np.asarray(c, dtype=float)
    if len(c) < n:
        c = np.concatenate([np.zeros(n - len(c)), c])
    return c

_H0p = _pad(_H0)
_H4p = _pad(_H4)
_H8p = _pad(_H8)
_H12p = _pad(_H12)

# g = c0*H0 + c1*H4 + c2*H8 + c3*H12, with c3 chosen so g(0)=0
# g(0) = c0*H0(0) + c1*H4(0) + c2*H8(0) + c3*H12(0) = 0
# H_n(0) for physicists: H_n(0) = 0 if n odd; if n even, H_n(0) = (-1)^(n/2) * (n-1)!! 
# H0(0)=1, H4(0) = -12, H8(0) = 1680, H12(0) = -1935360? Let me compute.
def _H_at_0(n):
    if n % 2 == 1:
        return 0.0
    # H_n(0) = (-1)^(n/2) * (n-1)!! * ... actually use the polynomial
    c = _H[n].coeffs
    # constant term is last element
    return float(c[-1])

_H0_0 = _H_at_0(0)
_H4_0 = _H_at_0(4)
_H8_0 = _H_at_0(8)
_H12_0 = _H_at_0(12)

def _build_g_coeffs(c0, c1, c2):
    """Return coefficient vector of g (degree <= 12)."""
    c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    coeffs = c0 * _H0p + c1 * _H4p + c2 * _H8p + c3 * _H12p
    return coeffs

def _value_from(c0, c1, c2):
    """Compute the objective value = r^2/(2*pi)."""
    coeffs = _build_g_coeffs(c0, c1, c2)
    # Check leading coefficient: if g tends to negative at infinity, flip
    # Leading coeff is coeffs[0] (highest degree)
    if coeffs[0] < 0:
        coeffs = -coeffs
    # q = g / x^2. Since g(0)=0 and g is degree 12 with g(0)=0, g = x^2 * q where q is degree 10.
    # coeffs: [a12, a11, ..., a0]. Since g(0)=0, a0=0. q has coeffs [a12, a11, ..., a2].
    q_coeffs = coeffs[:-2]  # drop a0 and a1 (a1 should be 0 too since g(0)=0 implies a1=0? Actually g(0)=a0=0)
    # Actually g(0)=0 means a0=0. g/x^2 = a12 x^10 + a11 x^9 + ... + a2. So q_coeffs = coeffs[0:-2]
    # Find roots of q
    roots = np.roots(q_coeffs)
    real_roots = roots[np.abs(roots.imag) < 1e-8].real
    # We need the largest real root at which q changes sign.
    # q is a degree 10 polynomial. For even degree with positive leading coeff (after flip),
    # q -> +inf as x -> +/-inf. We need roots where sign changes.
    # Sort real roots
    if len(real_roots) == 0:
        return float('inf')
    real_roots = np.sort(real_roots)
    # Evaluate q at each root's neighborhood to check sign change
    # Better: evaluate q at midpoints between consecutive roots and at extremes
    # Find the largest root where sign changes
    # Evaluate q at points
    def q_eval(x):
        return np.polyval(q_coeffs, x)
    
    # Check sign changes: evaluate q at large negative, at each root +/- epsilon, at large positive
    # Simpler: evaluate q at (root - eps, root + eps) for each root
    best_r = None
    for r in real_roots:
        # Check if sign changes at r
        eps = 1e-7 * (1 + abs(r))
        left = q_eval(r - eps)
        right = q_eval(r + eps)
        if left * right < 0:
            if best_r is None or r > best_r:
                best_r = r
    if best_r is None:
        # No sign change? Then maybe the largest root is a double root (touching). 
        # In that case, does q "change sign"? No. So we need the largest root with sign change.
        # If none, return inf (invalid)
        return float('inf')
    return best_r**2 / (2 * np.pi)

def _objective(x):
    c0, c1, c2 = x
    return _value_from(c0, c1, c2)

def construct():
    start_time = time.time()
    deadline = start_time + 200.0  # 10s safety margin
    
    seed = np.array([-113/100, 1/25, 1/3240])
    best_val = _value_from(*seed)
    best_x = seed.copy()
    
    # Local optimization from seed
    try:
        res = minimize(_objective, seed, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    except Exception:
        pass
    
    # Try multiple random starts
    np.random.seed(42)
    n_starts = 30
    for i in range(n_starts):
        if time.time() > deadline:
            break
        # Random perturbation around seed or broader
        if i < 10:
            x0 = seed + np.random.normal(0, 0.1, 3)
        else:
            x0 = np.array([np.random.uniform(-2, 2),
                           np.random.uniform(-0.5, 1.5),
                           np.random.uniform(-0.01, 0.1)])
        try:
            res = minimize(_objective, x0, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    # Differential evolution for global search
    if time.time() < deadline - 30:
        try:
            bounds = [(-3, 3), (-1, 2), (-0.05, 0.2)]
            res_de = differential_evolution(_objective, bounds, maxiter=50,
                                            popsize=15, tol=1e-10, seed=42,
                                            polish=True, init='sobol')
            if res_de.fun < best_val:
                best_val = res_de.fun
                best_x = res_de.x
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 5:
        try:
            res = minimize(_objective, best_x, method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-16})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    return np.array(best_x)
# EVOLVE-BLOCK-END
