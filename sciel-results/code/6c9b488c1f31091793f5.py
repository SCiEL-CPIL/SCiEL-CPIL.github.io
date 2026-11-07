"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import brentq
import time

def _compute_q_coeffs(c0, c1, c2):
    """Compute coefficients of q(x) = g(x)/x^2 as a polynomial in x (degree 10).
    q(x) = a10*x^10 + a8*x^8 + a6*x^6 + a4*x^4 + a2*x^2 + a0
    (only even powers since g is even)
    """
    c3 = -(c0 + 3*c1 + 15*c2) / 105.0
    
    # q(x) = c3*(x^10 - 60x^8 + 900x^6 - 3600x^4 + 2880x^2 - 432)
    #       + c2*(x^6 - 24x^4 + 72x^2 - 32)
    #       + c1*(x^2 - 6)
    a10 = c3
    a8 = -60*c3
    a6 = 900*c3 + c2
    a4 = -3600*c3 - 24*c2
    a2 = 2880*c3 + 72*c2 + c1
    a0 = -432*c3 - 32*c2 - 6*c1
    
    return np.array([a10, a8, a6, a4, a2, a0])  # coefficients for x^10, x^8, x^6, x^4, x^2, x^0

def _eval_q(x, coeffs):
    """Evaluate q(x) given coefficients [a10, a8, a6, a4, a2, a0]"""
    a10, a8, a6, a4, a2, a0 = coeffs
    x2 = x*x
    x4 = x2*x2
    x6 = x4*x2
    x8 = x4*x4
    x10 = x8*x2
    return a10*x10 + a8*x8 + a6*x6 + a4*x4 + a2*x2 + a0

def _eval_q_y(y, coeffs):
    """Evaluate q as polynomial in y = x^2: p(y) = a10*y^5 + a8*y^4 + a6*y^3 + a4*y^2 + a2*y + a0"""
    a10, a8, a6, a4, a2, a0 = coeffs
    return a10*y**5 + a8*y**4 + a6*y**3 + a4*y**2 + a2*y + a0

def _eval_q_y_deriv(y, coeffs):
    """Derivative of p(y) with respect to y"""
    a10, a8, a6, a4, a2, a0 = coeffs
    return 5*a10*y**4 + 4*a8*y**3 + 3*a6*y**2 + 2*a4*y + a2

def _find_roots_in_interval(p, dp, a, b, max_iter=100, tol=1e-14):
    """Find all roots of p in [a,b] using bisection + Newton refinement."""
    roots = []
    # Subdivide interval into smaller pieces
    n_sub = 200
    xs = np.linspace(a, b, n_sub + 1)
    vals = np.array([p(x) for x in xs])
    
    for i in range(n_sub):
        if vals[i] == 0:
            roots.append(xs[i])
        elif vals[i] * vals[i+1] < 0:
            # Sign change in [xs[i], xs[i+1]], use bisection + Newton
            lo, hi = xs[i], xs[i+1]
            for _ in range(max_iter):
                mid = 0.5 * (lo + hi)
                pmid = p(mid)
                if abs(pmid) < tol:
                    lo = hi = mid
                    break
                if p(lo) * pmid < 0:
                    hi = mid
                else:
                    lo = mid
            # Newton refinement
            x = 0.5 * (lo + hi)
            for _ in range(20):
                dx = p(x) / dp(x) if dp(x) != 0 else 0
                x_new = x - dx
                if abs(x_new - x) < 1e-15:
                    break
                x = x_new
            if a <= x <= b and abs(p(x)) < 1e-10:
                roots.append(x)
    return roots

def _find_largest_sign_changing_root(coeffs):
    """Find the largest real root where q changes sign.
    coeffs: [a10, a8, a6, a4, a2, a0] for q(x) = a10*x^10 + a8*x^8 + a6*x^6 + a4*x^4 + a2*x^2 + a0
    """
    # Work in y = x^2 space: p(y) = a10*y^5 + a8*y^4 + a6*y^3 + a4*y^2 + a2*y + a0
    # Find all real roots y >= 0 of p(y), then x = sqrt(y)
    
    # Determine search range for y
    # Leading coefficient a10 = c3. If c3 > 0, p(y) -> +inf as y->inf
    # If c3 < 0, p(y) -> -inf as y->inf
    a10 = coeffs[0]
    
    # Find a large enough bound
    y_max = 10.0
    while y_max < 1000:
        val = _eval_q_y(y_max, coeffs)
        if abs(val) > 1e6:
            break
        y_max *= 2
    
    # Find roots of p(y) in [0, y_max]
    p = lambda y: _eval_q_y(y, coeffs)
    dp = lambda y: _eval_q_y_deriv(y, coeffs)
    
    y_roots = _find_roots_in_interval(p, dp, 0.0, y_max)
    
    if not y_roots:
        return None
    
    # For each root y, check if it's a sign-changing root of q(x)
    # q(x) = p(x^2), so at x = sqrt(y), q changes sign iff p changes sign at y
    # (since x^2 is monotonic for x > 0)
    
    best_r = None
    for yr in y_roots:
        if yr < 0:
            continue
        x = np.sqrt(yr)
        # Check sign change of q at x
        eps = 1e-10 * max(1, x)
        q_minus = _eval_q(x - eps, coeffs)
        q_plus = _eval_q(x + eps, coeffs)
        if q_minus * q_plus < 0:
            if best_r is None or x > best_r:
                best_r = x
    
    return best_r

def _objective(params):
    """Objective: minimize r^2 where r is largest sign-changing root of q"""
    c0, c1, c2 = params
    coeffs = _compute_q_coeffs(c0, c1, c2)
    r = _find_largest_sign_changing_root(coeffs)
    if r is None:
        return 1e10
    return r*r

def _objective_smooth(params, beta=50.0):
    """Smooth surrogate: approximate min over sign-changing roots using soft-min.
    This helps the optimizer navigate the nonsmooth landscape."""
    c0, c1, c2 = params
    coeffs = _compute_q_coeffs(c0, c1, c2)
    r = _find_largest_sign_changing_root(coeffs)
    if r is None:
        return 1e10
    # For smoothness, we can also add a penalty based on how "close" other roots are
    # to being sign-changing, but for now just return r^2
    return r*r

def construct():
    start_time = time.time()
    deadline = 200.0  # safety margin before 210s limit
    
    # Seed
    seed = np.array([-113/100, 1/25, 1/3240])
    best_params = seed.copy()
    best_value = _objective(seed)
    
    # Multi-start optimization
    rng = np.random.RandomState(42)
    n_starts = 80
    
    for i in range(n_starts):
        if time.time() > deadline - 15:
            break
        
        if i == 0:
            x0 = seed
        elif i < n_starts // 3:
            # Perturbation around seed
            x0 = seed + rng.randn(3) * 0.05
        elif i < 2 * n_starts // 3:
            # Wider perturbation
            x0 = seed + rng.randn(3) * 0.5
        else:
            # Uniform in reasonable range
            x0 = rng.uniform([-3, -3, -3], [3, 3, 3])
        
        try:
            # Use COBYLA for derivative-free optimization with tighter tolerance
            result = minimize(_objective, x0, method='Nelder-Mead', 
                            options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 
                                    'adaptive': True})
            if result.fun < best_value:
                best_value = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Final polish: try to improve with very tight tolerances
    if time.time() < deadline - 5:
        try:
            result = minimize(_objective, best_params, method='Nelder-Mead', 
                            options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16,
                                    'adaptive': True})
            if result.fun < best_value:
                best_value = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
