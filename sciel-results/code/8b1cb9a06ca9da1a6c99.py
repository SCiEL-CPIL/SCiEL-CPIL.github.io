"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.special import factorial

# Deadline guard: 210s limit, use 195s for safety
_START_TIME = time.time()
_DEADLINE = _START_TIME + 195

def _time_left():
    return _DEADLINE - time.time()

def _poly_coefficients(a0, a4, a8):
    """
    Return the coefficients of P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
    where a12 is chosen so P(0) = 0.
    Returns coefficients [c0, c2, c4, c6, c8, c10, c12] for P(x) = sum c_k * x^k
    (only even powers).
    """
    H0_0, H4_0, H8_0, H12_0 = 1.0, 12.0, 3024.0, 10321920.0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    # H0 = 1
    # H4 = 16x^4 - 48x^2 + 12
    # H8 = 256x^8 - 4608x^6 + 23040x^4 - 34560x^2 + 10080
    # H12 = 4096x^12 - 147456x^10 + 2090112x^8 - 12902400x^6 + 34406400x^4 - 34406400x^2 + 10321920
    
    c0 = a0 * 1.0 + a4 * 12.0 + a8 * 10080.0 + a12 * 10321920.0
    c2 = a4 * (-48.0) + a8 * (-34560.0) + a12 * (-34406400.0)
    c4 = a4 * 16.0 + a8 * 23040.0 + a12 * 34406400.0
    c6 = a8 * (-4608.0) + a12 * (-12902400.0)
    c8 = a8 * 256.0 + a12 * 2090112.0
    c10 = a12 * (-147456.0)
    c12 = a12 * 4096.0
    
    return np.array([c0, c2, c4, c6, c8, c10, c12])

def _find_real_roots(coeffs, tol=1e-10):
    """
    Find real roots of a polynomial given by coefficients [c0, c2, c4, ..., c12]
    where the polynomial is c0 + c2*x^2 + c4*x^4 + ... + c12*x^12.
    Returns sorted list of real roots.
    """
    # Convert to standard polynomial form: c0 + c2*t + c4*t^2 + ... + c12*t^6 where t = x^2
    t_coeffs = np.poly1d(coeffs[::-1])  # highest degree first
    t_roots = np.roots(t_coeffs)
    
    # Filter for real roots of t (t >= 0 for real x)
    real_t_roots = []
    for r in t_roots:
        if abs(r.imag) < tol:
            real_part = r.real
            if real_part >= -tol:  # allow small negative due to numerical error
                real_t_roots.append(max(0.0, real_part))
    
    # Convert to x roots: x = ±sqrt(t)
    x_roots = set()
    for t in real_t_roots:
        if t < tol:
            x_roots.add(0.0)
        else:
            sqrt_t = np.sqrt(t)
            x_roots.add(sqrt_t)
            x_roots.add(-sqrt_t)
    
    # Remove duplicates and sort
    x_roots = sorted(set(round(r, 12) for r in x_roots))
    return x_roots

def _compute_l1_exact(a0, a4, a8):
    """
    Compute ||g||_1 exactly by finding zeros of P(x) and integrating piecewise.
    g(x) = P(x) * exp(-x^2/2)
    ||g||_1 = integral |P(x)| * exp(-x^2/2) dx
    
    Since P is even, ||g||_1 = 2 * integral_0^inf |P(x)| * exp(-x^2/2) dx
    
    We find the positive roots of P(x), split [0, inf) into intervals,
    and compute the integral on each interval.
    """
    coeffs = _poly_coefficients(a0, a4, a8)
    
    # Find real roots
    roots = _find_real_roots(coeffs)
    
    # Filter to positive roots (we'll double for symmetry)
    pos_roots = [r for r in roots if r > 1e-10]
    pos_roots.sort()
    
    # Define P(x) for evaluation
    def P(x):
        x2 = x**2
        return (coeffs[0] + coeffs[1]*x2 + coeffs[2]*x2**2 + coeffs[3]*x2**3 +
                coeffs[4]*x2**4 + coeffs[5]*x2**5 + coeffs[6]*x2**6)
    
    # Integrate |P(x)| * exp(-x^2/2) from 0 to inf
    # Split at positive roots
    boundaries = [0.0] + pos_roots + [np.inf]
    
    total = 0.0
    for i in range(len(boundaries) - 1):
        a = boundaries[i]
        b = boundaries[i+1]
        
        if b == np.inf:
            # For the tail, use numerical quadrature
            from scipy.integrate import quad
            # Check sign at a point in this interval
            mid = a + 1.0 if a < 100 else a * 1.1
            sign = np.sign(P(mid))
            if sign == 0:
                sign = 1.0
            
            # Integrate sign * P(x) * exp(-x^2/2) from a to inf
            if a < 1e-10:
                f = lambda x: sign * P(x) * np.exp(-x**2/2.0)
                val, _ = quad(f, 0, 50, limit=200)
            else:
                f = lambda x: sign * P(x) * np.exp(-x**2/2.0)
                val, _ = quad(f, a, 50, limit=200)
            total += val
        else:
            # Finite interval
            from scipy.integrate import quad
            # Check sign at midpoint
            mid = (a + b) / 2.0
            sign = np.sign(P(mid))
            if sign == 0:
                sign = 1.0
            
            f = lambda x: sign * P(x) * np.exp(-x**2/2.0)
            val, _ = quad(f, a, b, limit=100)
            total += val
    
    # Double for symmetry (even function)
    l1_norm = 2.0 * total
    
    if not np.isfinite(l1_norm) or l1_norm <= 0:
        return 1e10
    
    # The bound is proportional to ||g||_1^2 (since ||F[g]||_1 = ||g||_1 up to constant)
    return l1_norm ** 2

def _compute_l1_fast(a0, a4, a8):
    """
    Fast grid-based computation for use during optimization.
    """
    coeffs = _poly_coefficients(a0, a4, a8)
    
    N = 8192
    x = np.linspace(-8, 8, N)
    dx = x[1] - x[0]
    x2 = x**2
    
    P = (coeffs[0] + coeffs[1]*x2 + coeffs[2]*x2**2 + coeffs[3]*x2**3 +
         coeffs[4]*x2**4 + coeffs[5]*x2**5 + coeffs[6]*x2**6)
    
    g = P * np.exp(-x2 / 2.0)
    l1 = np.sum(np.abs(g)) * dx
    
    bound = l1 * l1
    if not np.isfinite(bound) or bound <= 0:
        return 1e10
    return bound

def _durand_kerner_3d(f, x0, maxiter=200, tol=1e-12):
    """
    Durand-Kerner method adapted for 3D root finding.
    We're looking for critical points of f in R^3.
    """
    x = np.array(x0, dtype=float)
    for _ in range(maxiter):
        if _time_left() < 10:
            break
        # Compute gradient numerically
        eps = 1e-8
        grad = np.zeros(3)
        for i in range(3):
            xp = x.copy()
            xp[i] += eps
            xm = x.copy()
            xm[i] -= eps
            grad[i] = (f(xp) - f(xm)) / (2 * eps)
        
        if np.linalg.norm(grad) < tol:
            break
        
        # Simple gradient step with adaptive step size
        step_size = 0.1
        for _ in range(20):
            x_new = x - step_size * grad
            if f(x_new) < f(x):
                x = x_new
                break
            step_size *= 0.5
        
        if step_size < 1e-15:
            break
    
    return x

def construct():
    # Seed result
    seed_result = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Evaluate seed with the fast objective
    seed_bound = _compute_l1_fast(*seed_result)
    
    best_result = seed_result.copy()
    best_bound = seed_bound
    
    # Bounds for the three coefficients
    bounds = [(-3.0, 3.0), (-0.2, 0.2), (-0.002, 0.002)]
    
    # Phase 1: Broad random sampling to find good regions
    if _time_left() > 60:
        rng = np.random.default_rng(42)
        for _ in range(200):
            if _time_left() < 50:
                break
            x = np.array([
                rng.uniform(-3.0, 3.0),
                rng.uniform(-0.2, 0.2),
                rng.uniform(-0.002, 0.002)
            ])
            val = _compute_l1_fast(*x)
            if val < best_bound:
                best_bound = val
                best_result = x.copy()
    
    # Phase 2: Differential evolution
    if _time_left() > 50:
        try:
            de_result = differential_evolution(
                _compute_l1_fast,
                bounds,
                seed=42,
                maxiter=100,
                tol=1e-8,
                polish=True,
                init='sobol',
                popsize=15
            )
            if de_result.fun < best_bound:
                best_bound = de_result.fun
                best_result = de_result.x.copy()
        except Exception:
            pass
    
    # Phase 3: Multi-start local refinement
    starts = [
        best_result,
        seed_result,
        np.array([-1.0, 0.04, 0.0003]),
        np.array([-1.2, 0.05, 0.0004]),
        np.array([-0.9, 0.03, 0.0002]),
        np.array([-1.5, 0.06, 0.0005]),
        np.array([-0.8, 0.02, 0.0001]),
        np.array([-1.1, 0.045, 0.00035]),
        np.array([-1.3, 0.055, 0.00045]),
    ]
    
    for start in starts:
        if _time_left() < 20:
            break
        try:
            res = minimize(
                _compute_l1_fast,
                start,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res.fun < best_bound:
                best_bound = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Fine-tune with L-BFGS-B
    if _time_left() > 15:
        try:
            res = minimize(
                _compute_l1_fast,
                best_result,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.fun < best_bound:
                best_bound = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    # Phase 5: Verify with exact computation and try to refine
    if _time_left() > 5:
        try:
            exact_val = _compute_l1_exact(*best_result)
            # If exact is better, try a few more local steps guided by exact
            if exact_val < best_bound:
                best_bound = exact_val
        except Exception:
            pass
    
    return best_result
# EVOLVE-BLOCK-END
