"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import erf, sqrt, pi
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_hermite

# H_n(0) values for even n
_H0_0 = 1.0
_H4_0 = -12.0
_H8_0 = 1680.0
_H12_0 = -145152.0

def _poly_coeffs(a0, a4, a8):
    """
    Return coefficients of the degree-12 polynomial P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
    in descending power order: [c12, c10, c8, c6, c4, c2, c0]
    P(x) = c12*x^12 + c10*x^10 + c8*x^8 + c6*x^6 + c4*x^4 + c2*x^2 + c0
    """
    a12 = (a0 - 12.0*a4 + 1680.0*a8) / 145152.0
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 42x^10 + 630x^8 - 3780x^6 + 7560x^4 - 5040x^2 + 945
    c12 = a12
    c10 = -42.0 * a12
    c8 = a8 + 630.0 * a12
    c6 = -28.0 * a8 - 3780.0 * a12
    c4 = a4 + 210.0 * a8 + 7560.0 * a12
    c2 = -6.0 * a4 - 420.0 * a8 - 5040.0 * a12
    c0 = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    return [c12, c10, c8, c6, c4, c2, c0]

def _poly_eval(x, coeffs):
    """Evaluate polynomial P(x) at x (scalar or array) using Horner's method for even powers."""
    c12, c10, c8, c6, c4, c2, c0 = coeffs
    x2 = x * x
    return ((c12 * x2 + c10) * x2 + c8) * x2 * x2 + ((c6 * x2 + c4) * x2 + c2) * x2 + c0

def _integral_abs_poly_gauss(coeffs, a, b):
    """
    Compute integral of |P(x)| * exp(-x^2/2) dx from a to b,
    assuming P(x) does not change sign in (a, b).
    Uses antiderivative of P(x)*exp(-x^2/2).
    """
    c12, c10, c8, c6, c4, c2, c0 = coeffs
    
    # Antiderivative of x^n * exp(-x^2/2):
    # For n=0: -sqrt(2*pi) * Phi(x) where Phi is the standard normal CDF, 
    #   equivalently sqrt(pi/2) * erf(x/sqrt(2))
    # For n=2: -x*exp(-x^2/2) + integral of exp(-x^2/2)
    # For n=4: -x^3*exp(-x^2/2) + 3*integral of x*exp(-x^2/2) ... 
    # Actually let's use the recursive formula:
    # I_n = integral x^n exp(-x^2/2) dx
    # I_n = -x^(n-1) exp(-x^2/2) + (n-1) * I_(n-2)
    # I_0 = sqrt(pi/2) * erf(x/sqrt(2))
    
    def _I0(x):
        return sqrt(pi/2) * erf(x / sqrt(2))
    
    def _I2(x):
        return -x * np.exp(-x**2/2) + _I0(x)
    
    def _I4(x):
        return -x**3 * np.exp(-x**2/2) + 3 * _I2(x)
    
    def _I6(x):
        return -x**5 * np.exp(-x**2/2) + 5 * _I4(x)
    
    def _I8(x):
        return -x**7 * np.exp(-x**2/2) + 7 * _I6(x)
    
    def _I10(x):
        return -x**9 * np.exp(-x**2/2) + 9 * _I8(x)
    
    def _I12(x):
        return -x**11 * np.exp(-x**2/2) + 11 * _I10(x)
    
    def F(x):
        val = c0 * _I0(x) + c2 * _I2(x) + c4 * _I4(x) + c6 * _I6(x) + c8 * _I8(x) + c10 * _I10(x) + c12 * _I12(x)
        return val
    
    return F(b) - F(a)

def _find_zeros(coeffs):
    """Find all real roots of the degree-12 polynomial P(x)."""
    # Build full coefficient array for numpy.roots (descending order)
    c12, c10, c8, c6, c4, c2, c0 = coeffs
    full_coeffs = np.array([c12, 0, c10, 0, c8, 0, c6, 0, c4, 0, c2, 0, c0])
    roots = np.roots(full_coeffs)
    # Filter real roots
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    return np.sort(real_roots)

def _compute_bound_analytic(a0, a4, a8):
    """
    Compute the C4 uncertainty bound analytically.
    Since g = ĝ (Fourier transform is identical for even-degree Hermite-Gaussians with degrees ≡ 0 mod 4),
    the bound is ||g||_1 / min|g|.
    """
    coeffs = _poly_coeffs(a0, a4, a8)
    
    # Find all real zeros of P(x)
    zeros = _find_zeros(coeffs)
    
    # The function g(x) = P(x) * exp(-x^2/2) has the same zeros as P(x)
    # We need to compute ||g||_1 = integral |P(x)| exp(-x^2/2) dx over R
    # and min|g(x)| = min |P(x)| exp(-x^2/2)
    
    # For ||g||_1: split at zeros and integrate piecewise
    # Extend with -inf and +inf
    # Use a large cutoff since exp(-x^2/2) decays rapidly
    cutoff = 20.0  # exp(-200/2) = exp(-100) ~ 1e-44, negligible
    
    # Sort zeros within [-cutoff, cutoff]
    zeros_clipped = zeros[(zeros >= -cutoff) & (zeros <= cutoff)]
    
    # Build integration intervals
    boundaries = np.concatenate([[-cutoff], zeros_clipped, [cutoff]])
    
    l1_norm = 0.0
    for i in range(len(boundaries) - 1):
        a = boundaries[i]
        b = boundaries[i + 1]
        if b - a < 1e-15:
            continue
        # Check sign at midpoint
        mid = (a + b) / 2
        sign = np.sign(_poly_eval(mid, coeffs))
        if sign == 0:
            sign = 1.0
        integral = _integral_abs_poly_gauss(coeffs, a, b)
        l1_norm += sign * integral
    
    l1_norm = abs(l1_norm)
    
    # For min|g(x)|: find critical points of |P(x)|exp(-x^2/2)
    # |P(x)|exp(-x^2/2) has minima at:
    # 1. Zeros of P(x) (where value is 0) - but we need to check if these are isolated
    # 2. Critical points where d/dx [P(x)exp(-x^2/2)] = 0, i.e., P'(x) - x*P(x) = 0
    # 3. The value at the zeros is 0, so if P has any real zero, min is 0 and bound is infinite!
    
    # Wait - if P(x) has a real zero, then g(x) = 0 at that point, and min|g| = 0,
    # making the bound infinite. So we need P(x) to have NO real zeros!
    
    # Check: does P(x) have any real zeros?
    if len(zeros) > 0:
        # Check if any zero is "real" (not a numerical artifact)
        # Evaluate P at each zero to confirm it's close to 0
        for z in zeros:
            val = _poly_eval(z, coeffs)
            if abs(val) < 1e-8:
                return np.inf
    
    # If no real zeros, P(x) has constant sign. Find the minimum of |P(x)|exp(-x^2/2).
    # Critical points: P'(x) - x*P(x) = 0
    # P'(x) = 12*c12*x^11 + 10*c10*x^9 + 8*c8*x^7 + 6*c6*x^5 + 4*c4*x^3 + 2*c2*x
    # P'(x) - x*P(x) = 12*c12*x^11 + 10*c10*x^9 + 8*c8*x^7 + 6*c6*x^5 + 4*c4*x^3 + 2*c2*x
    #                   - x*(c12*x^12 + c10*x^10 + c8*x^8 + c6*x^6 + c4*x^4 + c2*x^2 + c0)
    # = -c12*x^13 - c10*x^11 - c8*x^9 - c6*x^7 - c4*x^5 - c2*x^3 - c0*x
    # = -x*(c12*x^12 + c10*x^10 + c8*x^8 + c6*x^6 + c4*x^4 + c2*x^2 + c0)
    # = -x*P(x)
    
    # Wait, that's interesting. P'(x) - x*P(x) = -x*P(x) only if P is a specific form.
    # Let me recompute. P(x) = c12*x^12 + c10*x^10 + c8*x^8 + c6*x^6 + c4*x^4 + c2*x^2 + c0
    # P'(x) = 12*c12*x^11 + 10*c10*x^9 + 8*c8*x^7 + 6*c6*x^5 + 4*c4*x^3 + 2*c2*x
    # x*P(x) = c12*x^13 + c10*x^11 + c8*x^9 + c6*x^7 + c4*x^5 + c2*x^3 + c0*x
    # P'(x) - x*P(x) = 12*c12*x^11 + 10*c10*x^9 + 8*c8*x^7 + 6*c6*x^5 + 4*c4*x^3 + 2*c2*x
    #                   - c12*x^13 - c10*x^11 - c8*x^9 - c6*x^7 - c4*x^5 - c2*x^3 - c0*x
    # = -c12*x^13 + (12*c12 - c10)*x^11 + (10*c10 - c8)*x^9 + (8*c8 - c6)*x^7
    #   + (6*c6 - c4)*x^5 + (4*c4 - c2)*x^3 + (2*c2 - c0)*x
    
    c12, c10, c8, c6, c4, c2, c0 = coeffs
    # Q(x) = P'(x) - x*P(x) = -c12*x^13 + (12*c12-c10)*x^11 + (10*c10-c8)*x^9 + (8*c8-c6)*x^7 + (6*c6-c4)*x^5 + (4*c4-c2)*x^3 + (2*c2-c0)*x
    # This is an odd polynomial of degree 13.
    # Q(x) = x * R(x^2) where R is degree 6 in x^2
    # Find roots of Q(x): x=0 is always a root.
    # For other roots, find roots of R(y) = 0 where y = x^2 >= 0
    
    # R(y) = -c12*y^6 + (12*c12-c10)*y^5 + (10*c10-c8)*y^4 + (8*c8-c6)*y^3 + (6*c6-c4)*y^2 + (4*c4-c2)*y + (2*c2-c0)
    R_coeffs = np.array([-c12, 0, 12*c12-c10, 0, 10*c10-c8, 0, 8*c8-c6, 0, 6*c6-c4, 0, 4*c4-c2, 0, 2*c2-c0])
    
    # Find roots of R(y) for y >= 0
    R_roots = np.roots(R_coeffs)
    R_roots_real = R_roots[np.abs(R_roots.imag) < 1e-10].real
    R_roots_pos = R_roots_real[R_roots_real > 1e-12]
    
    # Critical points: x = 0, x = ±sqrt(y) for y in R_roots_pos
    critical_points = [0.0]
    for y in R_roots_pos:
        critical_points.append(np.sqrt(y))
        critical_points.append(-np.sqrt(y))
    
    # Evaluate |g(x)| at each critical point and at boundaries
    min_val = np.inf
    for x_cp in critical_points:
        val = abs(_poly_eval(x_cp, coeffs)) * np.exp(-x_cp**2 / 2.0)
        if val < min_val:
            min_val = val
    
    # Also check at large |x| (should be ~0)
    for x_check in [-cutoff, cutoff]:
        val = abs(_poly_eval(x_check, coeffs)) * np.exp(-x_check**2 / 2.0)
        if val < min_val:
            min_val = val
    
    if min_val < 1e-15:
        return np.inf
    
    return l1_norm / min_val


def construct():
    """
    Find the three coefficients (a0, a4, a8) that minimize the C4 uncertainty bound.
    """
    start_time = time.time()
    deadline = 210.0 - 15.0  # 15 second safety margin
    
    def objective(params):
        a0, a4, a8 = params
        return _compute_bound_analytic(a0, a4, a8)
    
    # Start with the known prior
    best_a0, best_a4, best_a8 = -113/100.0, 1/25.0, 1/3240.0
    best_bound = _compute_bound_analytic(best_a0, best_a4, best_a8)
    
    # Phase 1: Coarse grid search to find good starting points
    n_coarse = 12
    a0_range = np.linspace(-2, 2, n_coarse)
    a4_range = np.linspace(-0.5, 0.5, n_coarse)
    a8_range = np.linspace(-0.05, 0.05, n_coarse)
    
    grid_best = []
    for a0 in a0_range:
        if time.time() > deadline:
            break
        for a4 in a4_range:
            if time.time() > deadline:
                break
            for a8 in a8_range:
                if time.time() > deadline:
                    break
                bound = _compute_bound_analytic(a0, a4, a8)
                if bound < best_bound:
                    best_bound = bound
                    best_a0, best_a4, best_a8 = a0, a4, a8
                if bound < 100.0:  # collect promising points
                    grid_best.append((a0, a4, a8))
    
    # Phase 2: Differential evolution for global search
    if time.time() < deadline - 30:
        try:
            bounds = [(-2.0, 2.0), (-0.5, 0.5), (-0.05, 0.05)]
            result = differential_evolution(
                objective, bounds, maxiter=200, tol=1e-12,
                seed=42, polish=True, timeout=int(deadline - time.time() - 20),
                popsize=15
            )
            if result.fun < best_bound:
                best_a0, best_a4, best_a8 = result.x
                best_bound = result.fun
        except Exception:
            pass
    
    # Phase 3: Local refinement from multiple starting points
    if time.time() < deadline - 10:
        # Start from best found and a few grid points
        starts = [(best_a0, best_a4, best_a8)]
        for pt in grid_best[:5]:
            starts.append(pt)
        
        for s0, s4, s8 in starts:
            if time.time() > deadline - 5:
                break
            try:
                result = minimize(
                    objective, [s0, s4, s8],
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
                )
                if result.fun < best_bound:
                    best_a0, best_a4, best_a8 = result.x
                    best_bound = result.fun
            except Exception:
                pass
    
    # Phase 4: Very fine local search
    if time.time() < deadline - 5:
        try:
            result = minimize(
                objective, [best_a0, best_a4, best_a8],
                method='Powell',
                options={'maxiter': 10000, 'xtol': 1e-15, 'ftol': 1e-15}
            )
            if result.fun < best_bound:
                best_a0, best_a4, best_a8 = result.x
                best_bound = result.fun
        except Exception:
            pass
    
    return np.array([best_a0, best_a4, best_a8])
# EVOLVE-BLOCK-END
