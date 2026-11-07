"""Rewritten program: Gram-Schmidt parameterization + differential evolution."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import sympy as sp
from sympy.polys.orthopolys import hermite_poly
import warnings
warnings.filterwarnings("ignore")


def _compute_value(c0, c1, c2, use_sympy=False):
    """
    Compute the objective value for given coefficients (c0, c1, c2).
    Lower is better. Returns a large penalty for invalid configurations.
    """
    try:
        if use_sympy:
            # Use exact rational arithmetic via sympy
            c0_s = sp.Rational(c0).limit_denominator(10**12)
            c1_s = sp.Rational(c1).limit_denominator(10**12)
            c2_s = sp.Rational(c2).limit_denominator(10**12)
            
            H0 = hermite_poly(0)
            H4 = hermite_poly(4)
            H8 = hermite_poly(8)
            H12 = hermite_poly(12)
            
            # Compute c3 so that g(0) = 0
            x = sp.Symbol('x')
            h0_0 = H0.subs(x, 0)
            h4_0 = H4.subs(x, 0)
            h8_0 = H8.subs(x, 0)
            h12_0 = H12.subs(x, 0)
            
            c3_s = -(c0_s * h0_0 + c1_s * h4_0 + c2_s * h8_0) / h12_0
            
            g = c0_s * H0 + c1_s * H4 + c2_s * H8 + c3_s * H12
            
            # Check if g tends to negative limit at infinity
            if c3_s < 0:
                g = -g
                c0_s = -c0_s
                c1_s = -c1_s
                c2_s = -c2_s
                c3_s = -c3_s
            
            # q = g / x^2. This must divide exactly.
            gp = sp.diff(g, x)
            gp_0 = gp.subs(x, 0)
            
            if gp_0 != 0:
                return 1e10
            
            q = sp.simplify(g / x**2)
            
            # Find real roots of q where q changes sign.
            roots = sp.nroots(q, n=30)
            valid_roots = []
            for root in roots:
                if abs(root.as_real_imag()[1]) < 1e-10:
                    r_real = float(root.as_real_imag()[0])
                    eps = 1e-6
                    q_minus = float(q.subs(x, r_real - eps))
                    q_plus = float(q.subs(x, r_real + eps))
                    if q_minus * q_plus < 0:
                        valid_roots.append(r_real)
            
            if not valid_roots:
                return 1e10
            
            r = max(valid_roots)
            value = r**2 / (2 * np.pi)
            return value
        else:
            # Numerical approach using numpy for speed
            H0_0 = 1.0
            H4_0 = 12.0
            H8_0 = 1680.0
            H12_0 = 13305600.0
            
            c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
            
            # Build g coefficients (degree 12 down to 0)
            # Index 0 = x^12, index 12 = x^0
            g_coeffs = np.zeros(13)
            
            # H0: degree 0
            g_coeffs[12] += c0 * 1.0
            
            # H4: degree 4
            # x^4 at index 8, x^2 at index 10, x^0 at index 12
            g_coeffs[8] += c1 * 16.0
            g_coeffs[10] += c1 * (-48.0)
            g_coeffs[12] += c1 * 12.0
            
            # H8: degree 8
            # x^8 at index 4, x^6 at index 6, x^4 at index 8, x^2 at index 10, x^0 at index 12
            g_coeffs[4] += c2 * 256.0
            g_coeffs[6] += c2 * (-3072.0)
            g_coeffs[8] += c2 * 13440.0
            g_coeffs[10] += c2 * (-26880.0)
            g_coeffs[12] += c2 * 1680.0
            
            # H12: degree 12
            # x^12 at index 0, x^10 at index 2, x^8 at index 4, x^6 at index 6, x^4 at index 8, x^2 at index 10, x^0 at index 12
            g_coeffs[0] += c3 * 4096.0
            g_coeffs[2] += c3 * (-98304.0)
            g_coeffs[4] += c3 * 1032192.0
            g_coeffs[6] += c3 * (-5806080.0)
            g_coeffs[8] += c3 * 16515072.0
            g_coeffs[10] += c3 * (-25401600.0)
            g_coeffs[12] += c3 * 13305600.0
            
            # Check if g -> -infinity as x -> infinity
            if g_coeffs[0] < 0:
                g_coeffs = -g_coeffs
            
            # q = g / x^2. This requires g[12] = 0 and g[11] = 0
            if abs(g_coeffs[12]) > 1e-10 or abs(g_coeffs[11]) > 1e-10:
                return 1e10
            
            # q is a polynomial of degree 10: coefficients g_coeffs[0:11] (x^10 down to x^0)
            q_coeffs = g_coeffs[0:11]
            
            # Find real roots of q
            roots = np.roots(q_coeffs)
            valid_roots = []
            for root in roots:
                if abs(root.imag) < 1e-10:
                    r_real = root.real
                    eps = 1e-6
                    q_minus = np.polyval(q_coeffs, r_real - eps)
                    q_plus = np.polyval(q_coeffs, r_real + eps)
                    if q_minus * q_plus < 0:
                        valid_roots.append(r_real)
            
            if not valid_roots:
                return 1e10
            
            r = max(valid_roots)
            value = r**2 / (2 * np.pi)
            return value
            
    except Exception as e:
        return 1e10


def _objective(params):
    """
    Objective function for differential evolution.
    params = [c0, c1, c2]
    """
    c0, c1, c2 = params
    return _compute_value(c0, c1, c2, use_sympy=False)


def construct():
    """
    Construct three coefficients [c0, c1, c2] using differential evolution.
    """
    # Initial seed
    seed = np.array([-113 / 100, 1 / 25, 1 / 3240])
    seed_value = _compute_value(seed[0], seed[1], seed[2], use_sympy=False)
    
    # Differential evolution
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    try:
        result = differential_evolution(
            _objective,
            bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-8,
            polish=True,
            init='sobol'
        )
        
        if result.success and result.fun < seed_value:
            best_c0, best_c1, best_c2 = result.x
            # Verify with sympy for exactness
            sympy_value = _compute_value(best_c0, best_c1, best_c2, use_sympy=True)
            if sympy_value < seed_value:
                return np.array([best_c0, best_c1, best_c2])
        
        # Try a few random restarts
        np.random.seed(42)
        best_value = seed_value
        best_params = seed.copy()
        
        for i in range(20):
            c0 = np.random.uniform(-5, 5)
            c1 = np.random.uniform(-5, 5)
            c2 = np.random.uniform(-5, 5)
            val = _compute_value(c0, c1, c2, use_sympy=False)
            if val < best_value:
                best_value = val
                best_params = np.array([c0, c1, c2])
        
        if best_value < seed_value:
            # Verify with sympy
            sympy_value = _compute_value(best_params[0], best_params[1], best_params[2], use_sympy=True)
            if sympy_value < seed_value:
                return best_params
        
        return seed
        
    except Exception:
        return seed
# EVOLVE-BLOCK-END
