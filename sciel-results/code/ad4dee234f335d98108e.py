"""Rewritten program: Gram-Schmidt parameterization + differential evolution."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.special import factorial
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
            h0_0 = H0.subs(sp.Symbol('x'), 0)
            h4_0 = H4.subs(sp.Symbol('x'), 0)
            h8_0 = H8.subs(sp.Symbol('x'), 0)
            h12_0 = H12.subs(sp.Symbol('x'), 0)
            
            c3_s = -(c0_s * h0_0 + c1_s * h4_0 + c2_s * h8_0) / h12_0
            
            x = sp.Symbol('x')
            g = c0_s * H0 + c1_s * H4 + c2_s * H8 + c3_s * H12
            
            # Check if g tends to negative limit at infinity
            # For Hermite polynomials, the leading term dominates.
            # g ~ (c0*0 + c1*H4_lead + c2*H8_lead + c3*H12_lead) * x^12
            # Actually, H_n(x) ~ 2^n x^n for large x.
            # H0 = 1, H4 ~ 16x^4, H8 ~ 256x^8, H12 ~ 4096x^12
            # So g ~ (c1*16*x^4 + c2*256*x^8 + c3*4096*x^12) for large x
            # The dominant term is c3*4096*x^12.
            # If c3 < 0, g -> -infinity, so we replace g by -g.
            if c3_s < 0:
                g = -g
                c0_s = -c0_s
                c1_s = -c1_s
                c2_s = -c2_s
                c3_s = -c3_s
            
            # q = g / x^2. This must divide exactly.
            # g is a polynomial of degree at most 12.
            # g(0) = 0 by construction, so x divides g.
            # We need x^2 to divide g, i.e., g'(0) = 0.
            gp = sp.diff(g, x)
            gp_0 = gp.subs(x, 0)
            
            if gp_0 != 0:
                # q does not divide exactly
                return 1e10
            
            q = sp.simplify(g / x**2)
            
            # Find real roots of q where q changes sign.
            roots = sp.nroots(q, n=30)
            valid_roots = []
            for root in roots:
                if abs(root.as_real_imag()[1]) < 1e-10:
                    r_real = float(root.as_real_imag()[0])
                    # Check sign change
                    # Evaluate q at r - eps and r + eps
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
            # Hermite polynomials: H_n(x)
            # We'll use scipy or manual computation
            # For efficiency, we'll compute coefficients of g and q
            
            # Hermite polynomial coefficients (physicists' convention)
            # H_0(x) = 1
            # H_4(x) = 16x^4 - 48x^2 + 12
            # H_8(x) = 256x^8 - 3072x^6 + 13440x^4 - 26880x^2 + 1680
            # H_12(x) = 4096x^12 - 98304x^10 + 1032192x^8 - 5806080x^6 + 16515072x^4 - 25401600x^2 + 13305600
            
            # Coefficients from highest degree to constant
            # H0: [1]
            # H4: [16, 0, -48, 0, 12]
            # H8: [256, 0, -3072, 0, 13440, 0, -26880, 0, 1680]
            # H12: [4096, 0, -98304, 0, 1032192, 0, -5806080, 0, 16515072, 0, -25401600, 0, 13305600]
            
            H0_coeffs = np.array([1.0])
            H4_coeffs = np.array([16.0, 0.0, -48.0, 0.0, 12.0])
            H8_coeffs = np.array([256.0, 0.0, -3072.0, 0.0, 13440.0, 0.0, -26880.0, 0.0, 1680.0])
            H12_coeffs = np.array([4096.0, 0.0, -98304.0, 0.0, 1032192.0, 0.0, -5806080.0, 0.0, 16515072.0, 0.0, -25401600.0, 0.0, 13305600.0])
            
            # Evaluate at x=0
            H0_0 = 1.0
            H4_0 = 12.0
            H8_0 = 1680.0
            H12_0 = 13305600.0
            
            c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
            
            # Build g coefficients (degree 12 down to 0)
            g_coeffs = np.zeros(13)
            # H0 contributes to degree 0
            g_coeffs[12] += c0 * 0  # H0 has no x^12 term
            # Actually, let's be more careful.
            # H0 is degree 0: coefficient 1 at x^0
            # H4 is degree 4: coefficients [16, 0, -48, 0, 12] for x^4, x^3, x^2, x^1, x^0
            # H8 is degree 8: coefficients [256, 0, -3072, 0, 13440, 0, -26880, 0, 1680] for x^8..x^0
            # H12 is degree 12: coefficients [4096, 0, -98304, 0, 1032192, 0, -5806080, 0, 16515072, 0, -25401600, 0, 13305600] for x^12..x^0
            
            # Pad H0 to length 13
            H0_padded = np.zeros(13)
            H0_padded[12] = 0  # x^12
            H0_padded[11] = 0
            H0_padded[10] = 0
            H0_padded[9] = 0
            H0_padded[8] = 0
            H0_padded[7] = 0
            H0_padded[6] = 0
            H0_padded[5] = 0
            H0_padded[4] = 0
            H0_padded[3] = 0
            H0_padded[2] = 0
            H0_padded[1] = 0
            H0_padded[0] = 1  # x^0
            
            # Pad H4 to length 13
            H4_padded = np.zeros(13)
            H4_padded[8] = 16.0  # x^8? No, H4 is degree 4.
            # H4_coeffs = [16, 0, -48, 0, 12] corresponds to x^4, x^3, x^2, x^1, x^0
            # So in a 13-length array (x^12 down to x^0):
            H4_padded[8] = 0
            H4_padded[7] = 0
            H4_padded[6] = 0
            H4_padded[5] = 0
            H4_padded[4] = 0
            H4_padded[3] = 0
            H4_padded[2] = 0
            H4_padded[1] = 0
            H4_padded[0] = 0
            # Let me redo this. Index 0 = x^12, index 12 = x^0
            # H4: degree 4. x^4 is at index 12-4 = 8. x^3 at index 9, x^2 at index 10, x^1 at index 11, x^0 at index 12.
            H4_padded[8] = 16.0
            H4_padded[9] = 0.0
            H4_padded[10] = -48.0
            H4_padded[11] = 0.0
            H4_padded[12] = 12.0
            
            # H8: degree 8. x^8 at index 12-8 = 4. x^7 at 5, x^6 at 6, x^5 at 7, x^4 at 8, x^3 at 9, x^2 at 10, x^1 at 11, x^0 at 12.
            H8_padded = np.zeros(13)
            H8_padded[4] = 256.0
            H8_padded[5] = 0.0
            H8_padded[6] = -3072.0
            H8_padded[7] = 0.0
            H8_padded[8] = 13440.0
            H8_padded[9] = 0.0
            H8_padded[10] = -26880.0
            H8_padded[11] = 0.0
            H8_padded[12] = 1680.0
            
            # H12: degree 12. x^12 at index 0, x^11 at 1, ..., x^0 at 12.
            H12_padded = np.array([4096.0, 0.0, -98304.0, 0.0, 1032192.0, 0.0, -5806080.0, 0.0, 16515072.0, 0.0, -25401600.0, 0.0, 13305600.0])
            
            g_coeffs = c0 * H0_padded + c1 * H4_padded + c2 * H8_padded + c3 * H12_padded
            
            # Check if g -> -infinity as x -> infinity
            # Leading coefficient is g_coeffs[0] (x^12 term)
            if g_coeffs[0] < 0:
                g_coeffs = -g_coeffs
                c0 = -c0
                c1 = -c1
                c2 = -c2
                c3 = -c3
            
            # q = g / x^2. This requires g[12] = 0 and g[11] = 0 (x^0 and x^1 coefficients are zero)
            # g_coeffs[12] is x^0 coefficient, g_coeffs[11] is x^1 coefficient
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
                    # Check sign change using numerical evaluation
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
    # Bounds for c0, c1, c2
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
