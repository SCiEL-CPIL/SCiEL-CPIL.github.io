"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import factorial


def construct():
    """
    Tune a Fourier-invariant Hermite-polynomial times Gaussian test function to reduce
    the upper bound on A(f)*A(Fourier(f)).
    
    The test function is g(x) = exp(-x^2/2) * P(x^2) where
    P(z) = c0*H0(z) + c4*H4(z) + c8*H8(z) + c12*H12(z)
    with H_k(z) = He_k(z) (physicist's or probabilist's Hermite polynomials in z=x^2).
    
    The constraint g(0) = 0 forces c12 = -(c0 + c4 + c8) if H_k(0) = 1 for all k,
    or more generally c12 = -sum(c_i * H_i(0)) / H_12(0).
    
    For probabilist's Hermite polynomials He_k(x):
    He_0(x) = 1
    He_2(x) = x^2 - 1
    He_4(x) = x^4 - 6x^2 + 3
    He_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    He_12(x) = x^12 - 66x^10 + 1485x^8 - 11880x^6 + 29700x^4 - 27720x^2 + 10395
    
    Wait, the task says "Hermite-polynomial times Gaussian" with degrees 0,4,8,12.
    These are likely the Hermite polynomials evaluated at x^2, or perhaps the 
    standard Hermite functions. Let me reconsider.
    
    Actually, looking at the seed: coefficients [-113/100, 1/25, 1/3240] for degrees 0,4,8.
    The constraint g(0)=0 means the sum of coefficients times H_k(0) = 0.
    
    For probabilist's Hermite polynomials He_n(x):
    He_0(0) = 1
    He_4(0) = 3  (He_4(x) = x^4 - 6x^2 + 3)
    He_8(0) = 105 (He_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105)
    He_12(0) = 10395
    
    So c12 = -(c0*1 + c4*3 + c8*105) / 10395
    
    With seed: c0=-1.13, c4=0.04, c8=1/3240≈0.000308642
    c12 = -(-1.13 + 0.12 + 105*0.000308642)/10395 = -(-1.13+0.12+0.03240742)/10395
         = -(-0.97759258)/10395 ≈ 0.00009405
    
    Now for the uncertainty product bound. The key insight from the strategy is to use
    analytic properties of Hermite functions under Fourier transform.
    
    The Fourier transform of exp(-x^2/2) * He_n(x) is i^n * exp(-xi^2/2) * He_n(xi) (up to constants).
    
    Actually, let me think about this more carefully. The function is:
    g(x) = exp(-x^2/2) * [c0 + c4*He_4(x) + c8*He_8(x) + c12*He_12(x)]
    
    The Fourier transform of exp(-x^2/2)*He_n(x) is (i)^n * exp(-xi^2/2)*He_n(xi) in the 
    convention where F[f](xi) = integral f(x) exp(-i*xi*x) dx.
    
    But the task mentions "Fourier-invariant" which suggests the bound might be 
    invariant under Fourier transform, or we're looking at A(f)*A(F(f)) where A is some 
    norm or moment.
    
    Given the complexity, let me use a numerical optimization approach with an analytic 
    objective based on the known structure. The score is the upper bound on C4, which 
    likely relates to a moment ratio.
    
    Let me define the objective as the ratio of second moment to zeroth moment of |g|^2,
    which gives a variance-like quantity, and the product with the Fourier side gives 
    the uncertainty bound.
    
    Actually, for a Gaussian-modulated Hermite polynomial, we can compute everything analytically.
    Let g(x) = exp(-x^2/2) * p(x) where p(x) = sum c_k He_k(x).
    
    Then |g(x)|^2 = exp(-x^2) * p(x)^2
    
    The "A" quantity might be related to the L2 norm or a moment. Let me assume 
    A(f) = sqrt(integral |f|^2 * x^2 dx / integral |f|^2 dx) which is the RMS width.
    
    Then A(f)*A(F(f)) >= 1/2 by uncertainty principle, and we want to minimize the upper bound.
    
    For the Fourier transform: F[g](xi) = (1/sqrt(2*pi)) * integral exp(-x^2/2)*p(x)*exp(-i*xi*x) dx
    
    This equals (1/sqrt(2*pi)) * F[exp(-x^2/2)*p(x)](xi)
    
    Since p(x) is a polynomial, we can expand and use the fact that the Fourier transform 
    of exp(-x^2/2)*x^k is related to Hermite functions.
    
    Let me just set up a numerical optimization with a well-defined objective.
    """
    
    start_time = time.time()
    deadline = start_time + 200  # Safety margin before 210s limit
    
    # Hermite polynomials (probabilist's) evaluated at 0
    # He_0(x) = 1
    # He_4(x) = x^4 - 6x^2 + 3, so He_4(0) = 3
    # He_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105, so He_8(0) = 105
    # He_12(x) = x^12 - 66x^10 + 1485x^8 - 11880x^6 + 29700x^4 - 27720x^2 + 10395
    # He_12(0) = 10395
    
    H0_0 = 1.0
    H4_0 = 3.0
    H8_0 = 105.0
    H12_0 = 10395.0
    
    def compute_c12(c0, c4, c8):
        """Compute c12 to satisfy g(0) = 0."""
        return -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    def objective(coeffs):
        """
        Compute the uncertainty product upper bound.
        
        coeffs = [c0, c4, c8]
        c12 is determined by the constraint g(0) = 0.
        
        The objective is A(f) * A(F(f)) where A(f) = sqrt(<x^2>/<1>) for |f|^2.
        """
        c0, c4, c8 = coeffs
        c12 = compute_c12(c0, c4, c8)
        
        # g(x) = exp(-x^2/2) * [c0*He_0(x) + c4*He_4(x) + c8*He_8(x) + c12*He_12(x)]
        # |g(x)|^2 = exp(-x^2) * [c0 + c4*He_4(x) + c8*He_8(x) + c12*He_12(x)]^2
        
        # We need <1> = integral |g|^2 dx and <x^2> = integral x^2 |g|^2 dx
        # And similarly for the Fourier transform.
        
        # By Parseval, integral |g|^2 dx = integral |F[g]|^2 dxi
        # So <1> is the same on both sides.
        
        # For the Fourier transform side, we need <xi^2> = integral xi^2 |F[g](xi)|^2 dxi
        
        # The Fourier transform of exp(-x^2/2)*He_n(x) is (i)^n * exp(-xi^2/2)*He_n(xi)
        # (up to normalization constants that cancel in the ratio)
        
        # So F[g](xi) is proportional to exp(-xi^2/2) * [c0*He_0(xi) + c4*(i)^4*He_4(xi) + 
        #              c8*(i)^8*He_8(xi) + c12*(i)^12*He_12(xi)]
        # Since (i)^4 = 1, (i)^8 = 1, (i)^12 = -1
        # F[g](xi) ∝ exp(-xi^2/2) * [c0*He_0(xi) + c4*He_4(xi) + c8*He_8(xi) - c12*He_12(xi)]
        
        # So on the Fourier side, the polynomial is:
        # q(xi) = c0*He_0(xi) + c4*He_4(xi) + c8*He_8(xi) - c12*He_12(xi)
        
        # <1> = integral exp(-x^2) * p(x)^2 dx where p(x) = c0*He_0(x) + c4*He_4(x) + c8*He_8(x) + c12*He_12(x)
        # <x^2> = integral x^2 * exp(-x^2) * p(x)^2 dx
        # <xi^2> = integral xi^2 * exp(-xi^2) * q(xi)^2 dxi
        
        # We can compute these analytically using the orthogonality of Hermite polynomials.
        # integral exp(-x^2) He_m(x) He_n(x) dx = sqrt(pi) * 2^n * n! * delta_mn
        
        # But p(x)^2 is a product of two linear combinations, so we need:
        # integral exp(-x^2) He_m(x) He_n(x) He_p(x) He_q(x) dx
        
        # This is more complex. Let me use numerical integration for reliability.
        
        from scipy.integrate import quad
        
        def p_func(x):
            return (c0 * (x**0) +  # He_0(x) = 1
                    c4 * (x**4 - 6*x**2 + 3) +  # He_4(x)
                    c8 * (x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105) +  # He_8(x)
                    c12 * (x**12 - 66*x**10 + 1485*x**8 - 11880*x**6 + 29700*x**4 - 27720*x**2 + 10395))
        
        def q_func(x):
            # Fourier side: c12 gets a minus sign due to (i)^12 = -1
            return (c0 * (x**0) +
                    c4 * (x**4 - 6*x**2 + 3) +
                    c8 * (x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105) +
                    (-c12) * (x**12 - 66*x**10 + 1485*x**8 - 11880*x**6 + 29700*x**4 - 27720*x**2 + 10395))
        
        def integrand0_p(x):
            return np.exp(-x**2) * p_func(x)**2
        
        def integrand2_p(x):
            return x**2 * np.exp(-x**2) * p_func(x)**2
        
        def integrand2_q(x):
            return x**2 * np.exp(-x**2) * q_func(x)**2
        
        # Numerical integration
        try:
            norm, err1 = quad(integrand0_p, -np.inf, np.inf, limit=100)
            moment2_p, err2 = quad(integrand2_p, -np.inf, np.inf, limit=100)
            moment2_q, err3 = quad(integrand2_q, -np.inf, np.inf, limit=100)
        except Exception:
            return 1e10
        
        if norm <= 0 or not np.isfinite(norm):
            return 1e10
        
        var_x = moment2_p / norm
        var_xi = moment2_q / norm
        
        if var_x < 0 or var_xi < 0:
            return 1e10
        
        # The uncertainty product is sqrt(var_x) * sqrt(var_xi)
        # But the "upper bound on C4" might be this product or a scaled version.
        # The seed score is 1.0, so let's compute what the seed gives and calibrate.
        
        return np.sqrt(var_x * var_xi)
    
    # Start with the seed values
    seed_coeffs = np.array([-113/100, 1/25, 1/3240])
    seed_obj = objective(seed_coeffs)
    
    # Try to improve using optimization
    best_coeffs = seed_coeffs.copy()
    best_obj = seed_obj
    
    # Use multiple starting points and local optimization
    start_points = [
        seed_coeffs.copy(),
        np.array([-1.0, 0.05, 0.0003]),
        np.array([-1.2, 0.03, 0.0004]),
        np.array([-1.1, 0.045, 0.00028]),
        np.array([-1.15, 0.038, 0.00032]),
    ]
    
    for sp in start_points:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(objective, sp, method='Nelder-Mead', 
                             options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if result.fun < best_obj and np.isfinite(result.fun):
                best_obj = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 30:
        try:
            result = minimize(objective, best_coeffs, method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if result.fun < best_obj and np.isfinite(result.fun):
                best_obj = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
