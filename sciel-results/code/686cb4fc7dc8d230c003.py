"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial

# Deadline guard: kill after 200s, 10s margin before 210s limit
_START_TIME = time.time()
_DEADLINE = _START_TIME + 200.0

def _time_remaining():
    return _DEADLINE - time.time()

def _hermite_values_at_zero():
    """
    Physicists' Hermite polynomials H_n(0):
    H_0(0) = 1
    H_2(0) = -1
    H_4(0) = 3
    H_6(0) = -15
    H_8(0) = 105
    H_10(0) = -945
    H_12(0) = 10395
    
    In general H_{2k}(0) = (-1)^k (2k)! / k!
    """
    H0 = 1.0
    H4 = 3.0
    H8 = 105.0
    H12 = 10395.0
    return H0, H4, H8, H12

def _compute_bound(a0, a4, a8):
    """
    Compute the upper bound on C4 for the Hermite-Gaussian test function.
    
    The test function is g(x) = exp(-x^2/2) * (c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x))
    where c12 is determined by g(0) = 0, i.e., c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    
    The uncertainty product A(f)*A(F) is bounded by a constant C4 that depends on the coefficients.
    We need to minimize this bound.
    
    For a Gaussian times Hermite polynomial, the key quantities are:
    - The L2 norm of g
    - The L2 norm of g' (or related quantities)
    - The Fourier transform properties
    
    Actually, let me think about this more carefully. The task says "upper bound on A(f)*A(Fourier(f))".
    A(f) typically denotes the spread (variance) of the function, and A(Fourier(f)) the spread of its Fourier transform.
    By the uncertainty principle, A(f)*A(F) >= 1/4 (in some conventions) or similar.
    
    For a test function g, we want to minimize the product of the variances of g and its Fourier transform.
    The minimum is achieved by pure Gaussians, but here we're using a perturbed Gaussian.
    
    Actually, re-reading: "Tune a Fourier-invariant Hermite-polynomial times Gaussian test function to reduce the upper bound on A(f)*A(Fourier(f))"
    
    The "upper bound on C4" suggests C4 is a constant in an inequality. Let me think about what C4 represents.
    
    In the context of uncertainty inequalities, if we have:
    (Δx)^2 * (Δp)^2 >= C
    
    Then for a specific test function, we can compute the exact value. But the "upper bound on C4" suggests we're looking for the best constant in some inequality.
    
    Actually, I think the problem is: given the test function g, compute the quantity that bounds the uncertainty product, and minimize it.
    
    Let me reconsider. The product A(f)*A(F(f)) for a function f is related to the variance. For g(x) = exp(-x^2/2) * P(x) where P is a polynomial:
    
    Δx^2 = <x^2> - <x>^2
    Δk^2 (in Fourier space) = <k^2> - <k>^2
    
    And by Parseval, these can be computed from g.
    
    For even functions (which we have, since H_{2k} are even), <x> = 0 and <k> = 0.
    
    So Δx^2 = <x^2> and Δk^2 = <k^2>.
    
    <x^2> = ∫ x^2 |g(x)|^2 dx / ∫ |g(x)|^2 dx
    <k^2> = ∫ k^2 |G(k)|^2 dk / ∫ |G(k)|^2 dk
    
    By Parseval, ∫ |G(k)|^2 dk = (1/2π) ∫ |g(x)|^2 dx (depending on convention).
    
    And ∫ k^2 |G(k)|^2 dk = (1/2π) ∫ |g'(x)|^2 dx (for appropriate Fourier transform convention).
    
    So the product is:
    (Δx)^2 * (Δk)^2 = [∫ x^2 |g|^2 dx / ∫ |g|^2 dx] * [∫ |g'|^2 dx / ∫ |g|^2 dx]
    
    Wait, that's not quite right. Let me be more careful.
    
    With Fourier transform G(k) = ∫ g(x) e^{-ikx} dx (no normalization), Parseval gives:
    ∫ |G(k)|^2 dk = 2π ∫ |g(x)|^2 dx
    
    And ∫ k^2 |G(k)|^2 dk = 2π ∫ |g'(x)|^2 dx (assuming g decays fast enough)
    
    So:
    <k^2> = ∫ k^2 |G(k)|^2 dk / ∫ |G(k)|^2 dk = ∫ |g'(x)|^2 dx / ∫ |g(x)|^2 dx
    
    And <x^2> = ∫ x^2 |g(x)|^2 dx / ∫ |g(x)|^2 dx
    
    So the product A(f)*A(F) = <x^2> * <k^2> = [∫ x^2 |g|^2 dx] * [∫ |g'|^2 dx] / [∫ |g|^2 dx]^2
    
    Now, for g(x) = exp(-x^2/2) * P(x), where P is a polynomial:
    
    Let's compute these integrals. Since g is even, all integrals of odd functions vanish.
    
    Let P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    
    Actually, let me just numerically compute the bound for given coefficients and minimize it.
    
    The constraint is g(0) = 0, so c12 = -(c0*H0(0) + c4*H4(0) + c8*H8(0)) / H12(0)
    = -(c0 + 3*c4 + 105*c8) / 10395
    
    Let me compute the objective function numerically.
    """
    if _time_remaining() < 5.0:
        return None
    
    H0_0, H4_0, H8_0, H12_0 = _hermite_values_at_zero()
    
    # Compute c12 from constraint g(0) = 0
    c12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    # Define the polynomial P(x)
    # H0(x) = 1
    # H4(x) = x^4 - 6x^2 + 3
    # H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12(x) = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    
    def P(x):
        H0 = np.ones_like(x)
        H4 = x**4 - 6*x**2 + 3
        H8 = x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
        H12 = x**12 - 66*x**10 + 1155*x**8 - 6930*x**6 + 13860*x**4 - 7560*x**2 + 945
        return a0 * H0 + a4 * H4 + a8 * H8 + c12 * H12
    
    def g(x):
        return np.exp(-x**2 / 2.0) * P(x)
    
    def g_prime(x):
        # g'(x) = exp(-x^2/2) * P'(x) - x * exp(-x^2/2) * P(x)
        #      = exp(-x^2/2) * (P'(x) - x * P(x))
        H0 = np.ones_like(x)
        H4 = x**4 - 6*x**2 + 3
        H8 = x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
        H12 = x**12 - 66*x**10 + 1155*x**8 - 6930*x**6 + 13860*x**4 - 7560*x**2 + 945
        
        H0_p = np.zeros_like(x)
        H4_p = 4*x**3 - 12*x
        H8_p = 8*x**7 - 168*x**5 + 840*x**3 - 840*x
        H12_p = 12*x**11 - 660*x**9 + 9240*x**7 - 41580*x**5 + 55440*x**3 - 15120*x
        
        P_prime = a0 * H0_p + a4 * H4_p + a8 * H8_p + c12 * H12_p
        P_val = a0 * H0 + a4 * H4 + a8 * H8 + c12 * H12
        
        return np.exp(-x**2 / 2.0) * (P_prime - x * P_val)
    
    # Use numerical integration
    from scipy.integrate import quad
    
    try:
        # ∫ |g(x)|^2 dx
        integrand_g2 = lambda x: g(np.array([x]))[0]**2
        I_g2, _ = quad(integrand_g2, -10, 10, limit=100)
        
        # ∫ x^2 |g(x)|^2 dx
        integrand_x2g2 = lambda x: x**2 * g(np.array([x]))[0]**2
        I_x2g2, _ = quad(integrand_x2g2, -10, 10, limit=100)
        
        # ∫ |g'(x)|^2 dx
        integrand_gp2 = lambda x: g_prime(np.array([x]))[0]**2
        I_gp2, _ = quad(integrand_gp2, -10, 10, limit=100)
        
        if I_g2 <= 0 or not np.isfinite(I_g2) or not np.isfinite(I_x2g2) or not np.isfinite(I_gp2):
            return None
        
        # The product A(f)*A(F) = <x^2> * <k^2>
        # <x^2> = I_x2g2 / I_g2
        # <k^2> = I_gp2 / I_g2
        # Product = I_x2g2 * I_gp2 / I_g2^2
        
        product = I_x2g2 * I_gp2 / (I_g2 ** 2)
        
        if not np.isfinite(product) or product < 0:
            return None
        
        return product
    
    except Exception:
        return None

def construct():
    """
    Find coefficients (a0, a4, a8) that minimize the upper bound on C4.
    Returns np.array([a0, a4, a8]).
    """
    # Start with the prior best
    best_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_score = _compute_bound(best_coeffs[0], best_coeffs[1], best_coeffs[2])
    
    if best_score is None:
        best_score = np.inf
    
    # Try local optimization
    if _time_remaining() > 30.0:
        def objective(x):
            result = _compute_bound(x[0], x[1], x[2])
            if result is None:
                return 1e10
            return result
        
        # Try multiple starting points
        start_points = [
            best_coeffs,
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, -1.0, 0.0]),
            np.array([0.0, 0.0, -1.0]),
            np.array([-0.5, 0.1, 0.001]),
            np.array([-1.2, 0.05, 0.0005]),
            np.array([-1.0, 0.04, 0.0003]),
        ]
        
        for sp in start_points:
            if _time_remaining() < 15.0:
                break
            try:
                res = minimize(objective, sp, method='Nelder-Mead', 
                              options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                if res.success or res.fun < best_score:
                    score = res.fun
                    if np.isfinite(score) and score < best_score:
                        best_score = score
                        best_coeffs = res.x
            except Exception:
                pass
        
        # Also try BFGS with gradient approximation
        if _time_remaining() > 20.0:
            for sp in start_points[:3]:
                if _time_remaining() < 10.0:
                    break
                try:
                    res = minimize(objective, sp, method='L-BFGS-B',
                                  options={'maxiter': 200, 'gtol': 1e-8})
                    if np.isfinite(res.fun) and res.fun < best_score:
                        best_score = res.fun
                        best_coeffs = res.x
                except Exception:
                    pass
    
    # Verify the best result
    final_score = _compute_bound(best_coeffs[0], best_coeffs[1], best_coeffs[2])
    if final_score is not None and np.isfinite(final_score) and final_score < best_score:
        best_score = final_score
    
    return best_coeffs
# EVOLVE-BLOCK-END
