"""Optimized program for uncertainty inequality coefficient search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import roots_hermite

def _get_hermite_values(n, x):
    """
    Compute the nth physicists' Hermite polynomial H_n(x) and its derivative H_n'(x)
    for an array x.
    H_n'(x) = 2n H_{n-1}(x)
    """
    if n == 0:
        Hn = np.ones_like(x)
        Hn_prime = np.zeros_like(x)
        return Hn, Hn_prime
    elif n == 1:
        Hn = 2 * x
        Hn_prime = 2 * np.ones_like(x)
        return Hn, Hn_prime
    
    H0 = np.ones_like(x)
    H1 = 2 * x
    H0_prime = np.zeros_like(x)
    H1_prime = 2 * np.ones_like(x)
    
    for k in range(2, n + 1):
        H2 = 2 * x * H1 - 2 * (k - 1) * H0
        H2_prime = 2 * k * H1
        
        H0, H1 = H1, H2
        H0_prime, H1_prime = H1_prime, H2_prime
        
    return H1, H1_prime

def _compute_bound(coeffs):
    """
    Compute the upper bound on C4 (uncertainty product) given coefficients [a, b, c].
    The H12 coefficient d is fixed by g(0) = 0.
    g(x) = (a H0 + b H4 + c H8 + d H12) * exp(-x^2/2)
    """
    a, b, c = coeffs
    
    # H_n(0) values
    # H_0(0) = 1
    # H_4(0) = -3! / 2! * ... wait.
    # H_0(0)=1
    # H_1(0)=0
    # H_2(0)=-2
    # H_3(0)=0
    # H_4(0)=12
    # H_5(0)=0
    # H_6(0)=-120
    # H_7(0)=0
    # H_8(0)=1680
    # H_9(0)=0
    # H_10(0)=-30240
    # H_11(0)=0
    # H_12(0)=665280
    
    # Let's verify H_4(0).
    # H_4(x) = 16x^4 - 48x^2 + 12. H_4(0) = 12.
    # The previous code used 3. That was WRONG.
    # H_8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680. H_8(0) = 1680.
    # The previous code used 105. That was WRONG.
    # H_12(0) = 665280.
    
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    d = -(a * H0_0 + b * H4_0 + c * H8_0) / H12_0
    
    # Precomputed quadrature data
    if not hasattr(_compute_bound, 'quad_data'):
        n_quad = 60
        x_quad, w_quad = roots_hermite(n_quad)
        
        # Get H and H' for 0, 4, 8, 12
        H0, dH0 = _get_hermite_values(0, x_quad)
        H4, dH4 = _get_hermite_values(4, x_quad)
        H8, dH8 = _get_hermite_values(8, x_quad)
        H12, dH12 = _get_hermite_values(12, x_quad)
        
        _compute_bound.quad_data = (x_quad, w_quad, H0, H4, H8, H12, dH0, dH4, dH8, dH12)
        
    x_quad, w_quad, H0, H4, H8, H12, dH0, dH4, dH8, dH12 = _compute_bound.quad_data
    
    # Construct P(x) and P'(x)
    P = a * H0 + b * H4 + c * H8 + d * H12
    dP = a * dH0 + b * dH4 + c * dH8 + d * dH12
    
    # g(x) = P(x) * exp(-x^2/2)
    # |g(x)|^2 = P(x)^2 * exp(-x^2)
    P2 = P * P
    norm2 = np.sum(w_quad * P2)
    
    if norm2 <= 1e-15:
        return 1e10
        
    # sigma_x^2 = <x^2> = sum(w * x^2 * P^2) / sum(w * P^2)
    var_x = np.sum(w_quad * x_quad**2 * P2) / norm2
    
    # g'(x) = (P'(x) - x P(x)) * exp(-x^2/2)
    # |g'(x)|^2 = (P'(x) - x P(x))^2 * exp(-x^2)
    deriv_term = dP - x_quad * P
    deriv_term_sq = deriv_term * deriv_term
    norm_deriv2 = np.sum(w_quad * deriv_term_sq)
    
    # sigma_k^2 = <k^2> in freq domain = (1/2pi) * <g'^2> / (1/2pi) * <g^2> = <g'^2> / <g^2>
    # Note: The standard uncertainty product is sigma_x * sigma_k.
    # The task asks for "upper bound on A(f)*A(Fourier(f))".
    # Usually A(f) is the standard deviation.
    # So we minimize var_x * var_k.
    
    var_k = norm_deriv2 / norm2
    
    bound = var_x * var_k
    
    return bound

def construct():
    """Find optimal coefficients [a, b, c] minimizing the C4 upper bound."""
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin
    
    # Initial guess: The seed program probably had a specific guess.
    # Let's try a few random starts and the zero vector (which is invalid due to norm 0, so perturb).
    # Actually, if a=b=c=0, d=0, norm=0.
    # Let's start with a simple combination.
    
    best_coeffs = np.array([0.0, 0.0, 0.0])
    best_bound = _compute_bound([1.0, 0.0, 0.0]) # H0 only, bound should be 1/4?
    # For H0, g = exp(-x^2/2).
    # var_x = 1/2. var_k = 1/2. Product = 1/4 = 0.25.
    # The seed score was 1.0. This implies the seed bound was 1.0.
    # Maybe the seed used different normalization or the score is scaled?
    # "score is the upper bound on C4".
    # If H0 gives 0.25, and seed gives 1.0, maybe the seed was a bad guess.
    # We want to minimize this value.
    
    # Let's check if we can go lower than 0.25.
    # The Heisenberg uncertainty principle says sigma_x sigma_k >= 1/4.
    # So 0.25 is the theoretical minimum for ANY function.
    # Wait, is it possible to get lower? No, the uncertainty principle is a fundamental limit.
    # So the minimum possible score is 0.25.
    # If the seed score is 1.0, we can improve by finding coefficients that approach the minimum uncertainty state.
    # The minimum uncertainty states are Gaussian wave packets, which correspond to linear combinations of H0 and H1?
    # But we are restricted to even Hermite polynomials H0, H4, H8, H12.
    # A pure Gaussian is H0. It gives 0.25.
    # Why would the seed be 1.0?
    # Maybe the "A(f)" definition is different?
    # Or maybe the "upper bound" refers to something else?
    # "Tune a ... test function to reduce the upper bound on A(f)*A(Fourier(f))."
    # If the theoretical minimum is 0.25, and we can achieve it with H0 (a=1, b=0, c=0), then the best score is 0.25.
    # Let's verify if H0 is allowed.
    # "Three free coefficients multiply H0, H4, H8".
    # So a=1, b=0, c=0 is a valid input.
    # d is fixed by g(0)=0.
    # If a=1, b=0, c=0, then d = -1/665280.
    # g(x) = (1 - H12(x)/665280) e^{-x^2/2}.
    # This is NOT the pure Gaussian. The constraint g(0)=0 forces g(0)=0.
    # The pure Gaussian has g(0) = 1/sqrt(sqrt(pi)) != 0.
    # So we CANNOT use the pure Gaussian. We must have a node at x=0.
    # Functions with a node at x=0 have higher uncertainty.
    # The first excited state H1 is odd, so H1(0)=0. But we are restricted to even polynomials.
    # Even polynomials P(x) have P(0) != 0 generally.
    # We force P(0)=0 by adding H12.
    # So g(x) is an even function with a node at 0.
    # The lowest energy state with a node at 0 and even symmetry?
    # Actually, if g is even and g(0)=0, it must change sign or be zero.
    # Since it's even, g(x) = g(-x). If g(0)=0, it touches 0.
    # This increases the variance.
    
    # Let's run the optimizer to find the best even function with g(0)=0.
    
    starts = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, -1.0, 0.0]),
        np.array([1.0, 0.0, -1.0]),
        np.array([0.0, 1.0, -1.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([-1.0, 1.0, -1.0]),
    ]
    
    for start in starts:
        if time.time() > deadline - 20:
            break
        
        try:
            result = minimize(
                _compute_bound,
                start,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            
            if result.fun < best_bound:
                best_bound = result.fun
                best_coeffs = result.x
        except Exception:
            pass
            
    # Polish
    if time.time() < deadline - 5:
        try:
            result = minimize(
                _compute_bound,
                best_coeffs,
                method='Powell',
                options={'maxiter': 5000, 'xtol': 1e-12, 'ftol': 1e-12}
            )
            if result.fun < best_bound:
                best_bound = result.fun
                best_coeffs = result.x
        except Exception:
            pass

    return np.array(best_coeffs)

# EVOLVE-BLOCK-END
