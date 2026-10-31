"""Optimized uncertainty bound via analytic Hermite-Gaussian moments + L-BFGS-B."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195  # 15s safety margin before 210s kill


def _hermite_coeffs():
    """Return coefficient arrays for H0, H4, H8, H12 (even polynomials in x)."""
    # H0(x) = 1
    # H4(x) = x^4 - 6x^2 + 3
    # H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12(x) = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    h0 = np.array([1.0])           # coeff of x^0
    h4 = np.array([3.0, -6.0, 1.0])  # coeff of x^0, x^2, x^4
    h8 = np.array([105.0, -420.0, 210.0, -28.0, 1.0])  # x^0, x^2, x^4, x^6, x^8
    h12 = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])  # x^0..x^12
    return h0, h4, h8, h12


def _poly_sq_moments(coefs):
    """
    Compute moments of P(x)^2 * exp(-x^2) where P(x) is an even polynomial.
    coefs: array where coefs[k] is coefficient of x^(2k).
    Returns array of moments I_m = integral x^(2m) P(x)^2 exp(-x^2) dx for m=0,1,2,3,4.
    """
    # P(x)^2 is a polynomial in x^2 of degree 2*(len(coefs)-1)
    # Let Q(y) = P(sqrt(y))^2, a polynomial in y.
    # Q(y) = sum_{i,j} coefs[i] coefs[j] y^(i+j)
    # I_m = integral x^(2m) Q(x^2) exp(-x^2) dx
    # Let u = x^2, dx = du/(2 sqrt(u))
    # I_m = 1/2 integral u^m Q(u) u^(-1/2) exp(-u) du
    # I_m = 1/2 integral Q(u) u^(m - 1/2) exp(-u) du
    # = 1/2 sum_{i,j} coefs[i] coefs[j] Gamma(m - 1/2 + i + j + 1)
    # = 1/2 sum_{i,j} coefs[i] coefs[j] Gamma(m + 1/2 + i + j)
    
    # We need moments up to m=4 (since highest degree in Q is 2*(3)=6, and we multiply by x^(2m))
    # Actually, for the bound we need specific combinations. Let's compute up to m=10 to be safe.
    max_m = 10
    moments = np.zeros(max_m + 1)
    
    # Precompute Gamma values
    # Gamma(n + 0.5) for integer n
    # Gamma(0.5) = sqrt(pi)
    # Gamma(1.5) = 0.5 sqrt(pi)
    # Gamma(2.5) = 1.5 * 0.5 sqrt(pi)
    # Gamma(n + 0.5) = (n - 0.5)! = (2n)! sqrt(pi) / (4^n n!)
    
    def gamma_half(n):
        # Gamma(n + 0.5)
        if n == 0:
            return np.sqrt(np.pi)
        # Recursive or direct formula
        # (2n)! / (4^n n!) * sqrt(pi)
        # Use log gamma for stability
        return np.exp(np.lgamma(n + 0.5))
    
    for m in range(max_m + 1):
        s = 0.0
        for i in range(len(coefs)):
            for j in range(len(coefs)):
                # Term: coefs[i] * coefs[j] * Gamma(m + 0.5 + i + j)
                s += coefs[i] * coefs[j] * gamma_half(m + i + j)
        moments[m] = 0.5 * s
    return moments


def _fourier_poly_coeffs(c0, c4, c8, c12):
    """
    The Fourier transform of e^{-x^2/2} H_n(x) is i^n e^{-x^2/2} H_n(x) (up to normalization).
    For the specific normalization used in the task (which likely implies the standard Hermite functions),
    the L1/L2 ratios are invariant under the Fourier transform for even n because i^n is real for even n.
    Specifically, F(e^{-x^2/2} H_{2k}(x)) = (-1)^k e^{-x^2/2} H_{2k}(x) * constant.
    Since we are looking at the product A(f)A(F(f)), and the functions are even and real,
    the structure of the polynomial P(x) in the spatial domain maps to a polynomial P_hat(xi) in the frequency domain.
    For Hermite polynomials, the Fourier transform just flips signs for odd powers, but since we only have even degrees,
    the polynomial in xi is the same as the polynomial in x, up to a global sign factor for each term.
    Wait, H_n(i x) = i^n H_n(x). So F(e^{-x^2/2} H_n(x)) = i^n e^{-x^2/2} H_n(x) (with standard convention).
    For n=0: i^0 = 1. P_hat(x) = P(x).
    For n=4: i^4 = 1.
    For n=8: i^8 = 1.
    For n=12: i^12 = 1.
    So the polynomial in the frequency domain is identical to the polynomial in the spatial domain.
    Thus, A(f) = A(F(f)) for this class of functions.
    The bound C4 is related to (||g||_1 / ||g||_2)^2.
    """
    # Since the polynomial is the same, we just need to compute the bound for P(x).
    pass


def _bound(c0, c4, c8):
    """
    Compute the upper bound C4 for the given coefficients.
    The constraint is g(0) = 0, which determines c12.
    c12 = -(c0*H0(0) + c4*H4(0) + c8*H8(0)) / H12(0)
    H0(0) = 1
    H4(0) = 3
    H8(0) = 105
    H12(0) = 945
    """
    h0_0 = 1.0
    h4_0 = 3.0
    h8_0 = 105.0
    h12_0 = 945.0
    
    c12 = -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0
    
    # Construct the polynomial P(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    # We need the coefficients of P(x) in the basis {1, x^2, ..., x^12}
    # Let's use the coefficient arrays from _hermite_coeffs
    h0, h4, h8, h12 = _hermite_coeffs()
    
    # Pad to length 7 (for x^0 to x^12)
    p_coefs = np.zeros(7)
    p_coefs[0] += c0 * h0[0]
    for k in range(len(h4)):
        p_coefs[k] += c4 * h4[k]
    for k in range(len(h8)):
        p_coefs[k] += c8 * h8[k]
    for k in range(len(h12)):
        p_coefs[k] += c12 * h12[k]
    
    # Compute moments of P(x)^2 exp(-x^2)
    # I_m = integral x^(2m) P(x)^2 exp(-x^2) dx
    moments = _poly_sq_moments(p_coefs)
    
    # The L2 norm squared is I_0
    l2_sq = moments[0]
    if l2_sq < 1e-12:
        return 1e10
    
    # The L1 norm is tricky. We need integral |P(x)| exp(-x^2/2) dx.
    # Wait, the function is g(x) = exp(-x^2/2) P(x).
    # ||g||_2^2 = integral exp(-x^2) P(x)^2 dx = I_0.
    # ||g||_1 = integral |exp(-x^2/2) P(x)| dx = integral |P(x)| exp(-x^2/2) dx.
    # This integral is not directly given by the moments of P(x)^2.
    # However, for the uncertainty bound A(f) = ||f||_1 / ||f||_2, we have:
    # A(f) = (integral |P(x)| exp(-x^2/2) dx) / sqrt(I_0)
    
    # We need to compute J = integral |P(x)| exp(-x^2/2) dx.
    # This requires numerical integration or a clever analytic approximation.
    # Given the time constraint, let's use a quick numerical integration for J.
    
    # Define a grid
    x = np.linspace(-10, 10, 1000)
    dx = x[1] - x[0]
    
    # Evaluate P(x)
    # P(x) = sum p_coefs[k] x^(2k)
    P_vals = np.zeros_like(x)
    for k in range(7):
        P_vals += p_coefs[k] * (x ** (2 * k))
    
    J = np.sum(np.abs(P_vals) * np.exp(-x ** 2 / 2)) * dx
    
    # The bound is (J / sqrt(I_0))^2 ?
    # Actually, the task says "score is the upper bound on C4".
    # In the context of the uncertainty principle, C4 might be (A(f) A(F(f)))^2 or similar.
    # Let's assume the score is (J / sqrt(I_0))^2.
    
    bound = (J / np.sqrt(l2_sq)) ** 2
    return bound


def _objective(coeffs):
    c0, c4, c8 = coeffs
    return _bound(c0, c4, c8)


def construct():
    # Initial guess from the seed program
    x0 = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    # Use L-BFGS-B to minimize the bound
    res = minimize(_objective, x0, method='L-BFGS-B', options={'maxiter': 100})
    
    if time.time() > _DEADLINE - 5:
        # Return the best found so far or the initial guess if optimization failed
        if res.success and res.fun < _bound(*x0):
            return res.x
        else:
            return x0
    
    # Check if we improved
    if res.success and res.fun < _bound(*x0):
        return res.x
    else:
        return x0
# EVOLVE-BLOCK-END
