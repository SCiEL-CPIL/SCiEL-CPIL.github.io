"""Optimized program for the uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_hermite


def _hermite_h(n, x):
    """Compute the physicist's Hermite polynomial H_n(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x
    h_nm2 = np.ones_like(x)
    h_nm1 = 2.0 * x
    for k in range(2, n + 1):
        h_n = 2.0 * x * h_nm1 - 2.0 * (k - 1) * h_nm2
        h_nm2 = h_nm1
        h_nm1 = h_n
    return h_nm1


def _gaussian(x):
    """Standard Gaussian exp(-x^2/2)."""
    return np.exp(-0.5 * x * x)


def _compute_bound(c0, c4, c8):
    """
    Compute the upper bound on A(f)*A(Fhat(f)) for the test function
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    where c12 is chosen so that g(0) = 0.
    
    The uncertainty product bound is related to the ratio of norms.
    For a function g, A(g) is the L1 norm (or area under |g|).
    The Fourier transform of H_n * exp(-x^2/2) is (i^n) H_n * exp(-x^2/2) (up to normalization).
    
    We use the standard uncertainty principle bound:
    sigma_x * sigma_k >= 1/2 for normalized functions, but here we want
    the product of the L1 norms (or support measures).
    
    Actually, for the specific formulation in Goncalves et al. 2017,
    the bound on C4 involves the ratio of the L1 norm of g to the L1 norm of F(g).
    Since F(H_n * exp(-x^2/2)) = (i^n) H_n * exp(-x^2/2) (with appropriate normalization),
    the L1 norms are identical for the real part if we consider absolute values.
    
    Wait, the task says "upper bound on A(f)*A(Fourier(f))". 
    In the context of the uncertainty principle with Hermite functions, 
    the relevant quantity is often the product of the "widths" or the 
    uncertainty product. However, the seed returns specific coefficients.
    
    Let's look at the structure:
    g(x) = P(x) * exp(-x^2/2)
    F(g)(k) = (1/sqrt(2*pi)) * integral P(x) exp(-x^2/2) exp(-ikx) dx
    
    For Hermite polynomials, the Fourier transform of H_n(x) exp(-x^2/2) is 
    (i^n) H_n(x) exp(-x^2/2) (using the convention where the Gaussian is exp(-x^2/2)).
    Actually, the standard result is:
    Fourier transform of exp(-x^2/2) is exp(-k^2/2) (with 1/sqrt(2*pi) factor).
    The Hermite functions psi_n(x) = (2^n n! sqrt(pi))^{-1/2} H_n(x) exp(-x^2/2) are eigenfunctions.
    
    If we define g(x) = P(x) exp(-x^2/2), then 
    F(g)(k) = (1/sqrt(2*pi)) * sum c_n (i^n) H_n(k) exp(-k^2/2) * (normalization constants?)
    
    Actually, the integral of H_n(x) exp(-x^2/2) exp(-ikx) dx is related to H_n(k) exp(-k^2/2).
    Specifically, int H_n(x) exp(-x^2/2) exp(-ikx) dx = sqrt(2*pi) * (i)^n H_n(k) exp(-k^2/2).
    So F(g)(k) = (1/sqrt(2*pi)) * sum c_n (i)^n H_n(k) exp(-k^2/2) = (sum c_n (i)^n H_n(k)) exp(-k^2/2) * (1/sqrt(2*pi)).
    
    Thus, |F(g)(k)| = |Q(k)| exp(-k^2/2) / sqrt(2*pi) where Q(k) = sum c_n (i)^n H_n(k).
    And |g(x)| = |P(x)| exp(-x^2/2) where P(x) = sum c_n H_n(x).
    
    The "area" A(g) = int |g(x)| dx = int |P(x)| exp(-x^2/2) dx.
    The "area" A(F(g)) = int |F(g)(k)| dk = (1/sqrt(2*pi)) int |Q(k)| exp(-k^2/2) dk.
    
    The product is A(g) * A(F(g)) = (1/sqrt(2*pi)) * int |P(x)| exp(-x^2/2) dx * int |Q(k)| exp(-k^2/2) dk.
    
    We want to minimize this product.
    
    The coefficients are c0, c4, c8, and c12 is fixed by P(0)=0.
    P(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0.
    H0(0) = 1
    H4(0) = -12
    H8(0) = 1008
    H12(0) = -103950 (Let's verify: H_n(0) = (-1)^(n/2) (n-1)!! for even n.
    H0(0)=1
    H2(0)=-1
    H4(0)=-3*1 = -12? No, (4-1)!! = 3*1 = 3. (-1)^2 * 3 = 3? 
    Let's use the recurrence or standard values.
    H0(x)=1, H0(0)=1
    H1(x)=2x, H1(0)=0
    H2(x)=4x^2-2, H2(0)=-2
    H3(x)=8x^3-12x, H3(0)=0
    H4(x)=16x^4-48x^2+12, H4(0)=12
    H5(x)=32x^5-160x^3+120x, H5(0)=0
    H6(x)=64x^6-480x^4+720x^2-120, H6(0)=-120
    H7(x)=128x^7-1344x^5+3360x^3-1680x, H7(0)=0
    H8(x)=256x^8-4480x^6+23040x^4-26880x^2+5040, H8(0)=5040
    H9(x)=512x^9-14336x^7+107520x^5-295680x^3+181440x, H9(0)=0
    H10(x)=1024x^10-45056x^8+645120x^6-3225600x^4+4989600x^2-103950, H10(0)=-103950? Wait.
    Let's check H4(0) again. H4(x) = 16x^4 - 48x^2 + 12. H4(0) = 12.
    H8(0) = 5040.
    H12(0) = ?
    Formula: H_{2n}(0) = (-1)^n (2n)! / n!.
    H0(0) = 1! / 0! = 1. (-1)^0 * 1 = 1.
    H2(0) = - 2! / 1! = -2. Correct.
    H4(0) = 4! / 2! = 24 / 2 = 12. Correct.
    H6(0) = - 6! / 3! = -720 / 6 = -120. Correct.
    H8(0) = 8! / 4! = 40320 / 24 = 1680? 
    Wait, 8! = 40320. 4! = 24. 40320 / 24 = 1680.
    But my recurrence gave 5040. Let me re-check the recurrence.
    H_{n+1} = 2x H_n - 2n H_{n-1}.
    H0 = 1
    H1 = 2x
    H2 = 2x(2x) - 2(1)(1) = 4x^2 - 2. H2(0) = -2.
    H3 = 2x(4x^2-2) - 2(2)(2x) = 8x^3 - 4x - 8x = 8x^3 - 12x. H3(0)=0.
    H4 = 2x(8x^3-12x) - 2(3)(4x^2-2) = 16x^4 - 24x^2 - 24x^2 + 12 = 16x^4 - 48x^2 + 12. H4(0)=12.
    H5 = 2x(16x^4-48x^2+12) - 2(4)(8x^3-12x) = 32x^5 - 96x^3 + 24x - 64x^3 + 96x = 32x^5 - 160x^3 + 120x. H5(0)=0.
    H6 = 2x(32x^5-160x^3+120x) - 2(5)(16x^4-48x^2+12) = 64x^6 - 320x^4 + 240x^2 - 160x^4 + 480x^2 - 120 = 64x^6 - 480x^4 + 720x^2 - 120. H6(0)=-120.
    H7 = 2x(64x^6-480x^4+720x^2-120) - 2(6)(32x^5-160x^3+120x) = 128x^7 - 960x^5 + 1440x^3 - 240x - 384x^5 + 1920x^3 - 1440x = 128x^7 - 1344x^5 + 3360x^3 - 1680x. H7(0)=0.
    H8 = 2x(128x^7-1344x^5+3360x^3-1680x) - 2(7)(64x^6-480x^4+720x^2-120)
       = 256x^8 - 2688x^6 + 6720x^4 - 3360x^2 - 896x^6 + 6720x^4 - 10080x^2 + 1680
       = 256x^8 - 3584x^6 + 13440x^4 - 13440x^2 + 1680.
    H8(0) = 1680.
    So H8(0) = 1680.
    
    H12(0) = 12! / 6! = 479001600 / 720 = 665280.
    Sign is (-1)^6 = +1. So H12(0) = 665280.
    
    So the constraint P(0)=0 is:
    c0 * 1 + c4 * 12 + c8 * 1680 + c12 * 665280 = 0
    c12 = -(c0 + 12*c4 + 1680*c8) / 665280
    
    Now, P(x) = c0 H0 + c4 H4 + c8 H8 + c12 H12.
    Q(k) = c0 H0(k) + c4 (i^4) H4(k) + c8 (i^8) H8(k) + c12 (i^12) H12(k)
         = c0 H0(k) + c4 H4(k) + c8 H8(k) + c12 H12(k)
         = P(k).
    
    So P(x) and Q(k) are the same polynomial!
    Therefore, A(g) = int |P(x)| exp(-x^2/2) dx
    A(F(g)) = (1/sqrt(2*pi)) int |P(k)| exp(-k^2/2) dk = (1/sqrt(2*pi)) A(g).
    
    So the product is A(g)^2 / sqrt(2*pi).
    Minimizing the product is equivalent to minimizing A(g) = int |P(x)| exp(-x^2/2) dx.
    
    So we need to minimize int |c0 H0(x) + c4 H4(x) + c8 H8(x) + c12 H12(x)| exp(-x^2/2) dx
    subject to c12 = -(c0 + 12*c4 + 1680*c8) / 665280.
    
    This is a 3-variable optimization problem.
    """
    c12 = -(c0 + 12.0 * c4 + 1680.0 * c8) / 665280.0
    
    # Use Gauss-Hermite quadrature for the integral
    # The weight is exp(-x^2), but we have exp(-x^2/2).
    # Standard Gauss-Hermite uses exp(-x^2).
    # We can transform: int f(x) exp(-x^2/2) dx = sqrt(2) int f(sqrt(2) t) exp(-t^2) dt.
    # Or just use numerical integration with a fine grid.
    
    x = np.linspace(-10, 10, 10000)
    h0 = _hermite_h(0, x)
    h4 = _hermite_h(4, x)
    h8 = _hermite_h(8, x)
    h12 = _hermite_h(12, x)
    
    p = c0 * h0 + c4 * h4 + c8 * h8 + c12 * h12
    integrand = np.abs(p) * np.exp(-0.5 * x * x)
    
    integral = np.trapz(integrand, x)
    
    return integral


def construct():
    # Optimize c0, c4, c8
    # Use differential evolution for global optimization
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    def objective(params):
        c0, c4, c8 = params
        return _compute_bound(c0, c4, c8)
    
    result = differential_evolution(objective, bounds, maxiter=100, tol=1e-8, polish=True, seed=42)
    
    c0, c4, c8 = result.x
    return np.array([c0, c4, c8])

# EVOLVE-BLOCK-END
