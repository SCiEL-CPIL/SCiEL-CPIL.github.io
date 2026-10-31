# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import factorial
from math import sqrt, pi

def _hermite_polynomial_values(n, x):
    """Compute H_n(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    h0 = np.ones_like(x)
    h1 = 2 * x
    for k in range(2, n + 1):
        h2 = 2 * x * h1 - 2 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1

def _compute_bound(c0, c4, c8):
    """Compute the uncertainty upper bound for given coefficients."""
    # c12 derived from g(0) = 0
    # g(0) = (c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0)) * exp(0)
    # H_n(0) for even n: H_{2k}(0) = (-1)^k * (2k)! / k!
    H0_0 = 1.0
    H4_0 = -12.0  # (-1)^2 * 4!/2! = 24/2 = 12, but sign: H4(x)=16x^4-48x^2+12, H4(0)=12... 
    # Let me recalculate: H0=1, H2=4x^2-2, H4=16x^4-48x^2+12, H8=256x^8-3072x^6+11520x^4-13440x^2+560
    # H4(0) = 12, H8(0) = 560, H12(0) = ?
    # H_{2k}(0) = (-1)^k * (2k)! / k!
    # H4(0): k=2, (-1)^2 * 24/2 = 12
    # H8(0): k=4, (-1)^4 * 40320/24 = 1680... wait let me just compute numerically
    
    # Use recurrence to get H_n(0)
    def H_at_zero(n):
        if n == 0:
            return 1.0
        if n == 1:
            return 0.0
        h0, h1 = 1.0, 0.0
        for k in range(2, n + 1):
            h2 = 0.0 * h1 - 2 * (k - 1) * h0  # at x=0: 2*0*h1 - 2*(k-1)*h0
            h0, h1 = h1, h2
        return h1
    
    H0z = H_at_zero(0)
    H4z = H_at_zero(4)
    H8z = H_at_zero(8)
    H12z = H_at_zero(12)
    
    # g(0) = (c0*H0z + c4*H4z + c8*H8z + c12*H12z) * 1 = 0
    # c12 = -(c0*H0z + c4*H4z + c8*H8z) / H12z
    c12 = -(c0 * H0z + c4 * H4z + c8 * H8z) / H12z
    
    # Now compute the bound. The test function is g(x) = P(x) * exp(-x^2/4)
    # where P(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    # 
    # For the uncertainty principle bound, we need:
    # A(f) = sqrt(∫|f|² dx) and the spread, but the "upper bound on C4" 
    # in the context of the uncertainty inequality is typically:
    # C4 = (∫x²|f|² dx / ∫|f|² dx) * (∫ξ²|F|² dξ / ∫|F|² dξ)
    # 
    # But actually, for the Fourier uncertainty principle, the bound is:
    # Δx * Δξ ≥ 1/2 (standard), and the constant C4 might refer to a specific
    # formulation. Given the task says "upper bound on A(f)*A(Fourier(f))",
    # this is likely the product of L2 norms or the uncertainty product.
    #
    # Looking at the Goncalves et al. 2017 reference, this is about the 
    # Fourier uncertainty principle where you minimize the uncertainty product.
    # The "upper bound on C4" likely means the uncertainty product constant.
    #
    # Let me compute: for g(x) = P(x)exp(-x²/4), 
    # |g|² = P(x)² exp(-x²/2)
    # ∫|g|² dx = ∫ P(x)² exp(-x²/2) dx
    # ∫x²|g|² dx = ∫ x² P(x)² exp(-x²/2) dx
    #
    # For the Fourier transform: F(ξ) = ∫ g(x) e^{-2πiξx} dx
    # With the convention exp(-x²/4), the Fourier transform of exp(-x²/4) is 
    # sqrt(2π) * exp(-π²ξ²) ... actually let me use the standard math convention.
    #
    # Actually, for the Hermite-Gaussian basis with weight exp(-x²/4), the 
    # Fourier transform properties are cleaner. Let me use numerical integration
    # to be safe.
    
    # Use numerical quadrature
    x = np.linspace(-20, 20, 100001)
    dx = x[1] - x[0]
    
    # Compute P(x)
    H0 = np.ones_like(x)
    H4 = _hermite_polynomial_values(4, x)
    H8 = _hermite_polynomial_values(8, x)
    H12 = _hermite_polynomial_values(12, x)
    
    P = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    g = P * np.exp(-x**2 / 4)
    
    # |g|²
    g2 = g**2
    
    # Moments for position
    int_g2 = np.trapz(g2, x)
    int_x2_g2 = np.trapz(x**2 * g2, x)
    
    if int_g2 <= 0:
        return np.inf
    
    # Variance in x
    var_x = int_x2_g2 / int_g2  # since mean is 0 by symmetry
    
    # Fourier transform using FFT or direct computation
    # For efficiency, use the fact that FT of exp(-x²/4)*P(x) can be computed
    # But let's do it numerically with a fine grid
    # Use the convention F(ξ) = ∫ g(x) exp(-2πiξx) dx
    
    # For the uncertainty product, we need ∫|F(ξ)|² dξ and ∫ξ²|F(ξ)|² dξ
    # By Parseval: ∫|F(ξ)|² dξ = ∫|g(x)|² dx (with proper normalization)
    # With F(ξ) = ∫ g(x) e^{-2πiξx} dx, we have ∫|F|² dξ = ∫|g|² dx
    
    # For ∫ξ²|F(ξ)|² dξ, we can use: ξ²|F(ξ)|² corresponds to 
    # (1/(4π²)) * |g''(x)|² in some sense, or more precisely:
    # ∫ξ²|F(ξ)|² dξ = (1/(4π²)) ∫|g'(x)|² dx ... no that's not right.
    #
    # Actually: if F(ξ) = ∫ g(x) e^{-2πiξx} dx, then 
    # d/dξ F(ξ) = ∫ (-2πix)g(x) e^{-2πiξx} dx = -2πi F_x(ξ)
    # where F_x is FT of xg(x)
    # So ∫ξ²|F|² dξ = (1/(4π²)) ∫|xg(x)|² dx ... no.
    #
    # Let me just compute it directly. The key identity is:
    # ∫ ξ² |F(ξ)|² dξ = (1/(4π²)) ∫ |g'(x)|² dx  [for the 2π convention]
    # 
    # Wait, let me be careful. With F(ξ) = ∫ g(x) e^{-2πiξx} dx:
    # F'(ξ) = ∫ g(x) (-2πix) e^{-2πiξx} dx = -2πi ∫ xg(x) e^{-2πiξx} dx
    # So ∫ |F'(ξ)|² dξ = 4π² ∫ |xg(x)|² dx (by Parseval on F')
    # 
    # But we want ∫ ξ²|F(ξ)|² dξ, not ∫|F'|².
    # 
    # Use integration by parts / Parseval differently:
    # ∫ ξ²|F(ξ)|² dξ = ∫ ξ F(ξ) * ξ \bar{F(ξ)} dξ
    # Note that ξ F(ξ) = -1/(2πi) F'(ξ)
    # So ∫ ξ²|F|² dξ = -1/(2πi) ∫ F'(ξ) * ξ \bar{F(ξ)} dξ ... this gets messy.
    #
    # Simpler: just compute the Fourier transform numerically and get the moments.
    
    # Use FFT for efficiency
    N = 2**16  # 65536 points
    x_fft = np.linspace(-10, 10, N, endpoint=False)
    dx_fft = x_fft[1] - x_fft[0]
    
    H0_f = np.ones_like(x_fft)
    H4_f = _hermite_polynomial_values(4, x_fft)
    H8_f = _hermite_polynomial_values(8, x_fft)
    H12_f = _hermite_polynomial_values(12, x_fft)
    
    P_f = c0 * H0_f + c4 * H4_f + c8 * H8_f + c12 * H12_f
    g_f = P_f * np.exp(-x_fft**2 / 4)
    
    # FFT: F(ξ_k) ≈ dx * sum_j g(x_j) e^{-2πi k j / N}
    # The frequency spacing is dξ = 1/(N*dx)
    G = np.fft.fft(g_f) * dx_fft
    dxi = 1.0 / (N * dx_fft)
    xi = np.fft.fftfreq(N, d=dx_fft)
    
    G2 = np.abs(G)**2
    int_G2 = np.sum(G2) * dxi
    int_xi2_G2 = np.sum(xi**2 * G2) * dxi
    
    if int_G2 <= 0:
        return np.inf
    
    var_xi = int_xi2_G2 / int_G2
    
    # Uncertainty product
    # The "A(f)" likely refers to the spread or the L2 norm
    # Given "upper bound on A(f)*A(Fourier(f))", and the context of uncertainty:
    # This is likely sqrt(var_x) * sqrt(var_xi) or var_x * var_xi
    # 
    # In the standard uncertainty principle: Δx * Δξ ≥ 1/(4π)
    # where Δx = sqrt(<x²> - <x>²) and Δξ = sqrt(<ξ²> - <ξ>²)
    # 
    # The "upper bound on C4" where C4 = A(f)*A(F(f)) suggests we're computing
    # the uncertainty product and minimizing it.
    
    uncertainty_product = sqrt(var_x) * sqrt(var_xi)
    
    return uncertainty_product

def construct():
    deadline = time.time() + 180
    
    # Start with the seed
    best_coeffs = np.array([-113/100, 1/25, 1/3240])
    best_bound = _compute_bound(*best_coeffs)
    
    elapsed = time.time() - (deadline - 180)
    if elapsed > 160:
        return best_coeffs
    
    # Try local optimization using Nelder-Mead
    from scipy.optimize import minimize
    
    def objective(coeffs):
        c0, c4, c8 = coeffs
        return _compute_bound(c0, c4, c8)
    
    # Try a few different starting points
    start_points = [
        best_coeffs,
        np.array([-1.0, 0.05, 0.0003]),
        np.array([-1.2, 0.03, 0.0004]),
        np.array([-1.1, 0.045, 0.00035]),
        np.array([-1.15, 0.035, 0.00025]),
    ]
    
    for sp in start_points:
        if time.time() > deadline - 20:
            break
        try:
            res = minimize(objective, sp, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.success and res.fun < best_bound:
                best_bound = res.fun
                best_coeffs = res.x
        except:
            pass
    
    # Try a grid search in the neighborhood of the best found
    if time.time() < deadline - 30:
        c0_center, c4_center, c8_center = best_coeffs
        for dc0 in np.linspace(-0.05, 0.05, 5):
            for dc4 in np.linspace(-0.01, 0.01, 5):
                for dc8 in np.linspace(-0.0001, 0.0001, 5):
                    if time.time() > deadline - 20:
                        break
                    trial = np.array([c0_center + dc0, c4_center + dc4, c8_center + dc8])
                    b = _compute_bound(*trial)
                    if b < best_bound:
                        best_bound = b
                        best_coeffs = trial
    
    return best_coeffs
# EVOLVE-BLOCK-END
