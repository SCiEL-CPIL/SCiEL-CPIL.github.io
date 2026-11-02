"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import factorial2
import time

_DEADLINE = 205.0  # safety margin before 210s kill

def _hermite_coeffs(n):
    """
    Return coefficients of the physicist's Hermite polynomial H_n(x).
    H_n(x) = n! * sum_{k=0}^{floor(n/2)} (-1)^k (2x)^(n-2k) / (k! (n-2k)!)
    Returns list of length n+1 where index i is coeff of x^i.
    """
    coeffs = [0.0] * (n + 1)
    for k in range(n // 2 + 1):
        # term: (-1)^k * n! / (k! * (n-2k)!) * 2^(n-2k) * x^(n-2k)
        log_term = (np.log(factorial2(n, 2)) if n > 0 else 0)  # placeholder, we'll compute directly
        # Better: compute directly
        import math
        num = math.factorial(n)
        den = math.factorial(k) * math.factorial(n - 2 * k)
        coeff_val = ((-1.0) ** k) * num / den * (2.0 ** (n - 2 * k))
        coeffs[n - 2 * k] = coeff_val
    return coeffs

def _poly_mul(a, b):
    """Multiply two polynomials represented as coefficient lists (low to high degree)."""
    result = [0.0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            result[i + j] += ai * bj
    return result

def _poly_add(a, b):
    """Add two polynomials."""
    n = max(len(a), len(b))
    result = [0.0] * n
    for i in range(n):
        result[i] = (a[i] if i < len(a) else 0.0) + (b[i] if i < len(b) else 0.0)
    return result

def _poly_scale(a, s):
    return [s * x for x in a]

def _compute_C4_bound(c0, c4, c8):
    """
    Compute the upper bound on C4 = A(f) * A(F) for the Hermite-Gaussian test function.
    
    The test function is g(x) = exp(-x^2/2) * (c0*H0 + c4*H4 + c8*H8 + c12*H12)
    where c12 is chosen so that g(0) = 0.
    
    H0(0)=1, H4(0)=0, H8(0)=0, H12(0)=0 (for even n>0, H_n(0)=0)
    Wait: H_n(0) for even n: H0(0)=1, H2(0)=-2, H4(0)=12, H6(0)=-120, H8(0)=1680, H10(0)=-30240, H12(0)=665280
    
    Actually let me reconsider. The physicist's Hermite polynomials:
    H0(x) = 1
    H1(x) = 2x
    H2(x) = 4x^2 - 2
    H3(x) = 8x^3 - 12x
    H4(x) = 16x^4 - 48x^2 + 12
    So H4(0) = 12, H8(0) = 1680, H12(0) = 665280
    
    So g(0) = c0*1 + c4*12 + c8*1680 + c12*665280 = 0
    => c12 = -(c0 + 12*c4 + 1680*c8) / 665280
    """
    c12 = -(c0 + 12.0 * c4 + 1680.0 * c8) / 665280.0
    
    # Build the polynomial P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    H0 = _hermite_coeffs(0)
    H4 = _hermite_coeffs(4)
    H8 = _hermite_coeffs(8)
    H12 = _hermite_coeffs(12)
    
    P = _poly_add(
        _poly_add(_poly_add(_poly_scale(H0, c0), _poly_scale(H4, c4)),
                  _poly_scale(H8, c8)),
        _poly_scale(H12, c12)
    )
    
    # g(x) = exp(-x^2/2) * P(x)
    # |g(x)|^2 = exp(-x^2) * P(x)^2
    # A(g) = (∫ |g(x)|^2 dx)^(-1/2) * (∫ x^2 |g(x)|^2 dx)^(1/2)
    
    # Q(x) = P(x)^2, a polynomial of degree up to 24
    Q = _poly_mul(P, P)
    
    # ∫ x^m exp(-x^2) dx from -inf to inf
    # For even m=2k: ∫ x^(2k) exp(-x^2) dx = sqrt(pi) * (2k-1)!! / 2^k = sqrt(pi) * (2k)! / (4^k * k!)
    # For odd m: 0
    
    def integral_m(m):
        """Compute ∫_{-inf}^{inf} x^m exp(-x^2) dx"""
        if m % 2 != 0:
            return 0.0
        k = m // 2
        # (2k)! / (4^k * k!) * sqrt(pi)
        import math
        if k == 0:
            return np.sqrt(np.pi)
        return np.sqrt(np.pi) * math.factorial(2 * k) / (4.0 ** k * math.factorial(k))
    
    # I0 = ∫ Q(x) exp(-x^2) dx = sum_i Q[i] * integral_m(i)
    I0 = sum(Q[i] * integral_m(i) for i in range(len(Q)))
    
    # I2 = ∫ x^2 Q(x) exp(-x^2) dx = sum_i Q[i] * integral_m(i+2)
    I2 = sum(Q[i] * integral_m(i + 2) for i in range(len(Q)))
    
    if I0 <= 0:
        return np.inf
    
    A_g = np.sqrt(I2 / I0)
    
    # Fourier transform: F(g)(ξ) = ∫ g(x) exp(-2πi x ξ) dx
    # For g(x) = exp(-x^2/2) P(x), the Fourier transform is also a Gaussian times a polynomial.
    # Actually, the Fourier transform of exp(-x^2/2) is sqrt(2π) exp(-π^2 ξ^2) ... 
    # Wait, let's use the convention. The problem says Fourier(f), let's assume standard.
    
    # Actually, for the uncertainty principle, the exact FT convention matters for the constant.
    # Let's use: F(ξ) = ∫ f(x) e^{-2πi x ξ} dx
    # FT of exp(-x^2/2) = sqrt(2π) exp(-2π^2 ξ^2)
    # This is getting complex. Let me use a different approach.
    
    # For a function g(x) = exp(-x^2/2) P(x) where P is a polynomial,
    # the Fourier transform can be computed using the fact that multiplication by x 
    # corresponds to i/(2π) d/dξ in the frequency domain (for the e^{-2πixξ} convention).
    
    # Alternatively, since we're dealing with Hermite polynomials and Gaussians,
    # the FT of exp(-x^2/2) H_n(x) is related to exp(-2π^2 ξ^2) times a polynomial in ξ.
    
    # Let me use a numerical approach for the FT part, or better, use the known result.
    
    # Actually, let's use the fact that for the standard uncertainty principle,
    # A(f) * A(F) ≥ 1/(4π) for the convention F(ξ) = ∫ f(x) e^{-2πixξ} dx.
    
    # Let me compute A(F) numerically or use the polynomial structure.
    
    # The FT of exp(-x^2/2) * x^n can be expressed in terms of derivatives of the FT of exp(-x^2/2).
    
    # Let me just compute it numerically with a careful grid.
    
    # Actually, for a polynomial times Gaussian, we can compute the FT analytically.
    # FT{exp(-x^2/2) x^n}(ξ) = (i/(2π))^n d^n/dξ^n [sqrt(2π) exp(-2π^2 ξ^2)]
    
    # This is complex. Let me use a different strategy: compute the moments in frequency domain.
    
    # For g(x) = exp(-x^2/2) P(x), the FT is:
    # ĝ(ξ) = sqrt(2π) exp(-2π^2 ξ^2) * Q(ξ) where Q is some polynomial.
    
    # The relationship: if g(x) = e^{-x^2/2} P(x), then
    # ĝ(ξ) = e^{-2π^2 ξ^2} * [some polynomial in ξ]
    
    # Specifically, using the generating function or direct computation:
    # The FT of x^n e^{-x^2/2} is (i/(2π))^n ∂^n/∂ξ^n [√(2π) e^{-2π^2 ξ^2}]
    
    # Let me just do it numerically with a fine grid.
    
    # Grid for frequency domain
    xi_max = 5.0
    n_points = 2048
    xi = np.linspace(-xi_max, xi_max, n_points)
    dx = xi[1] - xi[0]
    
    # Compute g(x) on a spatial grid
    x_max = 10.0
    n_x = 4096
    x = np.linspace(-x_max, x_max, n_x)
    dx_x = x[1] - x[0]
    
    # Evaluate P(x)
    P_x = np.zeros_like(x)
    for i, coeff in enumerate(P):
        P_x += coeff * x ** i
    
    g_x = np.exp(-x ** 2 / 2) * P_x
    
    # FT using FFT
    # We need to be careful with the FFT convention
    G = np.fft.fft(g_x * dx_x)
    G = np.fft.fftshift(G)
    
    # Frequency axis
    freq = np.fft.fftshift(np.fft.fftfreq(n_x, d=dx_x))
    
    # |G(ξ)|^2
    absG2 = np.abs(G) ** 2
    
    # A(F) = (∫ |F(ξ)|^2 dξ)^(-1/2) * (∫ ξ^2 |F(ξ)|^2 dξ)^(1/2)
    I0_f = np.sum(absG2) * dx
    I2_f = np.sum(xi ** 2 * absG2) * dx
    
    if I0_f <= 0:
        return np.inf
    
    A_F = np.sqrt(I2_f / I0_f)
    
    return A_g * A_F

def construct():
    """
    Find three coefficients (c0, c4, c8) that minimize the C4 upper bound.
    Uses coordinate descent with analytical subproblems.
    """
    start_time = time.time()
    
    # Initial guess from the prior state of the art
    best_c = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    best_val = _compute_C4_bound(best_c[0], best_c[1], best_c[2])
    
    # Coordinate descent
    tol = 1e-10
    max_iters = 500
    
    for outer in range(max_iters):
        if time.time() - start_time > _DEADLINE - 10:
            break
        
        improved = False
        for idx in range(3):
            if time.time() - start_time > _DEADLINE - 10:
                break
            
            # Try to optimize coefficient idx while keeping others fixed
            # Use a local search around the current value
            c_trial = best_c.copy()
            
            # Golden section search or simple grid search for the 1D subproblem
            lo = best_c[idx] - 1.0
            hi = best_c[idx] + 1.0
            
            # First, find a bracket
            f_mid = _compute_C4_bound(*[best_c[j] if j != idx else (lo + hi) / 2 for j in range(3)])
            
            # Simple ternary search
            for _ in range(30):
                if time.time() - start_time > _DEADLINE - 10:
                    break
                m1 = lo + (hi - lo) / 3
                m2 = hi - (hi - lo) / 3
                f1 = _compute_C4_bound(*[best_c[j] if j != idx else m1 for j in range(3)])
                f2 = _compute_C4_bound(*[best_c[j] if j != idx else m2 for j in range(3)])
                if f1 < f2:
                    hi = m2
                else:
                    lo = m1
            
            optimal_val = (lo + hi) / 2
            f_opt = _compute_C4_bound(*[best_c[j] if j != idx else optimal_val for j in range(3)])
            
            if f_opt < best_val - tol:
                best_c[idx] = optimal_val
                best_val = f_opt
                improved = True
        
        if not improved:
            break
    
    return best_c
# EVOLVE-BLOCK-END
