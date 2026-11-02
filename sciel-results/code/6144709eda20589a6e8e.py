"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import herm


def _hermite_poly(n, x):
    """Compute physicist's Hermite polynomial H_n(x)."""
    return herm(n)(x)


def _g_and_fourier(a0, a4, a8, x):
    """
    Construct g(x) = P(x)*exp(-x^2/2) and its Fourier transform.
    P(x) = a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)
    with a12 chosen so that g(0) = 0, i.e., P(0) = 0.
    
    H0(0)=1, H4(0)=-12, H8(0)=1680, H12(0)=-102816
    So: a0 - 12*a4 + 1680*a8 - 102816*a12 = 0
    => a12 = (a0 - 12*a4 + 1680*a8) / 102816
    """
    a12 = (a0 - 12*a4 + 1680*a8) / 102816.0
    
    # Compute P(x)
    H0 = _hermite_poly(0, x)
    H4 = _hermite_poly(4, x)
    H8 = _hermite_poly(8, x)
    H12 = _hermite_poly(12, x)
    
    P = a0*H0 + a4*H4 + a8*H8 + a12*H12
    
    # g(x) = P(x) * exp(-x^2/2)
    g = P * np.exp(-x**2 / 2.0)
    
    # For Fourier transform of Hermite-Gaussian functions:
    # If g(x) = sum c_n H_n(x) exp(-x^2/2), then 
    # F(g)(k) = i^n * sum c_n H_n(k) exp(-k^2/2) (up to normalization)
    # More precisely, the Fourier transform of H_n(x)exp(-x^2/2) is i^n H_n(x)exp(-x^2/2)
    # with appropriate Fourier convention.
    
    # Using the convention F(f)(k) = integral f(x) exp(-ikx) dx
    # The Fourier transform of H_n(x) exp(-x^2/2) is i^n H_n(x) exp(-x^2/2) * sqrt(2*pi)
    # Actually, let me be more careful. The standard result is:
    # F[H_n(x) exp(-x^2/2)](k) = i^n H_n(k) exp(-k^2/2) * (2*pi)^{1/2}
    
    # So F(g)(k) = sqrt(2*pi) * [a0*H0(k) + a4*i^4*H4(k) + a8*i^8*H8(k) + a12*i^12*H12(k)] * exp(-k^2/2)
    # i^4 = 1, i^8 = 1, i^12 = i^4 = 1 (since i^4=1)
    # So F(g)(k) = sqrt(2*pi) * P(k) * exp(-k^2/2)
    
    # Wait, that's not right. Let me reconsider.
    # The Fourier transform of H_n(x) e^{-x^2/2} depends on the normalization.
    # With F(f)(k) = ∫ f(x) e^{-ikx} dx:
    # F[H_n(x) e^{-x^2/2}](k) = i^n √(2π) H_n(k) e^{-k^2/2}
    
    # i^0 = 1, i^4 = 1, i^8 = 1, i^12 = 1
    # So F(g)(k) = √(2π) * [a0*H0(k) + a4*H4(k) + a8*H8(k) + a12*H12(k)] * e^{-k^2/2}
    # = √(2π) * P(k) * e^{-k^2/2}
    
    # So |F(g)(k)| = √(2π) * |P(k)| * e^{-k^2/2}
    
    return g


def _compute_bound(a0, a4, a8):
    """
    Compute the upper bound on A(f)*A(Fourier(f)) for the test function.
    A(f) is likely the L1 norm: A(f) = ∫|f(x)|dx
    """
    # Create a fine grid
    x = np.linspace(-20, 20, 200001)
    dx = x[1] - x[0]
    
    # Compute g(x)
    g = _g_and_fourier(a0, a4, a8, x)
    
    # A(g) = ∫|g(x)|dx
    A_g = np.sum(np.abs(g)) * dx
    
    # Compute Fourier transform magnitude
    # F(g)(k) = √(2π) * P(k) * e^{-k^2/2}
    # So |F(g)(k)| = √(2π) * |P(k)| * e^{-k^2/2}
    
    # We need P(k) = a0*H0(k) + a4*H4(k) + a8*H8(k) + a12*H12(k)
    a12 = (a0 - 12*a4 + 1680*a8) / 102816.0
    
    H0_k = _hermite_poly(0, x)  # Same variable, just evaluating at k=x
    H4_k = _hermite_poly(4, x)
    H8_k = _hermite_poly(8, x)
    H12_k = _hermite_poly(12, x)
    
    P_k = a0*H0_k + a4*H4_k + a8*H8_k + a12*H12_k
    
    F_g_abs = np.sqrt(2*np.pi) * np.abs(P_k) * np.exp(-x**2 / 2.0)
    
    # A(F(g)) = ∫|F(g)(k)|dk
    A_Fg = np.sum(F_g_abs) * dx
    
    return A_g * A_Fg


def _objective(params):
    a0, a4, a8 = params
    return _compute_bound(a0, a4, a8)


def construct():
    # Start from the known good solution
    x0 = np.array([-113/100, 1/25, 1/3240])
    
    # Optimize
    result = minimize(_objective, x0, method='Nelder-Mead', 
                     options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
    
    if result.success:
        return result.x
    else:
        return x0
# EVOLVE-BLOCK-END

