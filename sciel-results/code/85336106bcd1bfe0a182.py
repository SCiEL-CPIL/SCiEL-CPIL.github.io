"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hermite(n, x):
    """Compute physicists' Hermite polynomial Hn(x)."""
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


def _compute_c4(a0, a4, a8):
    """Compute the uncertainty product for given coefficients."""
    # Determine a12 so that g(0) = 0
    # g(0) = (a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0)) * exp(0)
    # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=945
    h0_0 = 1.0
    h4_0 = 3.0
    h8_0 = 105.0
    h12_0 = 945.0
    a12 = -(a0 * h0_0 + a4 * h4_0 + a8 * h8_0) / h12_0
    
    # Use numerical integration over a sufficiently large domain
    # g(x) = (a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)) * exp(-x^2/2)
    x = np.linspace(-15, 15, 20001)
    dx = x[1] - x[0]
    
    H0 = _hermite(0, x)
    H4 = _hermite(4, x)
    H8 = _hermite(8, x)
    H12 = _hermite(12, x)
    
    P = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
    g = P * np.exp(-x**2 / 2)
    
    # Compute norms
    norm_g_sq = np.sum(g**2) * dx
    if norm_g_sq <= 0:
        return 1e10
    
    xg = x * g
    norm_xg_sq = np.sum(xg**2) * dx
    
    # Fourier transform: G(xi) = (1/sqrt(2*pi)) * integral g(x) * exp(-i*xi*x) dx
    # For efficiency, use FFT-based approach or direct computation
    # Since g is even (Hermite polys of even degree times Gaussian), G is also even and real
    
    # Use the fact that for even functions, Fourier transform is real and even
    # G(xi) = (1/sqrt(2*pi)) * 2 * integral_0^inf g(x) * cos(xi*x) dx
    
    # Alternative: use the property that FT of Hn(x)*exp(-x^2/2) = (-i)^n * Hn(xi) * exp(-xi^2/2)
    # This is exact for physicists' Hermite polynomials with this Gaussian weight
    
    # So G(xi) = (1/sqrt(2*pi)) * (a0*H0(xi) + a4*(-i)^4*H4(xi) + a8*(-i)^8*H8(xi) + a12*(-i)^12*H12(xi)) * exp(-xi^2/2)
    # (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = 1
    # So G(xi) = (1/sqrt(2*pi)) * (a0*H0(xi) + a4*H4(xi) + a8*H8(xi) + a12*H12(xi)) * exp(-xi^2/2)
    
    xi = x  # same grid
    G = (1.0 / np.sqrt(2 * np.pi)) * P * np.exp(-xi**2 / 2)
    
    norm_G_sq = np.sum(G**2) * dx
    if norm_G_sq <= 0:
        return 1e10
    
    xiG = xi * G
    norm_xiG_sq = np.sum(xiG**2) * dx
    
    # Uncertainty product
    A_f = np.sqrt(norm_xg_sq / norm_g_sq)
    A_F = np.sqrt(norm_xiG_sq / norm_G_sq)
    
    return A_f * A_F


def construct():
    """Find coefficients that minimize the uncertainty product."""
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin
    
    # Initial guess from seed
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Try multiple starting points
    candidates = [x0.copy()]
    
    # Add some perturbed starting points
    np.random.seed(42)
    for i in range(20):
        candidates.append(x0 + 0.5 * np.random.randn(3))
    
    # Also try symmetric points
    candidates.append(np.array([1.0, 0.1, 0.01]))
    candidates.append(np.array([-1.0, 0.1, 0.01]))
    candidates.append(np.array([0.5, -0.1, 0.001]))
    candidates.append(np.array([-0.5, 0.1, -0.001]))
    
    best_x = x0.copy()
    best_val = _compute_c4(*x0)
    
    for x_init in candidates:
        if time.time() > deadline - 30.0:
            break
        
        try:
            result = minimize(
                _compute_c4,
                x_init,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            continue
    
    # Final polish with tighter tolerance if time permits
    if time.time() < deadline - 10.0:
        try:
            result = minimize(
                _compute_c4,
                best_x,
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'xatol': 1e-10,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    return np.array(best_x)
# EVOLVE-BLOCK-END
