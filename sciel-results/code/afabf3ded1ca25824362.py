import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite
import time


def construct():
    t_start = time.time()
    deadline = 190.0  # 20s safety margin from 210s

    # Hermite polynomial values at 0 for the constraint
    # H_n(0) = 0 for odd n, and for even n=2k: H_{2k}(0) = (-1)^k (2k)! / k!
    # Physicist's Hermite polynomials
    H0_0 = 1.0
    H4_0 = 12.0   # H_4(0) = 12
    H8_0 = 1680.0 # H_8(0) = 1680
    H12_0 = 3991680.0  # H_12(0) = 3991680

    def compute_a12(a0, a4, a8):
        """Compute a12 from constraint P(0) = 0"""
        return -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0

    # Precompute Hermite polynomials on a grid for fast evaluation
    # Use a moderate grid size for speed
    N_grid = 4096
    x = np.linspace(-15, 15, N_grid)
    dx = x[1] - x[0]
    
    # Precompute H0, H4, H8, H12 on the grid
    H0_grid = np.ones_like(x)
    H4_grid = eval_hermite(4, x)
    H8_grid = eval_hermite(8, x)
    H12_grid = eval_hermite(12, x)
    
    gauss = np.exp(-np.pi * x**2)
    
    # Effective basis vectors incorporating the g(0)=0 constraint
    # g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * gauss
    # a12 = -(a0 + 12*a4 + 1680*a8) / 3991680
    # g(x) = a0*(H0 - H12/3991680)*gauss + a4*(H4 - 12*H12/3991680)*gauss + a8*(H8 - 1680*H12/3991680)*gauss
    
    v0 = (H0_grid - H12_grid / H12_0) * gauss
    v4 = (H4_grid - 12.0 * H12_grid / H12_0) * gauss
    v8 = (H8_grid - 1680.0 * H12_grid / H12_0) * gauss

    # Precompute Fourier transforms of the basis vectors
    # Using FFT with proper scaling
    G0 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(v0))) * dx
    G4 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(v4))) * dx
    G8 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(v8))) * dx
    
    freqs = np.fft.fftshift(np.fft.fftfreq(N_grid, d=dx))

    def objective(params):
        if time.time() > deadline:
            return 1e10
            
        a0, a4, a8 = params
        
        # Compute g(x) using precomputed basis
        g = a0 * v0 + a4 * v4 + a8 * v8
        
        # Compute Fourier transform using precomputed basis
        g_hat = a0 * G0 + a4 * G4 + a8 * G8
        
        norm_g = np.max(np.abs(g))
        norm_g_hat = np.max(np.abs(g_hat))
        
        return norm_g * norm_g_hat

    # Initial guess from the seed program
    x0 = np.array([-113/100, 1/25, 1/3240])
    
    best_result = None
    best_score = np.inf
    
    # Try multiple starting points for robustness
    start_points = [
        x0,
        np.array([-1.0, 0.1, 0.01]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-0.5, 0.05, 0.005]),
        np.array([-113/100, 1/25, 1/3240] * 1.1),
        np.array([-113/100, 1/25, 1/3240] * 0.9),
    ]
    
    for sp in start_points:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                objective,
                sp,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()
                
        except Exception as e:
            continue
    
    # If no valid result, fall back to seed
    if best_result is None:
        return np.array([-113/100, 1/25, 1/3240])
    
    # Final polish with tighter tolerance if time permits
    if time.time() < deadline - 10:
        try:
            result = minimize(
                objective,
                best_result,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()
        except Exception:
            pass
    
    return best_result
