import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_hermite
import time


def construct():
    t_start = time.time()
    deadline = 200.0  # Safety margin from 210s limit

    # Hermite polynomial values at 0 (physicists' convention)
    H0_0 = 1.0
    H4_0 = -12.0
    H8_0 = 1680.0
    H12_0 = -3991680.0

    def compute_a12(a0, a4, a8):
        """Compute a12 from constraint P(0) = 0"""
        return -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0

    # Precompute grids and Hermite values for speed
    # Use a fine grid for accurate sup-norm
    N_grid = 4096
    x_main = np.linspace(-8, 8, N_grid)
    dx_main = x_main[1] - x_main[0]
    
    # Precompute Hermite polynomials on the grid
    H0_main = eval_hermite(0, x_main)
    H4_main = eval_hermite(4, x_main)
    H8_main = eval_hermite(8, x_main)
    H12_main = eval_hermite(12, x_main)
    
    # Precompute Gaussian
    gaussian = np.exp(-np.pi * x_main**2)
    
    # Precompute FFT of the basis functions for fast FT evaluation
    # FT of H_n(x) * exp(-pi x^2)
    # We'll compute this numerically once for each basis
    fft_H0 = np.fft.fftshift(np.fft.fft(H0_main * gaussian)) * dx_main
    fft_H4 = np.fft.fftshift(np.fft.fft(H4_main * gaussian)) * dx_main
    fft_H8 = np.fft.fftshift(np.fft.fft(H8_main * gaussian)) * dx_main
    fft_H12 = np.fft.fftshift(np.fft.fft(H12_main * gaussian)) * dx_main
    
    def objective(params):
        a0, a4, a8 = params
        a12 = compute_a12(a0, a4, a8)
        
        # Compute g(x)
        P = a0 * H0_main + a4 * H4_main + a8 * H8_main + a12 * H12_main
        g = P * gaussian
        
        norm_g_inf = np.max(np.abs(g))
        norm_g_l2_sq = np.sum(g**2) * dx_main
        
        if norm_g_l2_sq < 1e-10:
            return 1e10
            
        # Compute Fourier Transform using precomputed basis
        G = a0 * fft_H0 + a4 * fft_H4 + a8 * fft_H8 + a12 * fft_H12
        norm_g_hat_inf = np.max(np.abs(G))
        
        # The quantity: sup(g) * sup(g_hat) / ||g||_2^2
        # This is scale-invariant and represents the bound constant
        return (norm_g_inf * norm_g_hat_inf) / norm_g_l2_sq

    # Initial guess from the seed program
    x0 = np.array([-113/100, 1/25, 1/3240])
    
    best_result = x0.copy()
    best_score = objective(x0)
    
    # Strategy: Use Nelder-Mead with multiple random restarts
    # The problem is 3D, so Nelder-Mead is efficient
    np.random.seed(42)
    
    # Generate diverse starting points
    start_points = [x0.copy()]
    
    # Add perturbations around x0
    for scale in [0.01, 0.05, 0.1, 0.5, 1.0]:
        for _ in range(3):
            perturbation = np.random.randn(3) * scale
            start_points.append(x0 + perturbation)
    
    # Add some structured points
    start_points.append(np.array([0.0, 0.0, 0.0]))
    start_points.append(np.array([-1.0, 0.1, 0.01]))
    start_points.append(np.array([-0.5, 0.05, 0.005]))
    start_points.append(np.array([-2.0, 0.2, 0.02]))
    start_points.append(np.array([-0.1, 0.01, 0.001]))
    start_points.append(np.array([-0.2, 0.02, 0.002]))
    start_points.append(np.array([-0.3, 0.03, 0.003]))
    start_points.append(np.array([-0.4, 0.04, 0.004]))
    
    # Evaluate all starting points and keep the best few
    scored_starts = []
    for sp in start_points:
        score = objective(sp)
        scored_starts.append((score, sp))
    
    scored_starts.sort(key=lambda x: x[0])
    
    # Take the top 10 starting points for local optimization
    top_starts = [sp for _, sp in scored_starts[:10]]
    
    for sp in top_starts:
        if time.time() > deadline - 20:
            break
        try:
            result = minimize(
                objective,
                sp,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()
        except Exception:
            continue
    
    # If we have time, try differential evolution for global search
    if time.time() < deadline - 40:
        try:
            bounds = [(-3, 3), (-1, 1), (-0.1, 0.1)]
            de_result = differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=30,
                popsize=15,
                tol=1e-8,
                polish=False,
                init='sobol'
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_result = de_result.x.copy()
                
                # Polish with Nelder-Mead
                if time.time() < deadline - 10:
                    polish_result = minimize(
                        objective,
                        best_result,
                        method='Nelder-Mead',
                        options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
                    )
                    if polish_result.fun < best_score:
                        best_score = polish_result.fun
                        best_result = polish_result.x.copy()
        except Exception:
            pass
    
    # Final polish if time allows
    if time.time() < deadline - 5:
        try:
            final_result = minimize(
                objective,
                best_result,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if final_result.fun < best_score:
                best_score = final_result.fun
                best_result = final_result.x.copy()
        except Exception:
            pass
    
    return best_result
