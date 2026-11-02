import numpy as np
import scipy.optimize as opt
import scipy.integrate as integrate
import time

def construct():
    start_time = time.time()
    deadline = 170.0
    
    def H0(x):
        return np.ones_like(x)
    
    def H4(x):
        return x**4 - 6*x**2 + 3
    
    def H8(x):
        return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    
    def H12(x):
        return x**12 - 66*x**10 + 1260*x**8 - 9240*x**6 + 23100*x**4 - 27720*x**2 + 10395
    
    def g(x, c0, c4, c8):
        c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
        return c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    
    def f_of_x(x, c0, c4, c8):
        return g(x, c0, c4, c8) * np.exp(-x**2 / 2.0)
    
    # Precompute grid for FFT
    x_grid = np.linspace(-15, 15, 8192)
    dx = x_grid[1] - x_grid[0]
    n = len(x_grid)
    
    def compute_A_f(c0, c4, c8):
        f_vals = f_of_x(x_grid, c0, c4, c8)
        return np.trapz(np.abs(f_vals), x_grid)
    
    def compute_A_F(c0, c4, c8):
        f_vals = f_of_x(x_grid, c0, c4, c8)
        # FFT: F[k] = dx * sum_j f[j] exp(-2πijk/n)
        F_vals = np.fft.fft(f_vals) * dx
        F_centered = np.fft.fftshift(F_vals)
        k_grid = np.fft.fftfreq(n, d=dx)
        k_centered = np.fft.fftshift(k_grid)
        dk = k_centered[1] - k_centered[0]
        return np.trapz(np.abs(F_centered), k_centered)
    
    def objective(coeffs):
        c0, c4, c8 = coeffs
        A_f = compute_A_f(c0, c4, c8)
        A_F = compute_A_F(c0, c4, c8)
        return A_f * A_F
    
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_score = objective(x0)
    best_coeffs = x0.copy()
    
    current_time = time.time()
    
    np.random.seed(42)
    for restart in range(50):
        if current_time > deadline - 10:
            break
        
        if restart == 0:
            x_try = x0.copy()
        else:
            scale = 0.1 / (1 + restart * 0.2)
            noise = np.random.normal(0, scale, 3)
            x_try = best_coeffs + noise
        
        try:
            res = opt.minimize(objective, x_try, method='Nelder-Mead',
                              options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 200, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
        
        current_time = time.time()
    
    return np.array(best_coeffs)
