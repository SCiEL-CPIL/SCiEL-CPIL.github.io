"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution
from scipy.integrate import quad

_START_TIME = time.time()
_DEADLINE = 200.0  # 10s margin before 210s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _compute_bound_fast(a0, a4, a8):
    """Compute the uncertainty bound efficiently."""
    # a12 determined by g(0) = 0
    a12 = -(a0 + 12*a4 + 168*a8) / 20901120.0
    
    # Coefficients for P(x) = c12*x^12 + c10*x^10 + c8*x^8 + c6*x^6 + c4*x^4 + c2*x^2 + c0
    c0 = 0.0  # by construction
    c2 = -48*a4 - 3840*a8 - 51609600*a12
    c4 = 16*a4 + 24192*a8 + 41287680*a12
    c6 = -4480*a8 - 13762560*a12
    c8 = 256*a8 + 2129920*a12
    c10 = -133120*a12
    c12 = 4096*a12
    
    # Compute ||g||_1 = ∫ |P(x)| * exp(-x^2/2) dx
    # Use Gauss-Hermite quadrature for efficiency
    # ∫ f(x) exp(-x^2) dx ≈ sum w_i f(x_i) for appropriate weights
    # But we have exp(-x^2/2), so let's use a change of variables or direct quadrature
    
    # For speed, use a fixed grid with trapezoidal rule
    x = np.linspace(-10, 10, 2001)
    p_vals = c12*x**12 + c10*x**10 + c8*x**8 + c6*x**6 + c4*x**4 + c2*x**2 + c0
    g_vals = np.abs(p_vals) * np.exp(-x*x/2.0)
    l1_val = np.trapz(g_vals, x)
    
    # For Fourier transform, use the fact that for even functions:
    # F(g)(k) = 2 * ∫_0^∞ g(x) cos(kx) dx
    # We can compute this using FFT on a symmetric grid
    x_fft = np.linspace(-12, 12, 4096)
    dx = x_fft[1] - x_fft[0]
    p_fft = c12*x_fft**12 + c10*x_fft**10 + c8*x_fft**8 + c6*x_fft**6 + c4*x_fft**4 + c2*x_fft**2 + c0
    g_fft = p_fft * np.exp(-x_fft*x_fft/2.0)
    
    # FFT
    ft_vals = np.fft.fftshift(np.fft.fft(g_fft)) * dx
    ft_abs = np.abs(ft_vals)
    k_grid = np.fft.fftshift(np.fft.fftfreq(len(x_fft), d=dx)) * 2 * np.pi
    dk = k_grid[1] - k_grid[0]
    l1_ft = np.sum(ft_abs) * dk
    
    bound = l1_val * l1_ft
    return bound

def _objective_normalized(theta, phi, r):
    """Objective function in spherical coordinates to remove scale ambiguity."""
    # Convert spherical to Cartesian
    a0 = r * np.sin(phi) * np.cos(theta)
    a4 = r * np.sin(phi) * np.sin(theta)
    a8 = r * np.cos(phi)
    
    if _time_left() < 5:
        return 1e10
    
    try:
        return _compute_bound_fast(a0, a4, a8)
    except Exception:
        return 1e10

def construct():
    """Optimize the three coefficients using Differential Evolution in spherical coordinates."""
    # Initial guess from the seed
    a0_init = -113.0/100
    a4_init = 1.0/25
    a8_init = 1.0/3240
    
    # Convert to spherical coordinates
    r_init = np.sqrt(a0_init**2 + a4_init**2 + a8_init**2)
    if r_init < 1e-15:
        r_init = 1.0
    phi_init = np.arccos(np.clip(a8_init / r_init, -1, 1))
    theta_init = np.arctan2(a4_init, a0_init)
    
    best_coeffs = np.array([a0_init, a4_init, a8_init])
    best_bound = _compute_bound_fast(a0_init, a4_init, a8_init)
    
    # Bounds for spherical coordinates
    # r: positive, phi: [0, pi], theta: [0, 2*pi]
    # Use reasonable bounds based on initial guess
    r_bounds = (0.1, 10.0)
    phi_bounds = (0.01, np.pi - 0.01)
    theta_bounds = (0.0, 2 * np.pi)
    
    # Use Differential Evolution
    if _time_left() > 30:
        try:
            def objective_spherical(x):
                theta, phi, r = x
                if _time_left() < 5:
                    return 1e10
                a0 = r * np.sin(phi) * np.cos(theta)
                a4 = r * np.sin(phi) * np.sin(theta)
                a8 = r * np.cos(phi)
                try:
                    return _compute_bound_fast(a0, a4, a8)
                except Exception:
                    return 1e10
            
            result = differential_evolution(
                objective_spherical,
                bounds=[theta_bounds, phi_bounds, r_bounds],
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                atol=1e-8,
                init='sobol',
                polish=True
            )
            
            if result.fun < best_bound:
                theta_opt, phi_opt, r_opt = result.x
                a0_opt = r_opt * np.sin(phi_opt) * np.cos(theta_opt)
                a4_opt = r_opt * np.sin(phi_opt) * np.sin(theta_opt)
                a8_opt = r_opt * np.cos(phi_opt)
                best_bound = result.fun
                best_coeffs = np.array([a0_opt, a4_opt, a8_opt])
        except Exception:
            pass
    
    # Local refinement with Nelder-Mead if time allows
    if _time_left() > 15:
        try:
            from scipy.optimize import minimize
            def objective_cartesian(x):
                if _time_left() < 3:
                    return 1e10
                try:
                    return _compute_bound_fast(x[0], x[1], x[2])
                except Exception:
                    return 1e10
            
            result2 = minimize(objective_cartesian, best_coeffs, method='Nelder-Mead',
                             options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10,
                                    'adaptive': True})
            if result2.fun < best_bound:
                best_bound = result2.fun
                best_coeffs = result2.x
        except Exception:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
