"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hermite_poly(n, x):
    """Compute physicist's Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x, dtype=float)
    elif n == 1:
        return 2 * x
    elif n == 2:
        return 4 * x**2 - 2
    elif n == 3:
        return 8 * x**3 - 12 * x
    elif n == 4:
        return 16 * x**4 - 48 * x**2 + 12
    elif n == 5:
        return 32 * x**5 - 160 * x**3 + 120 * x
    elif n == 6:
        return 64 * x**6 - 480 * x**4 + 720 * x**2 - 120
    elif n == 7:
        return 128 * x**7 - 1344 * x**5 + 3360 * x**3 - 1680 * x
    elif n == 8:
        return 256 * x**8 - 3584 * x**6 + 13440 * x**4 - 13440 * x**2 + 1680
    elif n == 12:
        # H12(x) = 4096x^12 - 122880x^10 + 1843200x^8 - 12902400x^6 + 40819200x^4 - 54425280x^2 + 10321920
        x2 = x**2
        return (4096 * x2**6 - 122880 * x2**5 + 1843200 * x2**4 - 
                12902400 * x2**3 + 40819200 * x2**2 - 54425280 * x2 + 10321920)
    else:
        raise ValueError("Only support up to H12")


def _build_g(coeffs, x):
    """Build the test function g(x) = sum a_k H_k(x) * exp(-x^2/2)."""
    a0, a4, a8 = coeffs
    # H_k(0) values
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 10321920.0
    
    # a12 from constraint g(0) = 0: a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    H0 = _hermite_poly(0, x)
    H4 = _hermite_poly(4, x)
    H8 = _hermite_poly(8, x)
    H12 = _hermite_poly(12, x)
    
    gaussian = np.exp(-x**2 / 2)
    g = (a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12) * gaussian
    return g


def _fourier_transform(g, x, dx):
    """Compute Fourier transform via numerical integration."""
    # Use FFT-based approach
    n = len(x)
    # Scale factor for FFT
    G = np.fft.fftshift(np.fft.fft(g)) * dx
    return G


def _compute_objective(coeffs):
    """Compute the objective: upper bound on A(f)*A(Fourier(f))."""
    # Use a grid
    x_max = 10.0
    n = 4096
    x = np.linspace(-x_max, x_max, n)
    dx = x[1] - x[0]
    
    g = _build_g(coeffs, x)
    
    # Compute L1 norm of g
    A_g = np.sum(np.abs(g)) * dx
    
    # Compute Fourier transform
    G = _fourier_transform(g, x, dx)
    
    # Compute L1 norm of G
    A_G = np.sum(np.abs(G)) * dx
    
    # The product
    product = A_g * A_G
    
    return product


def construct():
    start_time = time.time()
    deadline = start_time + 180.0  # 180 second limit
    
    # Initial guess from prior state of the art
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    # Evaluate initial
    f0 = _compute_objective(x0)
    
    best_coeffs = x0.copy()
    best_val = f0
    
    # Try multiple optimization methods and initial points
    methods = ['Nelder-Mead', 'Powell', 'BFGS']
    
    # Generate diverse initial points
    rng = np.random.default_rng(42)
    initial_points = [x0]
    for _ in range(20):
        perturbation = rng.normal(0, 0.1, size=3)
        initial_points.append(x0 + perturbation)
    
    for method in methods:
        for init in initial_points:
            if time.time() > deadline - 30.0:
                break
            
            try:
                if method == 'Nelder-Mead':
                    result = minimize(_compute_objective, init, method='Nelder-Mead',
                                     options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
                elif method == 'Powell':
                    result = minimize(_compute_objective, init, method='Powell',
                                     options={'maxiter': 500, 'xtol': 1e-10, 'ftol': 1e-10})
                elif method == 'BFGS':
                    result = minimize(_compute_objective, init, method='BFGS',
                                     options={'maxiter': 500, 'gtol': 1e-10})
                else:
                    continue
                
                if result.success and result.fun < best_val:
                    best_val = result.fun
                    best_coeffs = result.x.copy()
            except Exception:
                pass
        
        if time.time() > deadline - 30.0:
            break
    
    # Final high-precision polish
    if time.time() < deadline - 10.0:
        try:
            result = minimize(_compute_objective, best_coeffs, method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
            if result.success and result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
