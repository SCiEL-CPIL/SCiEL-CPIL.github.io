"""Optimize Hermite-Gaussian coefficients to minimize the uncertainty bound C4."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_hermite

# Deadline guard: kill at 200s to stay under 210s wall-clock
_DEADLINE = time.time() + 200.0

def _check_deadline():
    if time.time() > _DEADLINE:
        raise TimeoutError()

def _hermite_coefficients(c0, c4, c8):
    """Compute c12 from the constraint g(0) = 0."""
    H0_0 = eval_hermite(0, 0)
    H4_0 = eval_hermite(4, 0)
    H8_0 = eval_hermite(8, 0)
    H12_0 = eval_hermite(12, 0)
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    return c0, c4, c8, c12

def _g_and_fourier(c0, c4, c8, x):
    """Evaluate g(x) and its Fourier transform on grid x."""
    c12 = _hermite_coefficients(c0, c4, c8)[3]
    H0 = eval_hermite(0, x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    gauss = np.exp(-x**2 / 2.0)
    g = (c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12) * gauss
    # Fourier transform of H_n(x)*exp(-x^2/2) is (-i)^n H_n(x)*exp(-x^2/2)
    # For the product, we use the fact that F[H_n * exp(-x^2/2)] = (-i)^n H_n * exp(-x^2/2)
    # But we need the L2-normalized Fourier transform. Let's use the standard convention.
    # Actually, for the uncertainty bound, we need ||g||_inf * ||ĝ||_inf
    # The Fourier transform of exp(-x^2/2) is sqrt(2*pi)*exp(-xi^2/2)
    # For Hermite functions, the Fourier transform has a specific phase.
    # Let's compute numerically using FFT for accuracy.
    return g

def _compute_bound(c0, c4, c8):
    """Compute the upper bound C4 = ||g||_inf * ||F(g)||_inf."""
    _check_deadline()
    c12 = _hermite_coefficients(c0, c4, c8)[3]
    
    # Use a fine grid for accurate sup-norm estimation
    x = np.linspace(-20, 20, 200001)
    H0 = eval_hermite(0, x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    gauss = np.exp(-x**2 / 2.0)
    g = (c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12) * gauss
    
    # Compute Fourier transform numerically
    # Use the analytic result: F[H_n(x) * exp(-x^2/2)] involves H_n with phase
    # For the standard Fourier transform F[f](xi) = integral f(x) exp(-2*pi*i*x*xi) dx
    # But in physics convention, F[f](xi) = (1/sqrt(2*pi)) * integral f(x) exp(-i*x*xi) dx
    # Let's use the physics convention where F[exp(-x^2/2)] = exp(-xi^2/2)
    # F[H_n(x) * exp(-x^2/2)] = (-i)^n * H_n(x) * exp(-x^2/2) (in physics convention)
    
    # Actually, the precise relation: if psi_n(x) = (1/sqrt(2^n n! sqrt(pi))) H_n(x) exp(-x^2/2)
    # then F[psi_n] = (-i)^n psi_n
    # So for unnormalized: F[H_n(x) exp(-x^2/2)] = sqrt(2*pi) * (-i)^n * H_n(x) * exp(-x^2/2)
    # Wait, let me be more careful.
    
    # Standard result: F[exp(-x^2/2)](xi) = sqrt(2*pi) * exp(-xi^2/2) in the convention
    # F[f](xi) = integral_{-inf}^{inf} f(x) exp(-2*pi*i*x*xi) dx
    # In the convention F[f](xi) = (1/sqrt(2*pi)) integral f(x) exp(-i*x*xi) dx,
    # F[exp(-x^2/2)](xi) = exp(-xi^2/2)
    
    # Let's use the symmetric convention and compute numerically to be safe.
    # For a function on [-L, L], use FFT.
    
    # Actually, let me use the analytic formula. The key property is that the Hermite-Gaussian
    # functions are eigenfunctions of the Fourier transform. So:
    # If g(x) = sum_k c_k H_k(x) exp(-x^2/2), then
    # F[g](xi) = sum_k c_k * F[H_k(x) exp(-x^2/2)](xi)
    
    # In the convention where F is unitary (1/sqrt(2*pi) factor):
    # F[H_k(x) exp(-x^2/2)](xi) = (-i)^k H_k(x) exp(-x^2/2) * (normalization factor)
    
    # Let me just compute it numerically with FFT for robustness.
    L = 20.0
    N = 1 << 16  # 65536 points
    x = np.linspace(-L, L, N, endpoint=False)
    dx = x[1] - x[0]
    
    H0 = eval_hermite(0, x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    gauss = np.exp(-x**2 / 2.0)
    g = (c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12) * gauss
    
    # FFT to compute Fourier transform
    # F[g](xi) ≈ dx * FFT(g) * exp(-2*pi*i*k*x[0]) ... this gets complicated
    # Let's use the property that for the symmetric Fourier transform:
    # If we define F[f](xi) = integral f(x) exp(-i*x*xi) dx, then
    # F[g](xi) ≈ dx * sum_j g(x_j) exp(-i*x_j*xi)
    
    # For the sup norm, we need max |F[g](xi)|. Let's evaluate on a grid of xi.
    xi = np.linspace(-L, L, N, endpoint=False)
    # This would be O(N^2) which is too slow. Let's use FFT.
    
    # FFT approach: F[g](xi_k) where xi_k = 2*pi*k/(N*dx) for k = -N/2, ..., N/2-1
    G = np.fft.fftshift(np.fft.fft(g)) * dx
    xi_fft = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(N, d=dx))
    
    norm_g = np.max(np.abs(g))
    norm_G = np.max(np.abs(G))
    
    return norm_g * norm_G

def construct():
    """Optimize c0, c4, c8 to minimize the uncertainty bound C4."""
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin
    
    # Initial guess from Goncalves et al. 2017
    c0_init = -113.0 / 100.0
    c4_init = 1.0 / 25.0
    c8_init = 1.0 / 3240.0
    
    best_params = [c0_init, c4_init, c8_init]
    best_score = _compute_bound(c0_init, c4_init, c8_init)
    
    # Try differential evolution for global optimization
    if time.time() < deadline - 30:
        try:
            def objective(params):
                if time.time() > deadline:
                    return 1e10
                c0, c4, c8 = params
                try:
                    return _compute_bound(c0, c4, c8)
                except Exception:
                    return 1e10
            
            result = differential_evolution(
                objective,
                bounds=[(-10, 10), (-10, 10), (-10, 10)],
                seed=42,
                maxiter=50,
                tol=1e-8,
                popsize=15,
                polish=True
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Refine with local optimization
    if time.time() < deadline - 10:
        try:
            def objective_local(params):
                if time.time() > deadline:
                    return 1e10
                c0, c4, c8 = params
                try:
                    return _compute_bound(c0, c4, c8)
                except Exception:
                    return 1e10
            
            result2 = minimize(
                objective_local,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if result2.fun < best_score:
                best_score = result2.fun
                best_params = result2.x
        except Exception:
            pass
    
    # One more refinement with different starting points
    if time.time() < deadline - 5:
        try:
            for offset in [0.1, -0.1, 0.5, -0.5]:
                if time.time() > deadline - 5:
                    break
                start = [best_params[0] + offset, best_params[1] + offset, best_params[2] + offset]
                result3 = minimize(
                    objective_local,
                    start,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10}
                )
                if result3.fun < best_score:
                    best_score = result3.fun
                    best_params = result3.x
        except Exception:
            pass
    
    return np.array(best_params)
