"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize, dual_annealing
from scipy.special import eval_hermite

def construct():
    # Grid for numerical integration - use finer grid for accuracy
    x = np.linspace(-12, 12, 6000)
    dx = x[1] - x[0]
    
    # Hermite polynomials at x
    H0 = eval_hermite(0, x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    
    # Values at 0 for constraint
    H0_0 = eval_hermite(0, 0)
    H4_0 = eval_hermite(4, 0)
    H8_0 = eval_hermite(8, 0)
    H12_0 = eval_hermite(12, 0)
    
    # Precompute Gaussian weight
    gauss = np.exp(-0.5 * x**2)
    
    def compute_ratio(c0, c4, c8):
        """Compute the uncertainty ratio for given coefficients."""
        try:
            # Enforce g(0) = 0 => c12 is determined
            c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
            
            # Construct g(x)
            poly = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
            g = poly * gauss
            
            # Compute norms using trapezoidal rule for better accuracy
            g2 = g**2
            norm2_sq = np.trapz(g2, x)
            if norm2_sq < 1e-14:
                return 1e10
                
            norm4 = np.trapz(g2**2, x)
            if norm4 < 0:
                norm4 = 0
            norm4_val = norm4 ** 0.25
            
            # Fourier Transform using FFT with proper scaling
            n = len(g)
            nfft = 1
            while nfft < 4 * n:
                nfft *= 2
            
            G = np.fft.fft(g, n=nfft) * dx
            # Parseval: integral |f|^2 dx = (1/2pi) * (nfft * dx) * sum|F|^2 * (2pi/nfft)^2
            # Actually: F(k) = integral f(x) e^{-ikx} dx, so |F|^2 integral = 2pi * integral |f|^2 dx
            # With DFT: F_dft = sum g_j * dx * exp(-2pi i j k / nfft)
            # Continuous FT: F(k) = integral g(x) e^{-ikx} dx
            # The DFT approximates sum g_j dx e^{-2pi i j k / nfft} for k = 2pi m / (nfft*dx)
            # So F(k_m) ≈ F_dft[m] * dx (if we scale properly)
            # Actually F_dft[m] = sum_j g_j exp(-2pi i j m / nfft)
            # We want integral g(x) exp(-i x * 2pi m/(nfft*dx)) dx ≈ dx * sum_j g_j exp(-2pi i j m / nfft) = dx * F_dft[m]
            # So G[m] = dx * F_dft[m] approximates F(2pi m / (nfft*dx))
            
            # Parseval theorem: integral |f(x)|^2 dx = (1/2pi) integral |F(k)|^2 dk
            # Discrete: sum_m |G[m]|^2 * dk = 2pi * integral |g|^2 dx
            # dk = 2pi / (nfft * dx)
            # So sum_m |G[m]|^2 * (2pi/(nfft*dx)) = 2pi * norm2_sq
            # => sum_m |G[m]|^2 = nfft * dx * norm2_sq
            
            # Let's verify and use consistent scaling
            # norm2_hat_sq = (1/2pi) * integral |F(k)|^2 dk = (1/2pi) * sum |G|^2 * dk
            # = (1/2pi) * sum |G|^2 * (2pi/(nfft*dx)) = sum |G|^2 / (nfft*dx)
            # But also by Parseval, this should equal norm2_sq
            # So norm2_hat_sq = sum |G|^2 / (nfft*dx)
            
            norm2_hat_sq = np.sum(np.abs(G)**2) / (nfft * dx)
            
            # For L4: we need integral |F(k)|^4 dk
            # This is not directly given by Parseval, so we compute it directly
            # integral |F(k)|^4 dk ≈ sum_m |G[m]|^4 * dk = sum_m |G[m]|^4 * (2pi/(nfft*dx))
            norm4_hat_integral = np.sum(np.abs(G)**4) * (2 * np.pi / (nfft * dx))
            if norm4_hat_integral < 0:
                norm4_hat_integral = 0
            norm4_hat_val = norm4_hat_integral ** 0.25
            
            # The ratio is A(f) * A(F(f)) where A(f) = ||f||_4 / ||f||_2
            # A(f) = (||f||_4^4 / ||f||_2^4)^(1/4) / (||f||_2^2 / ||f||_2^2)^(1/4)
            # Wait, A(f) is defined as ||f||_4 / ||f||_2
            # So A(f) * A(F(f)) = (||f||_4 / ||f||_2) * (||F||_4 / ||F||_2)
            # = (||f||_4 * ||F||_4) / (||f||_2 * ||F||_2)
            
            ratio = (norm4_val * norm4_hat_val) / (np.sqrt(norm2_sq) * np.sqrt(norm2_hat_sq))
            
            return ratio
        except Exception:
            return 1e10
    
    def objective_from_params(params):
        """Convert log-magnitude and sign parameters to coefficients and compute objective."""
        try:
            m0, m4, m8, s0, s4, s8 = params
            
            # Use smooth sign via tanh to allow gradient-like exploration
            sign0 = np.tanh(s0)
            sign4 = np.tanh(s4)
            sign8 = np.tanh(s8)
            
            # Avoid zero by adding small epsilon
            eps = 1e-8
            sign0 = sign0 if abs(sign0) > eps else eps
            sign4 = sign4 if abs(sign4) > eps else eps
            sign8 = sign8 if abs(sign8) > eps else eps
            
            c0 = sign0 * np.exp(m0)
            c4 = sign4 * np.exp(m4)
            c8 = sign8 * np.exp(m8)
            
            return compute_ratio(c0, c4, c8)
        except Exception:
            return 1e10
    
    # Define bounds for differential evolution
    # m_i: log-magnitude, range from -5 to 2 (coefficients from ~0.0067 to ~7.4)
    # s_i: sign parameter, range from -3 to 3 (tanh maps to roughly -1 to 1)
    bounds = [(-5, 2), (-5, 2), (-5, 2), (-3, 3), (-3, 3), (-3, 3)]
    
    # Run multiple optimization strategies
    best_ratio = 1e10
    best_coeffs = None
    
    # Strategy 1: Differential Evolution with multiple seeds
    for seed in [42, 123, 456, 789]:
        try:
            result = differential_evolution(
                objective_from_params, bounds, 
                seed=seed, maxiter=100, popsize=25,
                tol=1e-6, polish=False,
                init='sobol',
                mutation=(0.5, 1.0),
                recombination=0.8
            )
            
            if result.fun < best_ratio:
                best_ratio = result.fun
                m0, m4, m8, s0, s4, s8 = result.x
                sign0 = np.tanh(s0)
                sign4 = np.tanh(s4)
                sign8 = np.tanh(s8)
                eps = 1e-8
                sign0 = sign0 if abs(sign0) > eps else eps
                sign4 = sign4 if abs(sign4) > eps else eps
                sign8 = sign8 if abs(sign8) > eps else eps
                best_coeffs = np.array([sign0 * np.exp(m0), sign4 * np.exp(m4), sign8 * np.exp(m8)])
        except Exception:
            pass
    
    # Strategy 2: Dual Annealing for global search
    try:
        da_result = dual_annealing(
            objective_from_params, bounds,
            seed=42, maxiter=100, restarts=5,
            local_search_options={'maxiter': 200}
        )
        if da_result.fun < best_ratio:
            best_ratio = da_result.fun
            m0, m4, m8, s0, s4, s8 = da_result.x
            sign0 = np.tanh(s0)
            sign4 = np.tanh(s4)
            sign8 = np.tanh(s8)
            eps = 1e-8
            sign0 = sign0 if abs(sign0) > eps else eps
            sign4 = sign4 if abs(sign4) > eps else eps
            sign8 = sign8 if abs(sign8) > eps else eps
            best_coeffs = np.array([sign0 * np.exp(m0), sign4 * np.exp(m4), sign8 * np.exp(m8)])
    except Exception:
        pass
    
    # Strategy 3: Multi-start local optimization from heuristic starting points
    def local_obj(coeffs):
        c0_l, c4_l, c8_l = coeffs
        return compute_ratio(c0_l, c4_l, c8_l)
    
    # Heuristic starting points based on Hermite structure
    start_points = [
        [-1.0, 0.1, 0.01],
        [1.0, -0.1, -0.01],
        [-0.5, 0.2, 0.005],
        [0.5, -0.2, -0.005],
        [-1.5, 0.05, 0.02],
        [1.5, -0.05, -0.02],
        [-0.8, 0.15, 0.008],
        [0.8, -0.15, -0.008],
        [-1.13, 0.04, 1.0/3240.0],  # Original seed
    ]
    
    for sp in start_points:
        try:
            res = minimize(
                local_obj, sp,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7}
            )
            if res.fun < best_ratio:
                best_ratio = res.fun
                best_coeffs = res.x
        except Exception:
            pass
    
    # Final polish with L-BFGS-B if we have a good candidate
    if best_coeffs is not None:
        try:
            final_res = minimize(
                local_obj, best_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if final_res.fun < best_ratio:
                best_ratio = final_res.fun
                best_coeffs = final_res.x
        except Exception:
            pass
    
    if best_coeffs is not None:
        return best_coeffs
    else:
        # Fallback to seed
        return np.array([-1.13, 0.04, 1.0 / 3240.0])

# EVOLVE-BLOCK-END
