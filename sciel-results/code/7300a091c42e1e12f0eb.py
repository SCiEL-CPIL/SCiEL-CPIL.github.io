"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_hermite

def construct():
    # Grid for numerical integration - use finer grid for accuracy
    x = np.linspace(-20, 20, 16000)
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
    
    def compute_ratio(c0, c4, c8):
        """Compute the uncertainty ratio for given coefficients."""
        try:
            # Enforce g(0) = 0 => c12 is determined
            c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
            
            # Construct g(x)
            g = (c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12) * np.exp(-0.5 * x**2)
            
            # Compute norms using trapezoidal rule for better accuracy
            norm2_sq = np.trapz(g**2, x)
            if norm2_sq < 1e-12:
                return 1e10
                
            norm4 = np.trapz(np.abs(g)**4, x)
            norm4_val = norm4 ** 0.25
            
            # Fourier Transform using FFT with proper scaling
            n = len(g)
            nfft = 1
            while nfft < 4 * n:
                nfft *= 2
            
            G = np.fft.fft(g, n=nfft) * dx
            freq = np.fft.fftfreq(nfft, d=dx)
            
            # Full integral over all frequencies
            norm2_hat_sq = np.sum(np.abs(G)**2) * (2 * np.pi / nfft)
            norm4_hat = np.sum(np.abs(G)**4) * (2 * np.pi / nfft)
            norm4_hat_val = norm4_hat ** 0.25
            
            ratio = (norm4_val * norm4_hat_val) / (np.sqrt(norm2_sq) * np.sqrt(norm2_hat_sq))
            
            return ratio
        except Exception:
            return 1e10
    
    def objective_from_params(params):
        """Convert log-magnitude and sign parameters to coefficients and compute objective."""
        try:
            m0, m4, m8, s0, s4, s8 = params
            
            sign0 = 1.0 if s0 > 0 else -1.0
            sign4 = 1.0 if s4 > 0 else -1.0
            sign8 = 1.0 if s8 > 0 else -1.0
            
            c0 = sign0 * np.exp(m0)
            c4 = sign4 * np.exp(m4)
            c8 = sign8 * np.exp(m8)
            
            return compute_ratio(c0, c4, c8)
        except Exception:
            return 1e10
    
    # Define bounds for differential evolution
    bounds = [(-5, 4), (-5, 4), (-5, 4), (-1, 1), (-1, 1), (-1, 1)]
    
    # Run differential evolution with multiple seeds for robustness
    best_ratio = 1e10
    best_params = None
    
    for seed in [42, 123, 456, 789, 101]:
        result = differential_evolution(
            objective_from_params, bounds, 
            seed=seed, maxiter=200, popsize=30,
            tol=1e-10, polish=False,
            init='sobol',
            mutation=(0.4, 1.0)
        )
        
        if result.fun < best_ratio:
            best_ratio = result.fun
            best_params = result.x
    
    if best_params is not None:
        m0, m4, m8, s0, s4, s8 = best_params
        sign0 = 1.0 if s0 > 0 else -1.0
        sign4 = 1.0 if s4 > 0 else -1.0
        sign8 = 1.0 if s8 > 0 else -1.0
        c0 = sign0 * np.exp(m0)
        c4 = sign4 * np.exp(m4)
        c8 = sign8 * np.exp(m8)
        
        # Polish with local optimization using direct coefficients
        def local_obj(coeffs):
            c0_l, c4_l, c8_l = coeffs
            return compute_ratio(c0_l, c4_l, c8_l)
        
        # Try multiple local optimization starts
        best_local = (c0, c4, c8)
        best_local_val = best_ratio
        
        # Start from DE result
        local_result = minimize(
            local_obj, 
            [c0, c4, c8],
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        if local_result.fun < best_local_val:
            best_local_val = local_result.fun
            best_local = tuple(local_result.x)
        
        # Try perturbations of the best solution
        np.random.seed(42)
        for _ in range(5):
            perturbed = [best_local[i] * (1 + np.random.uniform(-0.1, 0.1)) for i in range(3)]
            local_result = minimize(
                local_obj, 
                perturbed,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if local_result.fun < best_local_val:
                best_local_val = local_result.fun
                best_local = tuple(local_result.x)
        
        c0, c4, c8 = best_local
        return np.array([c0, c4, c8])
    else:
        # Fallback to seed
        return np.array([-1.13, 0.04, 1.0 / 3240.0])

# EVOLVE-BLOCK-END
