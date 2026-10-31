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
            # Pad to next power of 2 for efficiency
            n = len(g)
            nfft = 1
            while nfft < 4 * n:
                nfft *= 2
            
            G = np.fft.fft(g, n=nfft) * dx
            freq = np.fft.fftfreq(nfft, d=dx)
            
            # The full integral over all frequencies
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
            
            # Use smooth sign via tanh to avoid discontinuities
            sign0 = np.tanh(s0)
            sign4 = np.tanh(s4)
            sign8 = np.tanh(s8)
            
            # Avoid zero coefficients
            min_abs = 1e-6
            if abs(sign0) < min_abs:
                sign0 = min_abs if s0 >= 0 else -min_abs
            if abs(sign4) < min_abs:
                sign4 = min_abs if s4 >= 0 else -min_abs
            if abs(sign8) < min_abs:
                sign8 = min_abs if s8 >= 0 else -min_abs
            
            c0 = sign0 * np.exp(m0)
            c4 = sign4 * np.exp(m4)
            c8 = sign8 * np.exp(m8)
            
            return compute_ratio(c0, c4, c8)
        except Exception:
            return 1e10
    
    # Define bounds for differential evolution
    # m_i: log-magnitude, range from -5 to 4 (coefficients from ~0.0067 to ~54.6)
    # s_i: sign parameter via tanh, range from -2 to 2
    bounds = [(-5, 4), (-5, 4), (-5, 4), (-2, 2), (-2, 2), (-2, 2)]
    
    # Run differential evolution with multiple seeds for robustness
    best_ratio = 1e10
    best_params = None
    
    for seed in [42, 123, 456, 789, 321]:
        result = differential_evolution(
            objective_from_params, bounds, 
            seed=seed, maxiter=200, popsize=25,
            tol=1e-10, polish=False,
            init='sobol',
            mutation=(0.3, 1.2),
            recombination=0.9
        )
        
        if result.fun < best_ratio:
            best_ratio = result.fun
            best_params = result.x
    
    if best_params is not None:
        m0, m4, m8, s0, s4, s8 = best_params
        sign0 = np.tanh(s0)
        sign4 = np.tanh(s4)
        sign8 = np.tanh(s8)
        
        min_abs = 1e-6
        if abs(sign0) < min_abs:
            sign0 = min_abs if s0 >= 0 else -min_abs
        if abs(sign4) < min_abs:
            sign4 = min_abs if s4 >= 0 else -min_abs
        if abs(sign8) < min_abs:
            sign8 = min_abs if s8 >= 0 else -min_abs
        
        c0 = sign0 * np.exp(m0)
        c4 = sign4 * np.exp(m4)
        c8 = sign8 * np.exp(m8)
        
        # Polish with local optimization using direct coefficients
        def local_obj(coeffs):
            c0_l, c4_l, c8_l = coeffs
            return compute_ratio(c0_l, c4_l, c8_l)
        
        local_result = minimize(
            local_obj, 
            [c0, c4, c8],
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        
        if local_result.fun < best_ratio:
            c0, c4, c8 = local_result.x
        
        return np.array([c0, c4, c8])
    else:
        # Fallback to seed
        return np.array([-1.13, 0.04, 1.0 / 3240.0])

# EVOLVE-BLOCK-END
