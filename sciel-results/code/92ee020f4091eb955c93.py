"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_hermite
from scipy.integrate import quad

def construct():
    # Grid for numerical integration
    # The function decays as e^{-x^2}, so [-10, 10] is sufficient for high precision
    x = np.linspace(-10, 10, 20000)
    dx = x[1] - x[0]
    
    # Precompute Hermite polynomials on the grid
    H0 = eval_hermite(0, x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    
    # Values at 0 for constraint g(0) = 0
    # H0(0)=1, H4(0)=-12, H8(0)=1680, H12(0)=-103950
    H0_0 = 1.0
    H4_0 = -12.0
    H8_0 = 1680.0
    H12_0 = -103950.0

    def compute_ratio(c0, c4, c8):
        """Compute the uncertainty product A(f) * A(Fourier(f)) for given coefficients."""
        try:
            # Enforce g(0) = 0 => c12 is determined
            # c0*1 + c4*(-12) + c8*1680 + c12*(-103950) = 0
            c12 = -(c0 + c4 * H4_0 + c8 * H8_0) / H12_0
            
            # Construct polynomial part P(x)
            P = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
            
            # Construct g(x) = P(x) * exp(-0.5 * x^2)
            g = P * np.exp(-0.5 * x**2)
            
            # Compute norms in x-domain
            norm2_sq = np.trapz(g**2, x)
            if norm2_sq < 1e-12:
                return 1e10
            norm4 = np.trapz(g**4, x)
            
            # Compute norms in frequency domain
            # Fourier transform of H_n(x) exp(-x^2/2) is (-i)^n H_n(k) exp(-k^2/2) * sqrt(2*pi)
            # For n=0,4,8,12, (-i)^n = 1
            # So G(k) = sqrt(2*pi) * P(k) * exp(-0.5 * k^2)
            # Since P(k) uses the same Hermite polynomials evaluated at k, and our grid is symmetric,
            # we can reuse the same array P for the polynomial part if we assume k=x.
            # The grid x is used for both x and k integration.
            G = np.sqrt(2 * np.pi) * P * np.exp(-0.5 * x**2)
            
            norm2_hat_sq = np.trapz(G**2, x)
            norm4_hat = np.trapz(G**4, x)
            
            if norm2_hat_sq < 1e-12:
                return 1e10
                
            # A(f) = ||f||_4 / ||f||_2
            A_f = (norm4 ** 0.25) / np.sqrt(norm2_sq)
            # A(F) = ||F||_4 / ||F||_2
            A_F = (norm4_hat ** 0.25) / np.sqrt(norm2_hat_sq)
            
            ratio = A_f * A_F
            return ratio
        except Exception:
            return 1e10

    def objective_from_params(params):
        """Convert log-magnitude and sign parameters to coefficients and compute objective."""
        try:
            m0, m4, m8, s0, s4, s8 = params
            
            # Sign extraction
            sign0 = 1.0 if s0 > 0 else -1.0
            sign4 = 1.0 if s4 > 0 else -1.0
            sign8 = 1.0 if s8 > 0 else -1.0
            
            # Magnitude extraction
            c0 = sign0 * np.exp(m0)
            c4 = sign4 * np.exp(m4)
            c8 = sign8 * np.exp(m8)
            
            return compute_ratio(c0, c4, c8)
        except Exception:
            return 1e10

    # Define bounds for differential evolution
    # Log-magnitude range: -5 to 5 (covers ~0.006 to 148)
    # Sign parameter range: -1 to 1
    bounds = [(-5, 5), (-5, 5), (-5, 5), (-1, 1), (-1, 1), (-1, 1)]
    
    best_ratio = 1e10
    best_params = None
    
    # Run differential evolution with multiple seeds
    for seed in [42, 123, 456, 789, 101]:
        result = differential_evolution(
            objective_from_params, bounds, 
            seed=seed, maxiter=200, popsize=25,
            tol=1e-10, polish=False,
            init='sobol',
            mutation=(0.5, 1.0),
            recombination=0.8
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
        
        # Polish with local optimization
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
