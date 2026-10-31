"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _autoconvolution_upper_bound(heights):
    """
    Compute the upper bound on C3 for a given signed step function.
    """
    h = np.asarray(heights, dtype=float)
    n = len(h)
    if n == 0:
        return float('inf')
    
    # Compute autoconvolution
    autoconv = np.correlate(h, h, mode='full')
    
    # Sum of absolute values of autoconvolution
    sum_abs_autoconv = np.sum(np.abs(autoconv))
    
    # Sum of heights
    sum_h = np.sum(h)
    sum_h_sq = sum_h * sum_h
    
    if abs(sum_h_sq) < 1e-12:
        return float('inf')
    
    return sum_abs_autoconv / sum_h_sq


def construct():
    """
    Construct a signed step function that minimizes the autoconvolution upper bound.
    Uses band-limited signal model with multi-frequency sine components and
    robust optimization via L-BFGS-B with multiple restarts.
    """
    x = np.linspace(-1/4, 1/4, N, endpoint=False)
    
    # Use more frequency components for better expressiveness
    n_freq = 8
    freqs = np.arange(1, n_freq + 1)
    
    def base_signal(x, params):
        """
        params: [a1, ..., a_n, phi1, ..., phi_n]
        """
        amps = params[:n_freq]
        phases = params[n_freq:2*n_freq]
        signal = np.zeros_like(x)
        for i in range(n_freq):
            signal += amps[i] * np.sin(2 * np.pi * freqs[i] * x + phases[i])
        return signal
    
    def objective(params):
        heights = base_signal(x, params)
        return _autoconvolution_upper_bound(heights)
    
    # Define bounds for amplitudes and phases
    amp_bounds = [(-2.0, 2.0)] * n_freq
    phase_bounds = [(-np.pi, np.pi)] * n_freq
    bounds = amp_bounds + phase_bounds
    
    # Multiple random restarts to escape local minima
    np.random.seed(42)
    best_score = float('inf')
    best_params = None
    
    n_restarts = 10
    for restart in range(n_restarts):
        # Random initial parameters
        if restart == 0:
            # Start with a reasonable initial guess
            x0 = np.zeros(2 * n_freq)
            x0[0] = 1.0  # First amplitude
            x0[n_freq] = 0.0  # First phase
        else:
            x0 = np.random.uniform(-1.0, 1.0, 2 * n_freq)
        
        # Optimize using L-BFGS-B
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                             bounds=bounds,
                             options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6})
            
            if result.success or result.fun < best_score:
                if result.fun < best_score:
                    best_score = result.fun
                    best_params = result.x.copy()
        except Exception:
            continue
    
    # If no valid result, fall back to a simple oscillating signal
    if best_params is None:
        best_params = np.zeros(2 * n_freq)
        best_params[0] = 1.0
    
    # Get the optimal heights
    heights = base_signal(x, best_params)
    
    # Ensure the result is a valid signed step function
    heights = np.clip(heights, -10, 10)
    
    return heights
# EVOLVE-BLOCK-END
