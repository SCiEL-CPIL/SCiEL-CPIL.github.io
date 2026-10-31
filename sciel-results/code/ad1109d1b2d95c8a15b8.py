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
    Uses a band-limited signal model with a DC offset and multiple frequency components.
    Employs L-BFGS-B with multiple random restarts to find a robust local minimum.
    """
    x = np.linspace(-1/4, 1/4, N, endpoint=False)
    
    # Increase number of frequency components to allow for more complex cancellation patterns
    n_freq = 12
    freqs = np.arange(1, n_freq + 1)
    
    # Parameter vector: [dc_offset, a1, ..., a_n, phi1, ..., phi_n]
    # Total dimensions: 1 + 2 * n_freq
    n_params = 1 + 2 * n_freq
    
    def base_signal(x, params):
        """
        params: [dc, a1, ..., a_n, phi1, ..., phi_n]
        """
        dc = params[0]
        amps = params[1:1+n_freq]
        phases = params[1+n_freq:1+2*n_freq]
        signal = np.full_like(x, dc, dtype=float)
        for i in range(n_freq):
            signal += amps[i] * np.sin(2 * np.pi * freqs[i] * x + phases[i])
        return signal
    
    def objective(params):
        heights = base_signal(x, params)
        return _autoconvolution_upper_bound(heights)
    
    # Define bounds
    # DC offset should be non-zero to avoid division by zero, but optimizer handles inf.
    # We allow a wide range for DC and amplitudes.
    dc_bound = (-5.0, 5.0)
    amp_bounds = [(-2.0, 2.0)] * n_freq
    phase_bounds = [(-np.pi, np.pi)] * n_freq
    bounds = [dc_bound] + amp_bounds + phase_bounds
    
    # Multiple random restarts to escape local minima
    np.random.seed(42)
    best_score = float('inf')
    best_params = None
    
    n_restarts = 25
    for restart in range(n_restarts):
        # Random initial parameters
        if restart == 0:
            # Start with a reasonable initial guess: small DC, small amps
            x0 = np.zeros(n_params)
            x0[0] = 1.0  # DC
            x0[1] = 0.5  # First amp
            x0[1 + n_freq] = 0.0  # First phase
        else:
            # Random restarts with varying DC to ensure non-zero mean
            x0 = np.zeros(n_params)
            x0[0] = np.random.uniform(-2.0, 2.0)
            # Ensure DC is not too close to 0 initially
            if abs(x0[0]) < 0.1:
                x0[0] = 0.5 * (1 if x0[0] >= 0 else -1)
            x0[1:1+n_freq] = np.random.uniform(-1.0, 1.0, n_freq)
            x0[1+n_freq:1+2*n_freq] = np.random.uniform(-np.pi, np.pi, n_freq)
        
        # Optimize using L-BFGS-B
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                             bounds=bounds,
                             options={'maxiter': 800, 'ftol': 1e-9, 'gtol': 1e-7})
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            continue
    
    # If no valid result, fall back to a constant signal (score 1.0)
    if best_params is None:
        best_params = np.zeros(n_params)
        best_params[0] = 1.0
    
    # Get the optimal heights
    heights = base_signal(x, best_params)
    
    # Ensure the result is a valid signed step function
    # Clipping is safe as the objective is scale-invariant, but prevents numerical issues
    heights = np.clip(heights, -10, 10)
    
    return heights
# EVOLVE-BLOCK-END
