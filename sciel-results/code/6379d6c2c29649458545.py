"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _autoconvolution_upper_bound(heights):
    """
    Compute the upper bound on C3 for a given signed step function using FFT.
    """
    h = np.asarray(heights, dtype=float)
    n = len(h)
    if n == 0:
        return float('inf')
    
    # Compute linear autocorrelation using FFT
    # Zero-pad to length at least 2n
    N_fft = 1 << (2 * n - 1).bit_length()
    H = np.fft.rfft(h, N_fft)
    power = np.abs(H) ** 2
    autoconv_full = np.fft.irfft(power, N_fft)
    
    # Extract the positive lags (0 to n-1)
    # autoconv_full[0:n] contains c_0, c_1, ..., c_{n-1}
    c_pos = autoconv_full[0:n]
    
    # The autocorrelation is symmetric: c_{-k} = c_k
    # Sum of absolute values: |c_0| + 2 * sum(|c_1| ... |c_{n-1}|)
    # Note: c_0 is sum(h^2), which is always positive.
    # c_k for k>0 can be negative.
    sum_abs_autoconv = c_pos[0] + 2.0 * np.sum(np.abs(c_pos[1:]))
    
    sum_h = np.sum(h)
    sum_h_sq = sum_h * sum_h
    
    if abs(sum_h_sq) < 1e-12:
        return float('inf')
        
    return sum_abs_autoconv / sum_h_sq


def construct():
    """
    Construct a signed step function that minimizes the autoconvolution upper bound.
    Uses a band-limited signal model with multiple frequency components.
    """
    x = np.linspace(-1/4, 1/4, N, endpoint=False)
    
    # Increase number of frequency components to allow for more complex cancellation
    n_freq = 20
    freqs = np.arange(1, n_freq + 1)
    
    # Parameter vector: [dc_offset, a1, ..., a_n, phi1, ..., phi_n]
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
    dc_bound = (-5.0, 5.0)
    amp_bounds = [(-2.0, 2.0)] * n_freq
    phase_bounds = [(-np.pi, np.pi)] * n_freq
    bounds = [dc_bound] + amp_bounds + phase_bounds
    
    # Multiple random restarts
    np.random.seed(42)
    best_score = float('inf')
    best_params = None
    
    n_restarts = 50
    for restart in range(n_restarts):
        # Random initial parameters
        if restart == 0:
            # Start with a reasonable initial guess
            x0 = np.zeros(n_params)
            x0[0] = 1.0  # DC
            x0[1] = 0.5  # First amp
        else:
            # Random restarts
            x0 = np.zeros(n_params)
            x0[0] = np.random.uniform(-2.0, 2.0)
            if abs(x0[0]) < 0.1:
                x0[0] = 0.5 * (1 if x0[0] >= 0 else -1)
            x0[1:1+n_freq] = np.random.uniform(-1.0, 1.0, n_freq)
            x0[1+n_freq:1+2*n_freq] = np.random.uniform(-np.pi, np.pi, n_freq)
        
        # Optimize using L-BFGS-B
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                             bounds=bounds,
                             options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            continue
    
    # Fallback
    if best_params is None:
        best_params = np.zeros(n_params)
        best_params[0] = 1.0
    
    # Get the optimal heights
    heights = base_signal(x, best_params)
    
    # Clip to prevent extreme values if optimization diverges slightly
    heights = np.clip(heights, -10, 10)
    
    return heights
# EVOLVE-BLOCK-END
