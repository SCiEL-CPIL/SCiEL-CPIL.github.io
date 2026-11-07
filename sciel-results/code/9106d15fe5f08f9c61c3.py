"""Symbolic regression with learnable-frequency Fourier series."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a 10-parameter Fourier series with
    learnable angular frequencies and phases. Amplitudes are fixed to a
    deterministic decay sequence 1/k to keep the optimization landscape
    smooth for BFGS.
    
    Parameters (length 10):
    f0, f1, ..., f4: angular frequencies for the first 5 harmonics
    p0, p1, ..., p4: phases for the first 5 harmonics
    
    Model: y(x) = sum_{k=1}^{5} (1/k) * cos(f_{k-1} * x + p_{k-1})
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    # Extract frequencies and phases
    freqs = params[0:5]   # angular frequencies
    phases = params[5:10] # phases
    
    # Fixed amplitude decay: a_k = 1/k for k=1..5
    amps = 1.0 / np.arange(1, 6, dtype=np.float64)
    
    # Build the Fourier sum
    # y(x) = sum_{k=1}^{5} a_k * cos(f_{k-1} * x + p_{k-1})
    # Shape: (n_points, 5)
    x_expanded = x[:, np.newaxis]          # (n_points, 1)
    f_expanded = freqs[np.newaxis, :]      # (1, 5)
    p_expanded = phases[np.newaxis, :]     # (1, 5)
    a_expanded = amps[np.newaxis, :]       # (1, 5)
    
    # Compute argument: f_k * x + p_k
    args = f_expanded * x_expanded + p_expanded  # (n_points, 5)
    
    # Compute weighted cosines
    y = np.sum(a_expanded * np.cos(args), axis=1)  # (n_points,)
    
    return y
# EVOLVE-BLOCK-END
