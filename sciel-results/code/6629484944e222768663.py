"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier series with 5 learnable angular frequencies and 5 learnable phases.
    Amplitudes are fixed to a deterministic decay sequence (1/k for k=1..5).
    
    params[0:5] = angular frequencies (omega_1..omega_5)
    params[5:10] = phases (phi_1..phi_5)
    
    prediction(x) = sum_{k=1}^{5} (1/k) * sin(omega_k * x + phi_k)
    
    This design:
    - Uses a smooth, non-periodic-like basis through free frequencies
    - Fixed amplitudes keep the optimization landscape smoother for BFGS
    - 10 parameters total, matching the required interface
    """
    # Extract frequencies and phases
    omegas = params[0:5]
    phis = params[5:10]
    
    # Fixed amplitudes: 1/k for k = 1, 2, 3, 4, 5
    amplitudes = np.array([1.0, 0.5, 1.0/3.0, 0.25, 0.2])
    
    # Compute sum of sinusoidal terms
    # x has shape (N,), omegas and phis have shape (5,)
    # We need to compute sum_k amp_k * sin(omega_k * x + phi_k)
    
    # Reshape for broadcasting: x -> (N, 1), omegas -> (1, 5)
    x_col = x.reshape(-1, 1)
    omegas_row = omegas.reshape(1, -1)
    phis_row = phis.reshape(1, -1)
    amps_row = amplitudes.reshape(1, -1)
    
    # Compute the argument of sine for each harmonic at each point
    args = omegas_row * x_col + phis_row  # shape (N, 5)
    
    # Compute sine and weighted sum
    sines = np.sin(args)  # shape (N, 5)
    prediction = np.sum(amps_row * sines, axis=1)  # shape (N,)
    
    return prediction
# EVOLVE-BLOCK-END
