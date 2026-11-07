"""Symbolic regression with 10-parameter Fourier series and fixed amplitude decay."""
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-parameter Fourier series with learnable angular frequencies and phases.
    
    Parameters:
        params[0:5]  -> angular frequencies omega_1..omega_5 (positive)
        params[5:10] -> phases phi_1..phi_5
    
    Amplitudes are fixed to 1/k for k=1..5, giving a smooth, well-conditioned
    landscape for BFGS. The model is:
        f(x) = sum_{k=1}^{5} (1/k) * sin(omega_k * x + phi_k)
    
    This captures non-periodic smooth structures through constructive
    interference of optimally tuned harmonics.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Extract frequencies and phases
    # Use softplus-like transform to ensure frequencies stay positive
    # omega_k = exp(params[0:5]) to guarantee positivity and smooth gradient
    omegas = np.exp(params[0:5])  # shape (5,)
    phases = params[5:10]        # shape (5,)
    
    # Fixed amplitude decay: 1/k for k=1..5
    amps = 1.0 / np.arange(1, 6, dtype=np.float64)  # shape (5,)
    
    # Build prediction: sum_k amps[k] * sin(omegas[k] * x + phases[k])
    # x has shape (n,), omegas has shape (5,)
    # We need: for each x_i, sum over k of amps[k] * sin(omegas[k]*x_i + phases[k])
    # Use broadcasting: x[:, None] * omegas[None, :] gives (n, 5)
    arg = x[:, np.newaxis] * omegas[np.newaxis, :] + phases[np.newaxis, :]
    # arg shape: (n, 5)
    sines = np.sin(arg)  # shape (n, 5)
    prediction = sines @ amps  # shape (n,)
    
    return prediction
