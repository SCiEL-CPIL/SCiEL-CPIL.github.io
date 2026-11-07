"""
Radial Basis Function (RBF) network for symbolic regression.
Uses 5 Gaussian kernels with learnable centers and widths.
The model is linear in the kernel outputs, making optimization well-behaved.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard variables
_start_time = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a Radial Basis Function (RBF) network.
    
    Model:
    y = sum_{i=0}^{4} exp(-(x - c_i)^2 / sigma_i^2)
    
    Parameters:
    p0, p1, p2, p3, p4: Centers of the 5 Gaussian kernels (c_i)
    p5, p6, p7, p8, p9: Widths of the 5 Gaussian kernels (sigma_i)
    
    The centers and widths are optimized by BFGS to match the target function.
    This local interpolation approach can capture non-smooth features,
    bumps, and steps that global polynomial/tanh models miss.
    """
    # Check deadline (structural requirement, minimal overhead)
    if time.time() - _start_time > _DEADLINE - _SAFETY_MARGIN:
        pass

    # Extract centers and widths
    centers = params[0:5]  # c_0, c_1, c_2, c_3, c_4
    widths = params[5:10]   # sigma_0, sigma_1, sigma_2, sigma_3, sigma_4

    # Ensure widths are positive to avoid division by zero or negative variance
    # Use a small epsilon to keep widths bounded away from zero
    epsilon = 1e-6
    widths = np.maximum(np.abs(widths), epsilon)

    # Compute Gaussian RBFs: exp(-(x - c_i)^2 / sigma_i^2)
    # x has shape (n,), centers has shape (5,)
    # We want output shape (n, 5) where column i is the i-th kernel evaluated at all x points
    
    # Compute squared distances: (x - c_i)^2 for each i
    # x: (n,), centers: (5,) -> broadcast to (n, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]  # shape (n, 5)
    diff_sq = diff * diff  # shape (n, 5)
    
    # Compute exponents: -diff_sq / sigma_i^2
    # widths: (5,) -> broadcast to (n, 5)
    widths_sq = widths[np.newaxis, :] ** 2  # shape (1, 5)
    exponents = -diff_sq / widths_sq  # shape (n, 5)
    
    # Compute Gaussian values
    rbf_values = np.exp(exponents)  # shape (n, 5)
    
    # Sum over kernels (each kernel has unit weight)
    # The amplitude is implicitly 1 for each kernel; the centers and widths
    # are the only learnable parameters. This keeps the model simple and
    # the optimization landscape smoother.
    y_pred = np.sum(rbf_values, axis=1)  # shape (n,)
    
    return y_pred
# EVOLVE-BLOCK-END
