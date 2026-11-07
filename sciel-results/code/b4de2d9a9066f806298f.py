# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a hybrid basis:
    - 3 Gaussian RBFs (centers and widths fixed, amplitudes learned)
    - 4 Fourier basis functions (frequencies fixed, amplitudes and phases via params)
    - 1 linear term
    - 1 constant
    
    params (length 10):
      p0, p1, p2: amplitudes for 3 Gaussian RBFs
      p3, p4: amplitude, phase for sin(w1*x)
      p5, p6: amplitude, phase for cos(w2*x)
      p7: amplitude for x (linear term)
      p8: amplitude for x^2 (quadratic term)
      p9: constant offset
    """
    # Fixed centers for RBFs
    centers = np.array([-2.0, 0.0, 2.0])
    # Fixed widths for RBFs
    widths = np.array([1.0, 1.0, 1.0])
    
    # Compute RBF terms
    # x shape: (N,), centers shape: (3,)
    # We want (N, 3) for broadcasting
    x_expanded = x[:, np.newaxis]  # (N, 1)
    centers_expanded = centers[np.newaxis, :]  # (1, 3)
    widths_expanded = widths[np.newaxis, :]  # (1, 3)
    
    diff = x_expanded - centers_expanded  # (N, 3)
    rbf_terms = np.exp(-0.5 * (diff / widths_expanded) ** 2)  # (N, 3)
    
    # RBF contributions: p0, p1, p2 are amplitudes
    rbf_contrib = rbf_terms[:, 0] * params[0] + rbf_terms[:, 1] * params[1] + rbf_terms[:, 2] * params[2]
    
    # Fourier terms
    # w1 = 1.0, w2 = 2.0 (fixed frequencies)
    w1 = 1.0
    w2 = 2.0
    
    # sin(w1*x) with amplitude p3 and phase p4
    fourier1 = params[3] * np.sin(w1 * x + params[4])
    
    # cos(w2*x) with amplitude p5 and phase p6
    fourier2 = params[5] * np.cos(w2 * x + params[6])
    
    # Polynomial terms
    poly = params[7] * x + params[8] * x ** 2 + params[9]
    
    return rbf_contrib + fourier1 + fourier2 + poly
# EVOLVE-BLOCK-END
