"""Symbolic regression with adaptive RBF + polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using an adaptive basis:
    - 4 polynomial terms (constant, x, x^2, x^3) in scaled variable
    - 6 Gaussian RBF terms with learnable centers and amplitudes
    
    The problem is nonlinear in params (due to learnable centers), but BFGS
    can handle this. The polynomial part captures global trend, while the
    RBFs capture local structure.
    
    Parameters (length 10):
    p0: constant term
    p1: linear coefficient (for x)
    p2: quadratic coefficient (for x^2)
    p3: cubic coefficient (for x^3)
    p4-p9: 3 pairs of (center, amplitude) for 3 Gaussian RBFs
           p4: center1, p5: amplitude1
           p6: center2, p7: amplitude2
           p8: center3, p9: amplitude3
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    # Scale to [-1, 1] for numerical stability of polynomial terms
    z = x / 2.0
    
    # Polynomial part (degree 3)
    z2 = z * z
    z3 = z2 * z
    
    y = params[0] + params[1] * z + params[2] * z2 + params[3] * z3
    
    # RBF part: 3 Gaussian terms with learnable centers and amplitudes
    # Fixed sigma for stability; learnable centers and amplitudes
    sigma = 0.6  # moderate width
    
    # Extract centers and amplitudes
    centers = np.array([params[4], params[6], params[8]])
    amplitudes = np.array([params[5], params[7], params[9]])
    
    # Compute RBF contributions
    for i in range(3):
        diff = x - centers[i]
        gauss = amplitudes[i] * np.exp(-diff ** 2 / (2.0 * sigma ** 2))
        y += gauss
    
    return y
# EVOLVE-BLOCK-END
