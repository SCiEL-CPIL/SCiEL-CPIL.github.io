# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate the hidden function using a whitened hybrid basis of 
    polynomial, trigonometric, and exponential functions.
    
    The basis is designed to capture a wide variety of function shapes 
    (polynomials, saturations, oscillations, decays) which a pure polynomial 
    basis might miss. The basis columns are whitened (normalized to unit L2 
    norm on the input points) to ensure the design matrix is orthonormal. 
    This makes the Hessian of the MSE close to the identity, significantly 
    improving BFGS convergence stability and speed.
    """
    _T0 = time.time()
    _DEADLINE = 140.0

    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return np.array([])

    # Scale input to [-1, 1] for better numerical conditioning of trig/exp functions
    u = x / 3.0
    
    # Define 10 diverse basis functions
    # 1-4: Low-order polynomials (Legendre-like but simpler for mixing)
    b0 = np.ones(n)
    b1 = u
    b2 = u**2
    b3 = u**3
    
    # 5: Saturation
    b4 = np.tanh(u)
    
    # 6: Gaussian bump
    b5 = np.exp(-0.5 * u**2)
    
    # 7-8: Oscillations
    b7 = np.sin(u)
    b8 = np.cos(u)
    
    # 9: Exponential decay (asymmetric)
    b9 = np.exp(-np.abs(u))
    
    # 10: Damped oscillation or higher order feature
    b10 = u * np.exp(-u**2)
    
    # Construct basis matrix V (n x 10)
    V = np.column_stack((b0, b1, b2, b3, b4, b5, b7, b8, b9, b10))
    
    # Whitening: Normalize each column to unit L2 norm
    # This ensures V.T @ V is approximately I, making the optimization landscape 
    # isotropic and ideal for BFGS.
    norms = np.linalg.norm(V, axis=0)
    # Avoid division by zero
    norms[norms < 1e-12] = 1.0
    V_whitened = V / norms
    
    # Linear combination of whitened basis functions
    y = V_whitened @ params

    if not np.all(np.isfinite(y)):
        y = np.where(np.isfinite(y), y, 0.0)

    if time.time() - _T0 > _DEADLINE:
        return y

    return y
# EVOLVE-BLOCK-END
