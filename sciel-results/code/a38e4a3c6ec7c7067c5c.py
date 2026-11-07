"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a Fourier series basis.
    
    Strategy:
    - The target function is approximated by a linear combination of sine and cosine terms.
    - This basis is orthogonal on the domain (after scaling), which ensures the resulting
      least-squares problem is well-conditioned.
    - The model is linear in `params`, allowing the external BFGS optimizer to converge
      reliably to the global minimum (the least-squares solution) without getting stuck
      in local minima.
    
    Basis:
    - 1 constant term
    - 4 cosine terms: cos(k * x_scaled) for k=1..4
    - 5 sine terms: sin(k * x_scaled) for k=1..5
    Total: 10 parameters.
    
    Domain mapping:
    - x is in [-3, 3]. We scale it to [-pi, pi] to align with standard Fourier properties,
      although any linear scaling works for linearity. Scaling to [-pi, pi] makes the
      frequencies integer-like, which can help with conditioning.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        x = x.reshape(1)
        
    # Scale x from [-3, 3] to [-pi, pi]
    x_scaled = x * (np.pi / 3.0)
    
    # Build basis matrix
    # params: [c0, c1, c2, c3, c4, s1, s2, s3, s4, s5]
    
    # Constant term
    basis = np.ones((1, len(x_scaled)))
    
    # Cosine terms k=1..4
    for k in range(1, 5):
        basis = np.vstack([basis, np.cos(k * x_scaled)])
        
    # Sine terms k=1..5
    for k in range(1, 6):
        basis = np.vstack([basis, np.sin(k * x_scaled)])
        
    # Linear combination: y = params @ basis
    # params is shape (10,), basis is shape (10, N)
    # result is shape (N,)
    return basis.T @ params

# EVOLVE-BLOCK-END
