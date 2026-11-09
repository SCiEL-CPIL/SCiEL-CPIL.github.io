"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the hidden function using a Fourier series expansion.
    
    Structure:
    - 10 parameters total.
    - 5 sine terms and 5 cosine terms.
    - Fixed frequencies chosen to be orthogonal over the domain [0.1, 6].
    - The basis functions are sin(k * pi * (x - x_min) / L) and cos(k * pi * (x - x_min) / L)
      where L is the length of the domain and x_min is the start of the domain.
      This ensures the basis functions are well-behaved and orthogonal-ish,
      aiding BFGS convergence.
      
    Domain: [0.1, 6.0]
    L = 6.0 - 0.1 = 5.9
    x_min = 0.1
    
    Frequencies k = 1, 2, 3, 4, 5
    
    y = sum_{k=1}^{5} [ a_k * cos(k * pi * (x - 0.1) / 5.9) + b_k * sin(k * pi * (x - 0.1) / 5.9) ]
    
    params[0:5] = a_k (cosine coefficients)
    params[5:10] = b_k (sine coefficients)
    """
    x = np.asarray(x, dtype=float)
    
    # Domain parameters
    x_min = 0.1
    L = 5.9
    
    # Normalize x to [0, L] relative to x_min
    # t = (x - x_min) / L * pi  ->  range [0, pi * 5.9 / 5.9] = [0, pi]? 
    # Wait, standard Fourier basis on [0, L] uses sin(n*pi*x/L).
    # Let's use the standard form: sin(n * pi * (x - x_min) / L)
    
    t = np.pi * (x - x_min) / L
    
    # Frequencies
    k = np.arange(1, 6)  # [1, 2, 3, 4, 5]
    
    # Compute basis matrices
    # Shape of t: (N,)
    # Shape of k: (5,)
    # We want (N, 5) for each term
    
    # Cosine terms
    cos_args = np.outer(t, k)
    cos_basis = np.cos(cos_args)  # Shape (N, 5)
    
    # Sine terms
    sin_args = np.outer(t, k)
    sin_basis = np.sin(sin_args)  # Shape (N, 5)
    
    # Parameters
    a_coeffs = params[0:5]  # Cosine weights
    b_coeffs = params[5:10] # Sine weights
    
    # Linear combination
    # y = cos_basis @ a_coeffs + sin_basis @ b_coeffs
    y = cos_basis @ a_coeffs + sin_basis @ b_coeffs
    
    return y
# EVOLVE-BLOCK-END
