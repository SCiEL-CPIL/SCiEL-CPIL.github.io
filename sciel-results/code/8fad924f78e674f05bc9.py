"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # params[0]..params[9] are the 10 coefficients
    # We use a basis of 10 functions that span a rich space of smooth functions on [-3, 3]
    
    # Basis: powers of x (0 to 9)
    # This is a polynomial of degree 9.
    # With 10 parameters, we can fit any polynomial up to degree 9.
    # This is very flexible and smooth.
    # However, high-degree polynomials can be ill-conditioned.
    # Let's use a scaled variable to improve conditioning.
    
    # Scale x to [-1, 1] for better numerical stability
    z = x / 3.0
    
    # Compute powers of z
    # We want to compute sum_{i=0}^{9} params[i] * z^i
    # We can do this efficiently using Horner's method or simply matrix multiplication
    
    # Let's build the basis matrix
    # N x 10 matrix where row i is [1, z_i, z_i^2, ..., z_i^9]
    
    # To avoid potential overflow or underflow (though z is in [-1,1] so it's safe),
    # and to be efficient:
    
    # z_pow[0] = 1
    # z_pow[1] = z
    # ...
    # z_pow[9] = z^9
    
    # We can compute this iteratively
    basis = np.ones((len(x), 10))
    for i in range(1, 10):
        basis[:, i] = basis[:, i-1] * z
    
    return basis @ params
# EVOLVE-BLOCK-END
