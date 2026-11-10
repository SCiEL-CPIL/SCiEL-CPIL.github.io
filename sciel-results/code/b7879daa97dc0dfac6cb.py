# EVOLVE-BLOCK-START
import numpy as np


def rudin_problem10() -> np.ndarray:
    """
    Returns the coefficients for the zonal spherical harmonic of degree 10 (m=0).
    
    The L4/L2 norm ratio is minimized by the harmonic with the lowest spatial frequency,
    which is the zonal harmonic (m=0). This harmonic is independent of the azimuthal angle 
    phi, making it the "flattest" in the angular distribution compared to sectorial or 
    tesseral harmonics which oscillate in phi.
    
    The coefficients are ordered by m from -10 to 10.
    Index i corresponds to m = i - 10.
    For m = 0, i = 10.
    """
    k = 10
    num_coeffs = 2 * k + 1
    coeffs = np.zeros(num_coeffs, dtype=np.float64)
    
    # Set the coefficient for m=0 to 1.0
    # This ensures the L2 norm is 1 (since Y_0 is normalized) and max|c| = 1.0
    coeffs[k] = 1.0
    
    return coeffs


# EVOLVE-BLOCK-END
