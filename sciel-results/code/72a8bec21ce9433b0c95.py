# EVOLVE-BLOCK-START
import numpy as np


def rudin_problem10() -> np.ndarray:
    """
    Returns the coefficients for the sectorial spherical harmonic of degree 10 (m=10).
    This harmonic is expected to have a lower L4/L2 norm ratio than the zonal harmonic (m=0)
    because it is "flatter" (zero at the poles and spread over the equator) compared to
    the zonal harmonic which is peaked at the poles.
    """
    k = 10
    # Initialize coefficients to zero
    coeffs = np.zeros(2 * k + 1, dtype=np.float64)
    
    # The sectorial harmonic corresponds to m = 10.
    # The index for m is i, where m = i - k.
    # So for m = 10, i = k + k = 20.
    coeffs[20] = 1.0
    
    return coeffs


# EVOLVE-BLOCK-END
