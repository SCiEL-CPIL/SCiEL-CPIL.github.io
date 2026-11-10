"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the unknown function using a product-of-sinusoids (Weyl system) basis.
    Each basis function is cos(k_i * x + phi_i) or sin(k_i * x + phi_i) with
    non-integer frequencies to capture diverse smooth functions on [-3, 3].
    The model is linear in params, allowing BFGS to converge quickly.
    """
    x = np.asarray(x, dtype=np.float64)
    original_shape = x.shape
    x_flat = x.flatten()
    
    if x_flat.size == 0:
        return np.array([]).reshape(original_shape)
    
    # Weyl system basis: 10 distinct sinusoidal terms with non-integer frequencies
    # and fixed phases. Frequencies chosen to span low to moderate frequencies
    # relevant for functions on [-3, 3].
    
    # Frequencies (non-integer, spread across relevant range)
    k = np.array([
        0.5,    # low freq
        1.0,    # fundamental
        1.5,    # 
        2.0,    # second harmonic
        2.5,    #
        3.0,    # third harmonic
        3.5,    #
        4.0,    # fourth harmonic
        4.5,    #
        5.0     # fifth harmonic
    ])
    
    # Phases (fixed, distinct)
    phi = np.array([
        0.0,
        np.pi / 4,
        np.pi / 2,
        3 * np.pi / 4,
        np.pi,
        5 * np.pi / 4,
        3 * np.pi / 2,
        7 * np.pi / 4,
        2 * np.pi,
        9 * np.pi / 4
    ])
    
    # Build basis: even indices use cos, odd indices use sin
    basis = np.empty((x_flat.size, 10), dtype=np.float64)
    for i in range(10):
        arg = k[i] * x_flat + phi[i]
        if i % 2 == 0:
            basis[:, i] = np.cos(arg)
        else:
            basis[:, i] = np.sin(arg)
    
    # Linear combination
    y = basis @ params
    
    # Ensure finite values
    y = np.where(np.isfinite(y), y, 0.0)
    
    return y.reshape(original_shape)
# EVOLVE-BLOCK-END
