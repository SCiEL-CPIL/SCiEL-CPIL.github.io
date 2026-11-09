import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Product-of-sinusoids basis (Weyl system) with 10 terms.
    Each term is either cos(k_i * x + phi_i) or sin(k_i * x + phi_i)
    with fixed, non-integer frequencies and phases.
    """
    # Fixed non-integer frequencies and phases for the 10 basis functions
    # Using distinct frequencies to ensure good coverage of the domain [-3, 3]
    # The frequencies are chosen to be non-integer to avoid periodicity alignment
    # with the domain boundaries
    
    # Scale x to improve numerical conditioning
    t = x / 3.0  # Scale to roughly [-1, 1]
    
    # Define 10 distinct non-integer frequencies
    # These are chosen to be spread out and non-integer
    frequencies = np.array([
        1.7,   # ~1.7 * x
        2.3,   # ~2.3 * x
        3.1,   # ~3.1 * x
        4.7,   # ~4.7 * x
        5.3,   # ~5.3 * x
        6.1,   # ~6.1 * x
        7.9,   # ~7.9 * x
        8.3,   # ~8.3 * x
        9.7,   # ~9.7 * x
        11.3   # ~11.3 * x
    ])
    
    # Define phases to break symmetry
    phases = np.array([
        0.3,
        1.1,
        2.5,
        0.7,
        3.3,
        1.9,
        4.1,
        0.5,
        2.7,
        1.3
    ])
    
    # Compute arguments
    args = np.outer(t, frequencies) + phases[np.newaxis, :]  # Shape: (N, 10)
    
    # Use a mix of sin and cos for diversity
    # Even indices: cos, odd indices: sin
    cos_terms = np.cos(args[:, 0:10:2])  # Terms 0, 2, 4, 6, 8
    sin_terms = np.sin(args[:, 1:10:2])  # Terms 1, 3, 5, 7, 9
    
    # Combine into a single basis matrix
    # basis[:, 2*i] = cos term i, basis[:, 2*i+1] = sin term i
    basis = np.zeros((len(x), 10))
    basis[:, 0:10:2] = cos_terms
    basis[:, 1:10:2] = sin_terms
    
    # Linear combination with params
    y = basis @ params
    
    return y
