"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize input to [-1, 1] to align with standard Fourier series on [-1, 1]
    # This corresponds to the domain [-2, 2] scaled by 0.5
    z = x / 2.0
    
    # Define a trigonometric basis (Fourier Series)
    # This approach uses a different function family (periodic/oscillatory) 
    # compared to the polynomial basis in the seed program.
    
    # Basis functions:
    # 0: 1 (DC)
    # 1: cos(pi * z)
    # 2: sin(pi * z)
    # 3: cos(2 * pi * z)
    # 4: sin(2 * pi * z)
    # 5: cos(3 * pi * z)
    # 6: sin(3 * pi * z)
    # 7: cos(4 * pi * z)
    # 8: sin(4 * pi * z)
    # 9: cos(5 * pi * z)
    
    # Pre-compute basis values for efficiency
    # z is shape (N,)
    
    # We can compute these vectorized
    # cos_k = cos(k * pi * z)
    # sin_k = sin(k * pi * z)
    
    # To avoid creating too many temporary arrays, we can compute iteratively 
    # or use a small stack. Given N=200, stack is fine.
    
    # Frequencies: 1, 1, 2, 2, 3, 3, 4, 4, 5
    # Types: C, S, C, S, C, S, C, S, C
    
    basis = np.empty((x.shape[0], 10), dtype=float)
    
    # DC term
    basis[:, 0] = 1.0
    
    # k=1
    basis[:, 1] = np.cos(np.pi * z)
    basis[:, 2] = np.sin(np.pi * z)
    
    # k=2
    basis[:, 3] = np.cos(2 * np.pi * z)
    basis[:, 4] = np.sin(2 * np.pi * z)
    
    # k=3
    basis[:, 5] = np.cos(3 * np.pi * z)
    basis[:, 6] = np.sin(3 * np.pi * z)
    
    # k=4
    basis[:, 7] = np.cos(4 * np.pi * z)
    basis[:, 8] = np.sin(4 * np.pi * z)
    
    # k=5 (Cosine only to fit 10 params)
    basis[:, 9] = np.cos(5 * np.pi * z)
    
    # Linear combination
    return np.sum(basis * params, axis=1)
# EVOLVE-BLOCK-END
