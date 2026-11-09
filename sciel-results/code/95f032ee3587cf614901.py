"""Symbolic regression with a Fourier series basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fit a 10-term Fourier series (constant + 4 sine + 4 cosine terms) to the data.
    
    The input x is in [0.1, 6]. We use fixed orthogonal frequencies over this domain.
    The basis functions are:
      1, sin(2*pi*k*x/L), cos(2*pi*k*x/L) for k=1,2,3,4
    where L = 6.0 - 0.1 = 5.9 is the domain length.
    
    This gives 1 + 4 + 4 = 9 terms. We add one more term (sin with half-frequency)
    to make 10 terms total. Actually, let's use:
      1, cos(2*pi*x/L), sin(2*pi*x/L), cos(2*2*pi*x/L), sin(2*2*pi*x/L),
      cos(2*3*pi*x/L), sin(2*3*pi*x/L), cos(2*4*pi*x/L), sin(2*4*pi*x/L),
      and one more... let's just use 10 terms: 1 + 4 pairs = 9, plus one more cosine
      at a different frequency, or simply use 5 pairs: 1 + 5*2 = 11, too many.
    
    Let me reconsider: 10 parameters. Let's use:
      param[0] * 1
      param[1] * cos(2*pi*1*x/L)
      param[2] * sin(2*pi*1*x/L)
      param[3] * cos(2*pi*2*x/L)
      param[4] * sin(2*pi*2*x/L)
      param[5] * cos(2*pi*3*x/L)
      param[6] * sin(2*pi*3*x/L)
      param[7] * cos(2*pi*4*x/L)
      param[8] * sin(2*pi*4*x/L)
      param[9] * cos(2*pi*5*x/L)
    
    That's 10 terms.
    """
    if len(x) == 0:
        return np.array([])
    
    x_min = 0.1
    x_max = 6.0
    L = x_max - x_min
    
    n = len(x)
    n_terms = 10
    
    basis = np.zeros((n, n_terms))
    
    # Constant term
    basis[:, 0] = 1.0
    
    # Frequencies: k = 1, 2, 3, 4 for sin/cos pairs, and k=5 for cos only
    # This gives 1 + 4*2 + 1 = 10 terms
    
    for k in range(1, 5):
        freq = 2.0 * np.pi * k / L
        basis[:, 2 * k - 1] = np.cos(freq * x)
        basis[:, 2 * k] = np.sin(freq * x)
    
    # 10th term: cos at k=5
    freq5 = 2.0 * np.pi * 5 / L
    basis[:, 9] = np.cos(freq5 * x)
    
    # Normalize each basis function to have unit RMS over the domain
    # for better conditioning
    x_grid = np.linspace(x_min, x_max, 1000)
    L_grid = L
    
    basis_grid = np.zeros((len(x_grid), n_terms))
    basis_grid[:, 0] = 1.0
    for k in range(1, 5):
        freq = 2.0 * np.pi * k / L_grid
        basis_grid[:, 2 * k - 1] = np.cos(freq * x_grid)
        basis_grid[:, 2 * k] = np.sin(freq * x_grid)
    freq5 = 2.0 * np.pi * 5 / L_grid
    basis_grid[:, 9] = np.cos(freq5 * x_grid)
    
    rms = np.sqrt(np.mean(basis_grid ** 2, axis=0))
    rms = np.where(rms < 1e-12, 1.0, rms)
    
    basis_normalized = basis / rms
    
    return basis_normalized @ params
# EVOLVE-BLOCK-END
