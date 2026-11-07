"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier series basis with 10 parameters.
    Domain: x in [0.1, 6]
    
    We use sine and cosine terms with frequencies scaled to the domain.
    The basis is:
      1, cos(w*x), sin(w*x), cos(2*w*x), sin(2*w*x), ..., cos(4*w*x), sin(4*w*x)
    
    where w = 2*pi / (6 - 0.1) ≈ 2*pi / 5.9
    
    This gives 1 + 2*4 = 9... wait, we need 10 params.
    Let's use 1 + 2*4 = 9, so we need one more. Let's use 5 frequencies:
    1, cos(w*x), sin(w*x), cos(2w*x), sin(2w*x), cos(3w*x), sin(3w*x), 
    cos(4w*x), sin(4w*x), cos(5w*x)
    That's 1 + 2*4 + 1 = 10. 
    
    Actually, let me reconsider. A cleaner approach:
    Use constant + 4 full frequencies (sin+cos) + 1 extra cos = 10
    Or: constant + 5 full frequencies would be 11, too many.
    So: constant + 4 full frequencies (8) + 1 extra = 10.
    
    Let's do: 1, cos(k1*x), sin(k1*x), cos(k2*x), sin(k2*x), cos(k3*x), sin(k3*x), 
              cos(k4*x), sin(k4*x), cos(k5*x)
    where ki = i * 2*pi / L, L = 5.9
    """
    L = 6.0 - 0.1  # 5.9
    w = 2.0 * np.pi / L
    
    # Frequencies
    k1 = w
    k2 = 2.0 * w
    k3 = 3.0 * w
    k4 = 4.0 * w
    k5 = 5.0 * w
    
    # Build basis matrix (n_points, 10)
    # Columns: 1, cos(k1*x), sin(k1*x), cos(k2*x), sin(k2*x), 
    #           cos(k3*x), sin(k3*x), cos(k4*x), sin(k4*x), cos(k5*x)
    
    c1 = np.cos(k1 * x)
    s1 = np.sin(k1 * x)
    c2 = np.cos(k2 * x)
    s2 = np.sin(k2 * x)
    c3 = np.cos(k3 * x)
    s3 = np.sin(k3 * x)
    c4 = np.cos(k4 * x)
    s4 = np.sin(k4 * x)
    c5 = np.cos(k5 * x)
    
    # Stack into basis matrix
    basis = np.column_stack([
        np.ones_like(x),
        c1, s1,
        c2, s2,
        c3, s3,
        c4, s4,
        c5
    ])
    
    return basis @ params
# EVOLVE-BLOCK-END
