"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def _fourier_basis(x: np.ndarray, num_harmonics: int = 5) -> np.ndarray:
    """
    Compute Fourier basis functions: cos(w_i x) and sin(w_i x) for i=1..num_harmonics.
    Plus a constant term if needed, but we have 10 params for 2*num_harmonics terms.
    So we use 5 cos and 5 sin.
    Frequencies are chosen to cover the domain [-3, 3] with period 6.
    w_i = i * pi / 3.
    """
    n = len(x)
    basis = np.zeros((n, 2 * num_harmonics))
    
    for i in range(1, num_harmonics + 1):
        w = i * np.pi / 3.0
        basis[:, 2*(i-1)] = np.cos(w * x)
        basis[:, 2*(i-1)+1] = np.sin(w * x)
        
    return basis


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fit a Fourier series with 5 harmonics (10 params) to the data.
    params[0] = a_1 (cos), params[1] = b_1 (sin)
    ...
    params[8] = a_5 (cos), params[9] = b_5 (sin)
    """
    basis = _fourier_basis(x, num_harmonics=5)
    return basis @ params[:10]
# EVOLVE-BLOCK-END
