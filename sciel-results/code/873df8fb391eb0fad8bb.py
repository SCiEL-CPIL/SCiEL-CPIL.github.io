"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Use orthogonal Legendre polynomials on the domain [-3, 3] mapped to [-1, 1].
    # This avoids the ill-conditioning of raw monomials (x^2, x^4, etc.) and
    # the numerical instability of exp(x).
    # The basis is bounded and smooth, allowing BFGS to converge reliably.
    x = np.asarray(x, dtype=float)
    # Map x from [-3, 3] to [-1, 1] for standard Legendre polynomials
    u = x / 3.0
    # Compute Legendre polynomials of degree 0 to 9
    # legvander returns a matrix where column k is P_k(u)
    basis = np.polynomial.legendre.legvander(u, 9)
    return basis @ params
# EVOLVE-BLOCK-END
