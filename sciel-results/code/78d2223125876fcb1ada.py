"""Symbolic regression using a diverse basis of polynomials, trigonometric, and Gaussian terms."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes y = sum(params[i] * basis_i(x)) for i in 0..9.

    Basis functions:
    - Chebyshev polynomials T0..T5 for stable global polynomial approximation
    - Three Gaussian RBFs centered at -1.5, 0, 1.5 for localized features

    Chebyshev polynomials are orthogonal on [-1,1]; we map x in [-3,3] to [-1,1]
    to improve conditioning. This gives BFGS a better-conditioned design matrix
    while retaining local flexibility via RBFs.
    """
    x = np.asarray(x, dtype=float)

    # Map x from [-3, 3] to [-1, 1] for Chebyshev polynomials
    t = x / 3.0

    # Chebyshev polynomials T0..T5 via recurrence
    # T0 = 1, T1 = t, T_{n} = 2*t*T_{n-1} - T_{n-2}
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    T4 = 2.0 * t * T3 - T2
    T5 = 2.0 * t * T4 - T3

    # Gaussian RBFs with moderate width for local flexibility
    sigma_sq = 2 * 0.8**2
    centers = np.array([-1.5, 0.0, 1.5])

    # Compute all three Gaussians efficiently
    diffs = x[None, :] - centers[:, None]  # shape (3, n)
    gaussians = np.exp(-diffs**2 / sigma_sq)  # shape (3, n)

    # Linear combination: 6 Chebyshev + 3 Gaussian = 9 terms
    # We have 10 params, so add one more term: a scaled Chebyshev T6
    T6 = 2.0 * t * T5 - T4

    y = (params[0] * T0 +
         params[1] * T1 +
         params[2] * T2 +
         params[3] * T3 +
         params[4] * T4 +
         params[5] * T5 +
         params[6] * T6 +
         params[7] * gaussians[0] +
         params[8] * gaussians[1] +
         params[9] * gaussians[2])

    return y
# EVOLVE-BLOCK-END
