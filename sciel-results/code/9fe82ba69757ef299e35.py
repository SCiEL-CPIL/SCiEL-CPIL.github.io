import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Tensor product of 1D Chebyshev-like basis functions.
    Uses 5 product terms with fixed frequency pairs, linear in params.
    This captures separable and mildly non-separable smooth functions.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # 1D basis functions on [-1, 1]
    # Using cos(k * arccos(t)) = T_k(t), but for simplicity use polynomials
    # Actually, let's use a mix of low-order terms that are well-conditioned

    # Basis functions for x
    bx0 = np.ones_like(x)
    bx1 = x
    bx2 = x * x
    bx3 = x * x * x
    bx4 = x * x * x * x

    # Basis functions for y
    by0 = np.ones_like(y)
    by1 = y
    by2 = y * y
    by3 = y * y * y
    by4 = y * y * y * y

    # Tensor product terms (5 pairs, using 10 params as weights)
    # Choose diverse frequency combinations
    term0 = p0 * bx0 * by0  # constant
    term1 = p1 * bx1 * by1  # linear cross
    term2 = p2 * bx2 * by0  # x^2
    term3 = p3 * bx0 * by2  # y^2
    term4 = p4 * bx1 * by0  # x
    term5 = p5 * bx0 * by1  # y
    term6 = p6 * bx2 * by2  # x^2*y^2
    term7 = p7 * bx3 * by1  # x^3*y
    term8 = p8 * bx1 * by3  # x*y^3
    term9 = p9 * bx4 * by4  # x^4*y^4 (high order)

    return term0 + term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9
