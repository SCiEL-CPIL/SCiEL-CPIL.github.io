import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    # Design: Use Gaussian quadrature nodes for the weight function e^{-x} x^{-1/2}.
    # These are the roots of the generalized Laguerre polynomial L_12^{(-1/2)}(x).
    # This choice is deterministic, fast, and theoretically grounded in the
    # orthogonality properties of the basis functions used in the problem.
    try:
        # n=12, alpha=-0.5
        # roots_genlaguerre returns (x, w)
        x, w = roots_genlaguerre(12, -0.5)
        # Ensure they are positive and distinct (they are by definition)
        # Convert to list of floats
        z_list = [float(xi) for xi in x]
        return z_list
    except Exception:
        # Fallback to a simple geometric progression if scipy fails
        return [0.5 * (1.5 ** i) for i in range(12)]
