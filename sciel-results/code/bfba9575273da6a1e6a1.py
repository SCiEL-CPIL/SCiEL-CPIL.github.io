"""Program for the third autocorrelation inequality task.

This program formulates the problem as a Quadratic Program (QP) minimizing
the squared L2 norm of the autoconvolution $c = h * h$ subject to the linear
constraint $\sum h_i = 1$. This approach aims to suppress the global peak
magnitude by spreading the autocorrelation energy.
"""
import numpy as np
from scipy.optimize import minimize

N = 400


def _objective(h):
    c = np.convolve(h, h)
    return float(np.dot(c, c))


def construct():
    # Fix the normalisation sum(h) = 1 (objective is scale invariant).
    x0 = np.full(N, 1.0 / N)
    cons = [{"type": "eq", "fun": lambda h: np.sum(h) - 1.0,
             "jac": lambda h: np.ones(N)}]
    
    try:
        res = minimize(_objective, x0, method="SLSQP",
                       constraints=cons,
                       options={"maxiter": 100, "ftol": 1e-6})
        h = res.x
        s = np.sum(h)
        if not np.isfinite(s) or abs(s) < 1e-12:
            h = np.full(N, 1.0 / N)
        else:
            h = h / s
    except Exception:
        h = np.full(N, 1.0 / N)

    # Evaluate the actual task metric to ensure we don't return a worse result.
    # The uniform vector achieves the theoretical lower bound of 2.0.
    c = np.convolve(h, h)
    val = abs(2 * N * np.max(c) / (np.sum(h) ** 2))
    
    if val > 2.0 + 1e-9:
        h = np.full(N, 1.0 / N)
        
    return h
