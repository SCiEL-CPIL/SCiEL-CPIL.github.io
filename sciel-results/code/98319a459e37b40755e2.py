"""Program for the third autocorrelation inequality task.

Formulates the problem as minimizing the squared L2 norm of the autoconvolution
$c = h * h$ subject to $\sum h_i = 1$. Uses FFT-based gradient for efficiency.
"""
import numpy as np
from scipy.optimize import minimize

N = 400


def _autoconv_sq_norm(h):
    """Compute sum of squares of the full autoconvolution."""
    c = np.convolve(h, h)
    return float(np.sum(c ** 2))


def _autoconv_sq_norm_grad(h):
    """Compute gradient of sum of squares of full autoconvolution.
    
    J(h) = ||h * h||_2^2 where * denotes full linear convolution.
    
    Using the identity: grad J = 4 * fold(h * (h * h))
    where fold takes the central N elements of the length-3N-1 convolution.
    """
    # Full autoconvolution: length 2N-1
    c = np.convolve(h, h)
    # Now compute h * c (full convolution): length (2N-1) + N - 1 = 3N-1
    hc = np.convolve(h, c)
    # fold: take the central N elements
    # hc has length 3N-1. Indices 0..3N-2.
    # The central N elements are at indices N-1 .. 2N-2
    grad = 4.0 * hc[N-1:2*N-1]
    return grad


def construct():
    x0 = np.full(N, 1.0 / N)
    cons = [{"type": "eq", "fun": lambda h: np.sum(h) - 1.0,
             "jac": lambda h: np.ones(N)}]
    res = minimize(_autoconv_sq_norm, x0, method="SLSQP", 
                   jac=_autoconv_sq_norm_grad,
                   constraints=cons,
                   options={"maxiter": 200, "ftol": 1e-14})
    h = res.x
    s = np.sum(h)
    if not np.isfinite(s) or abs(s) < 1e-12:
        h = np.full(N, 1.0 / N)
    else:
        h = h / s
    return h
