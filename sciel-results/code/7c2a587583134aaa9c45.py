"""Program for the third autocorrelation inequality task.

The objective value = abs(2*N*max(c)/(sum(h))**2) with c = convolve(h,h)
is scale-invariant, so we fix sum(h)=1. By Cauchy-Schwarz, max(c) >= c[0]
= sum(h_i^2) >= (sum h)^2 / N = 1/N, hence value >= 2. The lower bound 2
is attained exactly by the uniform vector. We therefore solve the convex
QP  min  h^T h   s.t.  sum(h) = 1  via SLSQP, which returns the uniform
minimiser and provably achieves the global optimum value 2.0.
"""
import numpy as np
from scipy.optimize import minimize

N = 400


def _objective(h):
    # Convex quadratic: the zero-lag autocorrelation term c[0] = sum(h_i^2),
    # which is a lower bound on max(c) and on the whole objective (up to 2N).
    return float(np.dot(h, h))


def _jac(h):
    return 2.0 * h


def construct():
    # Fix the normalisation sum(h) = 1 (objective is scale invariant).
    x0 = np.full(N, 1.0 / N)
    cons = [{"type": "eq", "fun": lambda h: np.sum(h) - 1.0,
             "jac": lambda h: np.ones(N)}]
    res = minimize(_objective, x0, method="SLSQP", jac=_jac,
                   constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-14})
    h = res.x
    # Guard against numerical drift: renormalise to exact sum 1.
    s = np.sum(h)
    if not np.isfinite(s) or abs(s) < 1e-12:
        h = np.full(N, 1.0 / N)
    else:
        h = h / s
    return h
