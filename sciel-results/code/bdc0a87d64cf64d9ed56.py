"""Rewritten program for Erdos minimum overlap task."""
import numpy as np
from scipy.optimize import minimize_scalar, brentq

N = 100

def _max_overlap(h):
    """Compute the maximum overlap of h with shifted 1-h over all circular shifts."""
    one_minus_h = 1.0 - h
    max_ov = 0.0
    for s in range(N):
        shifted = np.roll(one_minus_h, s)
        ov = np.dot(h, shifted)
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _construct_h(c, tau):
    """Construct h from center c and width parameter tau."""
    i = np.arange(N)
    # h[i] = 0.5 + 0.5 * tanh((i - c) / tau)
    h = 0.5 + 0.5 * np.tanh((i - c) / tau)
    return h

def _sum_h(c, tau):
    return np.sum(_construct_h(c, tau))

def _find_c_for_tau(tau):
    """Find c such that sum(h) = N/2."""
    target = N / 2.0
    
    def f(c):
        return _sum_h(c, tau) - target
    
    try:
        c_opt = brentq(f, 0, N, xtol=1e-8)
    except ValueError:
        c_opt = N / 2.0
        
    return c_opt

def _objective_tau(tau):
    """Objective function for tau optimization."""
    if tau <= 1e-6:
        tau = 1e-6
    c = _find_c_for_tau(tau)
    h = _construct_h(c, tau)
    return _max_overlap(h)

def construct():
    """
    Construct the sequence of step heights.
    """
    res = minimize_scalar(_objective_tau, bounds=(0.1, N/2), method='bounded', options={'xatol': 1e-6})
    best_tau = res.x
    best_c = _find_c_for_tau(best_tau)
    h = _construct_h(best_c, best_tau)
    
    return h

# EVOLVE-BLOCK-START
# (The construct function is defined above)
# EVOLVE-BLOCK-END
