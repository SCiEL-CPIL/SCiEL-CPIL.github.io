import numpy as np
from scipy.optimize import minimize
import time

N = 400

def value_from_h(h):
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    conv = np.convolve(h, h)
    max_c = np.max(conv)
    return abs(2 * N * max_c / (s * s))

def construct():
    start_time = time.time()
    deadline = 140.0
    
    best_h = np.ones(N)
    best_val = value_from_h(best_h)
    
    # Parameterized families
    def h_linear(c):
        return 1.0 + c * np.linspace(0, 1, N)
        
    def h_sine(c, f):
        return 1.0 + c * np.sin(2 * np.pi * f * np.arange(N) / N)
        
    def h_cosine(c, f):
        return 1.0 + c * np.cos(2 * np.pi * f * np.arange(N) / N)

    # Optimize linear
    res = minimize(lambda c: value_from_h(h_linear(c[0])), [0.0], method='Nelder-Mead')
    if res.fun < best_val:
        best_val = res.fun
        best_h = h_linear(res.x[0])

    # Optimize sine with a few frequencies
    for f in [1, 2, 5, 10, 50]:
        res = minimize(lambda c: value_from_h(h_sine(c[0], f)), [0.5], method='Nelder-Mead')
        if res.fun < best_val:
            best_val = res.fun
            best_h = h_sine(res.x[0], f)
            
        res = minimize(lambda c: value_from_h(h_cosine(c[0], f)), [0.5], method='Nelder-Mead')
        if res.fun < best_val:
            best_val = res.fun
            best_h = h_cosine(res.x[0], f)

    # If time permits, do a full random search with local opt
    while time.time() < deadline - 10:
        h0 = np.random.normal(0, 1, N) + 1.0
        # Quick local search using coordinate descent or just evaluate
        # Let's try a few steps of gradient descent on the raw value? No, non-diff.
        # Just evaluate random and keep best?
        val = value_from_h(h0)
        if val < best_val:
            best_val = val
            best_h = h0
            
        # Try a small local perturbation
        h_pert = h0 + 0.1 * np.random.normal(0, 1, N)
        val_p = value_from_h(h_pert)
        if val_p < best_val:
            best_val = val_p
            best_h = h_pert

    return best_h
