"""Rewritten for the second autocorrelation inequality; maximizes ||f||_2^2 / (||f||_1 * ||f*f||_inf)
over 50 nonnegative step heights on [-1/4,1/4]."""

import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50

def _objective(x):
    # x is the array of heights (nonnegative)
    if np.any(x < 0):
        return 1e12
    x = np.maximum(x, 0.0)
    l1 = np.sum(x)
    if l1 < 1e-15:
        return 1e12
    # autoconvolution (full)
    conv = np.convolve(x, x, mode='full')
    sup = np.max(conv)
    if sup < 1e-15:
        return 1e12
    l2_sq = np.sum(x ** 2)
    return -(l2_sq / (l1 * sup))

def _score_from_obj(val):
    return -val

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s (and well before 150s kill)

    best_val = -1.0
    best_x = np.ones(N)
    best_obj = _objective(best_x)

    def try_candidate(x):
        nonlocal best_obj, best_x
        obj = _objective(x)
        if obj < best_obj:
            best_obj = obj
            best_x = x.copy()

    # Seed with uniform
    try_candidate(best_x)

    # Several local optimizations from diverse starts
    np.random.seed(42)
    starts = [
        np.ones(N),
        np.random.rand(N),
        np.random.rand(N) * 0.1,
        np.zeros(N),
        np.zeros(N)
    ]
    # Fix bad starts list
    starts = [
        np.ones(N),
        np.random.rand(N),
        np.random.rand(N) * 0.1 + 0.01,
        np.array([1.0] + [0.0]*(N-1)),
        np.array([0.0]*(N-1) + [1.0]),
        np.linspace(0.0, 1.0, N),
        np.linspace(1.0, 0.0, N),
    ]
    # Add random Gaussian blobs
    for _ in range(5):
        center = np.random.randint(0, N)
        width = np.random.uniform(2.0, 10.0)
        blob = np.exp(-0.5 * ((np.arange(N) - center) / width) ** 2)
        blob /= np.max(blob)
        starts.append(blob)

    for s in starts:
        s = np.maximum(s, 0.0)
        # Try SLSQP
        res = minimize(_objective, s, method='SLSQP',
                       bounds=[(0.0, None)] * N,
                       options={'maxiter': 500, 'ftol': 1e-12})
        try_candidate(res.x)
        if time.time() > deadline:
            break

    # Differential evolution for global exploration
    if time.time() < deadline - 20.0:
        res_de = differential_evolution(_objective,
                                        bounds=[(0.0, 10.0)] * N,
                                        seed=42,
                                        maxiter=50,
                                        tol=1e-8,
                                        popsize=15,
                                        polish=True)
        try_candidate(res_de.x)

    # Final polish from best so far
    if time.time() < deadline - 5.0:
        res = minimize(_objective, best_x, method='SLSQP',
                       bounds=[(0.0, None)] * N,
                       options={'maxiter': 2000, 'ftol': 1e-14})
        try_candidate(res.x)

    return [float(v) for v in best_x]
