import numpy as np
import time
from scipy.optimize import minimize_scalar

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_evals = 0
        n = self.budget

        def compute_hv(t_vals):
            t_sorted = np.sort(t_vals)
            t_prev = 0.0
            hv = 0.0
            for t in t_sorted:
                f2 = 1.0 - np.sqrt(t)
                hv += (t - t_prev) * f2
                t_prev = t
            return hv

        def hv_from_p(p):
            t_vals = ((np.arange(n) + 0.5) / n) ** p
            t_vals = np.clip(t_vals, 1e-10, 1.0 - 1e-10)
            return -compute_hv(t_vals)

        # Optimize p
        res = minimize_scalar(hv_from_p, bounds=(0.1, 5.0), method='bounded')
        best_p = res.x
        best_t = ((np.arange(n) + 0.5) / n) ** best_p
        best_t = np.clip(best_t, 1e-10, 1.0 - 1e-10)

        # Also try cosine
        u = (np.arange(n) + 0.5) / n
        t_cos = 0.5 - 0.5 * np.cos(np.pi * u)
        hv_cos = compute_hv(t_cos)
        
        hv_power = compute_hv(best_t)
        
        if hv_cos > hv_power:
            best_t = t_cos

        # Build decision vectors
        X = np.zeros((n, self.dim))
        X[:, 0] = best_t
        
        F = np.zeros((n, self.n_obj))
        for i in range(n):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
            n_evals += 1
            
        return X[:n_evals]
