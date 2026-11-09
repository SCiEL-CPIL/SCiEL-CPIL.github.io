"""Multi-scale coordinate descent + CMA-ES hybrid optimizer for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-scale coordinate descent for early progress + CMA-ES for refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        x_best = None
        f_best = np.inf
        calls_made = 0
        history_x = []
        history_f = []

        def evaluate(x):
            nonlocal x_best, f_best, calls_made
            if calls_made >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            fx = float(func(x))
            calls_made += 1
            history_x.append(x.copy())
            history_f.append(fx)
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
            return fx

        def fit_quadratic_model(X, Y):
            n, d = X.shape
            num_terms = 1 + d + d * (d + 1) // 2
            M = np.zeros((n, num_terms))
            M[:, 0] = 1.0
            M[:, 1:1+d] = X
            idx = 1 + d
            for i in range(d):
                for j in range(i, d):
                    if i == j:
                        M[:, idx] = X[:, i] ** 2
                    else:
                        M[:, idx] = 2.0 * X[:, i] * X[:, j]
                    idx += 1
            beta, _, _, _ = np.linalg.lstsq(M, Y, rcond=None)
            return beta

        def predict_quadratic(x, beta):
            d = self.dim
            num_terms = 1 + d + d * (d + 1) // 2
            phi = np.zeros(num_terms)
            phi[0] = 1.0
            phi[1:1+d] = x
            idx = 1 + d
            for i in range(d):
                for j in range(i, d):
                    if i == j:
                        phi[idx] = x[i] ** 2
                    else:
                        phi[idx] = 2.0 * x[i] * x[j]
                    idx += 1
            return np.dot(beta, phi)

        def surrogate_minimize(beta, x_start):
            def obj(x):
                return predict_quadratic(x, beta)
            bounds = [(-5, 5) for _ in range(dim)]
            try:
                res = minimize(obj, x_start, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100})
                return res.x
            except Exception:
                return x_start

        # Phase 1: Initial Sampling
        rng = np.random.default_rng(42)
        n_init = min(20, budget)
        for _ in range(n_init):
            if calls_made >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            evaluate(x)

        # Phase 2: Surrogate-Based Optimization
        model_size = 30
        
        while calls_made < budget and time.time() < deadline:
            if len(history_x) < model_size:
                idxs = range(len(history_x))
            else:
                sorted_idxs = np.argsort(history_f)
                idxs = sorted_idxs[:model_size]
            
            X_model = np.array([history_x[i] for i in idxs])
            Y_model = np.array([history_f[i] for i in idxs])
            
            beta = fit_quadratic_model(X_model, Y_model)
            
            x_start = x_best if x_best is not None else np.zeros(dim)
            x_next = surrogate_minimize(beta, x_start)
            
            f_next = evaluate(x_next)
            
            if calls_made % 10 == 0:
                x_rand = rng.uniform(-5, 5, dim)
                evaluate(x_rand)

        if x_best is None:
            x_best = np.zeros(dim)
            f_best = evaluate(x_best)

        return x_best, f_best
# EVOLVE-BLOCK-END
