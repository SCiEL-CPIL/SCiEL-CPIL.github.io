import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Population-based DE/rand/1 with adaptive F and a Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        import time
        start = time.time()
        deadline = 280.0
        max_calls = int(self.budget)
        dim = int(self.dim)
        calls = 0

        def f(x):
            nonlocal calls
            if calls >= max_calls:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            calls += 1
            try:
                val = func(x)
            except Exception:
                return float('inf')
            if not np.isfinite(val):
                return float('inf')
            return float(val)

        rng = np.random.default_rng(0)
        N = 8
        pop = rng.uniform(-5.0, 5.0, (N, dim))
        fpop = np.empty(N)
        for i in range(N):
            fpop[i] = f(pop[i])
        best_idx = int(np.argmin(fpop))
        best_x = pop[best_idx].copy()
        best_f = float(fpop[best_idx])
        if not np.isfinite(best_f):
            best_x = np.zeros(dim)
            best_f = f(best_x)

        F = 0.8
        CR = 0.5
        while calls + N <= max_calls and time.time() < deadline - 10.0:
            gen_success = 0
            for i in range(N):
                a, b = rng.choice(N, size=2, replace=False)
                v = pop[a] + F * (pop[b] - pop[i])
                r = rng.random(dim)
                jrand = int(rng.integers(dim))
                trial = np.where(r < CR, v, pop[i])
                trial[jrand] = v[jrand]
                trial = np.clip(trial, -5.0, 5.0)
                ft = f(trial)
                if ft < fpop[i]:
                    fpop[i] = ft
                    pop[i] = trial
                    gen_success += 1
                    if ft < best_f:
                        best_f = ft
                        best_x = trial.copy()
            rate = gen_success / float(N)
            F = float(np.clip(0.4 + 0.6 * rate, 0.4, 0.9))

        remaining = max_calls - calls
        if remaining > 5 and time.time() < deadline - 5.0:
            try:
                res = minimize(f, best_x, method='Nelder-Mead',
                               options={'maxiter': 200,
                                        'maxfev': int(remaining),
                                        'xatol': 1e-9,
                                        'fatol': 1e-11,
                                        'adaptive': True})
                if np.isfinite(res.fun) and res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = np.clip(np.asarray(res.x, dtype=float), -5.0, 5.0)
            except Exception:
                pass

        return best_x, best_f
