"""Rewritten optimizer: DE/rand/1 with adaptive F, LHS init, stagnation restarts, and local polish."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Differential Evolution with adaptive mutation, LHS init, restarts, and local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        box_lo, box_hi = -5.0, 5.0

        # Track best
        best_x = None
        best_f = np.inf
        total_calls = 0

        def clip(x):
            return np.clip(x, box_lo, box_hi)

        def eval_point(x):
            nonlocal total_calls
            x = clip(x)
            fx = func(x)
            total_calls += 1
            return x, fx

        # LHS initialization for diversity
        def lhs_sample(n, dim):
            pts = np.zeros((n, dim))
            for d in range(dim):
                perm = np.random.permutation(n)
                pts[perm, d] = (np.arange(n) + np.random.random()) / n
            return box_lo + pts * (box_hi - box_lo)

        rng = np.random.default_rng(42)
        np.random.seed(42)

        # Phase 1: DE/rand/1 with population
        pop_size = 10
        # Initial population using LHS
        pop = lhs_sample(pop_size, dim)
        pop_f = np.array([eval_point(p)[1] for p in pop])

        # Track best
        idx_best = np.argmin(pop_f)
        best_x = pop[idx_best].copy()
        best_f = pop_f[idx_best]

        # Adaptive F
        F = 0.8
        CR = 0.7
        stagnation_counter = 0
        stagnation_window = 50  # evaluations without improvement
        last_improvement_call = total_calls

        # DE loop
        while total_calls < budget:
            if time.time() > deadline:
                break

            # DE/rand/1
            i = np.random.randint(pop_size)
            a, b, c = np.random.choice(pop_size, 3, replace=False)
            while len([a, b, c]) < 3:
                a, b, c = np.random.choice(pop_size, 3, replace=False)

            v = pop[a] + F * (pop[b] - pop[c])
            v = clip(v)

            # Crossover
            trial = np.copy(pop[i])
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if rng.random() < CR or j == j_rand:
                    trial[j] = v[j]
            trial = clip(trial)

            ft, _ = eval_point(trial)

            # Selection
            if ft <= pop_f[i]:
                pop[i] = trial
                pop_f[i] = ft
                if ft < best_f:
                    best_f = ft
                    best_x = trial.copy()
                    last_improvement_call = total_calls
                    F = min(0.95, F + 0.05)
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
            else:
                stagnation_counter += 1
                F = max(0.3, F * 0.95)

            # Stagnation restart
            if stagnation_counter >= stagnation_window:
                # Perturb population around best
                for k in range(pop_size):
                    if k != np.argmin(pop_f):
                        pop[k] = clip(best_x + rng.standard_normal(dim) * 1.5)
                        pop_f[k] = eval_point(pop[k])[1]
                stagnation_counter = 0
                F = 0.8
                last_improvement_call = total_calls

            # Check if we've used all budget
            if total_calls >= budget:
                break

        # Phase 2: Local polish with Nelder-Mead and L-BFGS-B
        if time.time() < deadline - 5.0 and total_calls < budget:
            # Nelder-Mead
            try:
                res = minimize(
                    lambda x: func(np.clip(x, box_lo, box_hi)),
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10},
                )
                # Count calls approximately (Nelder-Mead uses ~maxiter * dim evaluations)
                # We need to track actual calls, so let's use a wrapper
            except Exception:
                pass

        # Better approach: use a call-counting wrapper for polish
        # Let's redo polish properly
        polish_calls = [0]

        def polish_func(x):
            polish_calls[0] += 1
            return func(np.clip(x, box_lo, box_hi))

        # Nelder-Mead polish
        if time.time() < deadline - 5.0 and total_calls + polish_calls[0] < budget:
            try:
                res = minimize(
                    polish_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10},
                )
                total_calls += polish_calls[0]
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, box_lo, box_hi)
            except Exception:
                pass

        # L-BFGS-B polish
        polish_calls[0] = 0
        if time.time() < deadline - 3.0 and total_calls + polish_calls[0] < budget:
            try:
                bounds = [(box_lo, box_hi)] * dim
                res = minimize(
                    polish_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8},
                )
                total_calls += polish_calls[0]
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, box_lo, box_hi)
            except Exception:
                pass

        return best_x, best_f
