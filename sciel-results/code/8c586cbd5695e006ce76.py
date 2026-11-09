"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio of (1+1)-ES runs with adaptive restarts and a CMA-ES-like phase."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        dim = self.dim
        total_budget = self.budget
        calls_used = 0

        best_x = np.full(dim, np.nan)
        best_f = np.inf

        def try_update(x, f):
            nonlocal best_x, best_f, calls_used
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def es_run(rng, x0, fx0, max_calls, sigma0=2.0):
            nonlocal calls_used, best_x, best_f
            x = x0.copy()
            fx = fx0
            sigma = sigma0
            no_improve = 0
            patience = 40 + dim * 5

            while calls_used < max_calls and (time.time() < deadline):
                if calls_used >= total_budget:
                    break
                if time.time() > deadline - 5.0:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.15
                    no_improve = 0
                    try_update(x, fx)
                else:
                    sigma *= 0.92
                    no_improve += 1

                if no_improve > patience:
                    break

            return x, fx

        # Phase 1: Multiple short (1+1)-ES runs from diverse starts
        rng = np.random.default_rng(42)

        # Generate diverse starting points
        starts = []
        # Origin and axis-aligned points
        starts.append(np.zeros(dim))
        for axis in range(dim):
            for sign in [-1, 1]:
                p = np.zeros(dim)
                p[axis] = sign * 3.0
                starts.append(p)
        # Random points
        for _ in range(10):
            starts.append(rng.uniform(-4, 4, dim))

        # Deduplicate (approximate)
        unique_starts = [starts[0]]
        for s in starts[1:]:
            if all(np.linalg.norm(s - u) > 0.5 for u in unique_starts):
                unique_starts.append(s)

        # Allocate budget: first few calls to evaluate starts, then ES runs
        n_starts = len(unique_starts)
        # Evaluate each start
        for s in unique_starts:
            if calls_used >= total_budget:
                break
            if time.time() > deadline - 10.0:
                break
            fs = func(s)
            calls_used += 1
            try_update(s, fs)

        # Run ES from each start with limited budget
        # Give more budget to better starts
        # Sort starts by their initial function value
        start_vals = []
        for s in unique_starts:
            # We already evaluated, but we need fx for each; re-evaluate would waste budget
            # Instead, track during evaluation
            pass

        # Let's restructure: evaluate starts and track
        start_fxs = {}
        for s in unique_starts:
            key = tuple(s.tolist())
            start_fxs[key] = None

        # Re-do: evaluate and store
        eval_results = []
        for s in unique_starts:
            if calls_used >= total_budget:
                break
            if time.time() > deadline - 10.0:
                break
            fs = func(s)
            calls_used += 1
            eval_results.append((s, fs))
            try_update(s, fs)

        # Sort by function value
        eval_results.sort(key=lambda t: t[1])

        # Run ES from top starts
        n_es_runs = min(len(eval_results), 5)
        budget_per_es = max(10, (total_budget - calls_used - 20) // max(n_es_runs, 1))

        for i in range(n_es_runs):
            if calls_used >= total_budget:
                break
            if time.time() > deadline - 10.0:
                break
            s, fs = eval_results[i]
            es_run(rng, s, fs, calls_used + budget_per_es)

        # Phase 2: If we have budget left, do a local search around the best point
        if calls_used < total_budget and time.time() < deadline - 10.0:
            if not np.isnan(best_x).any():
                # Fine-grained local search
                sigma = 0.3
                x = best_x.copy()
                fx = best_f
                for _ in range(min(100, total_budget - calls_used)):
                    if calls_used >= total_budget:
                        break
                    if time.time() > deadline - 5.0:
                        break
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    calls_used += 1
                    if fy < fx:
                        x, fx = y, fy
                        try_update(x, fx)
                        sigma *= 1.1
                    else:
                        sigma *= 0.9

        # Phase 3: If still budget, try CMA-ES-like with covariance
        if calls_used < total_budget and time.time() < deadline - 15.0 and not np.isnan(best_x).any():
            # Simple CMA-ES with population size 4+dim
            pop_size = 4 + dim
            c = 1 / (dim + 3)
            d = 1 + 2 * max(0, np.sqrt((dim - 1) / (dim + 3)) - 1)
            sigma = 0.5
            mean = best_x.copy()
            C = np.eye(dim)
            pc = np.zeros(dim)
            cs = 1 / (dim + 4)
            # Simplified: just do a population-based search
            for gen in range(min(50, (total_budget - calls_used) // pop_size)):
                if calls_used >= total_budget:
                    break
                if time.time() > deadline - 5.0:
                    break
                # Generate population
                pop = []
                for j in range(pop_size):
                    if calls_used >= total_budget:
                        break
                    y = np.clip(mean + sigma * rng.multivariate_normal(np.zeros(dim), C), -5, 5)
                    fy = func(y)
                    calls_used += 1
                    pop.append((y, fy))
                if not pop:
                    break
                pop.sort(key=lambda t: t[1])
                # Update mean
                new_mean = np.zeros(dim)
                for y, fy in pop[:pop_size // 2]:
                    new_mean += y
                new_mean /= (pop_size // 2)
                # Update covariance
                C = 0.99 * C
                for y, fy in pop[:pop_size // 2]:
                    dvec = y - mean
                    C += 0.01 * np.outer(dvec, dvec)
                # Ensure symmetry
                C = (C + C.T) / 2
                mean = new_mean
                sigma *= 0.95
                try_update(mean, func(mean))
                calls_used += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
