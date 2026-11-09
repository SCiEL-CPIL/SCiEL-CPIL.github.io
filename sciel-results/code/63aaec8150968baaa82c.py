"""Combined optimizer: Sobol global sampling + multi-start Nelder-Mead local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Two-stage global-to-local search: Sobol sampling + multi-start Nelder-Mead."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_used = 0

        def check_time():
            return time.time() > deadline or calls_used >= budget

        def sobol_sample(n, dim, seed=42):
            """Generate Sobol quasi-random points in [-5, 5]^dim."""
            # Use Halton sequence as a simple quasi-random alternative (no scipy dependency issues)
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
            points = sampler.random(n)  # in [0, 1)^dim
            return points * 10.0 - 5.0  # map to [-5, 5]^dim

        # Phase 1: Global sampling with Sobol sequence
        n_global = 400
        if n_global > budget:
            n_global = budget

        sobol_points = sobol_sample(n_global, dim)
        candidates = []

        for i in range(n_global):
            if check_time():
                break
            x = sobol_points[i]
            f = func(x)
            calls_used += 1
            candidates.append((f, x.copy()))
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Sort candidates and pick top 5
        candidates.sort(key=lambda t: t[0])
        top_k = min(5, len(candidates))
        top_candidates = [c[1] for c in candidates[:top_k]]

        if check_time() or len(top_candidates) == 0:
            return best_x, best_f

        # Phase 2: Multi-start Nelder-Mead local optimization
        # Compute pairwise distances to scale initial simplex sizes
        if len(top_candidates) >= 2:
            # Use median pairwise distance as scale
            dists = []
            for i in range(len(top_candidates)):
                for j in range(i + 1, len(top_candidates)):
                    dists.append(np.linalg.norm(top_candidates[i] - top_candidates[j]))
            scale = np.median(dists) if dists else 1.0
        else:
            scale = 1.0

        # Adaptive initial simplex size: between 0.5 and 2.0
        initial_alpha = max(0.5, min(2.0, scale * 0.5))

        remaining_budget = budget - calls_used
        # Reserve some budget for potential additional restarts
        budget_per_local = max(10, remaining_budget // max(1, top_k))

        for start_idx, x0 in enumerate(top_candidates):
            if check_time():
                break

            # Create initial simplex around x0
            # Nelder-Mead requires dim+1 points
            simplex = [x0.copy()]
            for j in range(dim):
                xj = x0.copy()
                xj[j] += initial_alpha
                xj = np.clip(xj, -5, 5)
                simplex.append(xj)

            # Run Nelder-Mead with evaluation budget
            local_best_f = float('inf')
            local_best_x = x0.copy()
            evals_in_local = 0

            def local_func(x):
                nonlocal calls_used, best_f, best_x, local_best_f, local_best_x, evals_in_local
                if time.time() > deadline or calls_used >= budget:
                    return 1e10  # signal to stop
                x = np.clip(x, -5, 5)
                f = func(x)
                calls_used += 1
                evals_in_local += 1
                if f < local_best_f:
                    local_best_f = f
                    local_best_x = x.copy()
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                return f

            try:
                result = minimize(
                    local_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

            # If local search found something better, update
            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()

            # Check if we should do another restart from a slightly perturbed best point
            if not check_time() and evals_in_local < budget_per_local:
                # Small perturbation restart
                rng = np.random.default_rng(42 + start_idx)
                x_perturb = best_x + rng.normal(0, 0.1, dim)
                x_perturb = np.clip(x_perturb, -5, 5)
                f_perturb = func(x_perturb)
                calls_used += 1
                if f_perturb < best_f:
                    best_f = f_perturb
                    best_x = x_perturb.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
