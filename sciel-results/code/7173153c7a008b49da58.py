"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer: brief Sobol global scan followed by (1+1)-ES with
    1/5 success rule and a single adaptive restart if budget remains.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # 5s safety margin before 300s limit
        budget = self.budget
        dim = self.dim

        calls_used = 0

        def f(x):
            nonlocal calls_used
            calls_used += 1
            return func(x)

        # --- Phase 1: Sobol quasi-random scan for global basin identification ---
        # Use a small number of Sobol points to find the best region quickly.
        # This is a structural change from pure random start: systematic coverage
        # of the domain before committing to local search.
        
        def sobol_sequence(n, dim, seed=42):
            """Generate n Sobol quasi-random points in [0,1]^dim."""
            try:
                from scipy.stats import qmc
                sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
                points = sampler.random(n)
                return points
            except ImportError:
                # Fallback: Latin hypercube sampling
                rng = np.random.default_rng(seed)
                points = np.zeros((n, dim))
                for d in range(dim):
                    perm = rng.permutation(n)
                    points[:, d] = (perm + 0.5) / n
                return points

        # Use a small fraction of budget for global scan (max 50 points)
        n_sobol = min(50, budget // 10)
        if n_sobol < 5:
            n_sobol = 5

        sobol_points = sobol_sequence(n_sobol, dim)
        sobol_points = sobol_points * 10.0 - 5.0  # Map [0,1] to [-5,5]

        best_x = None
        best_f = np.inf
        second_best_x = None
        second_best_f = np.inf

        for pt in sobol_points:
            if calls_used >= budget or time.time() > deadline:
                break
            pt_clipped = np.clip(pt, -5, 5)
            fp = f(pt_clipped)
            if fp < best_f:
                second_best_x, second_best_f = best_x, best_f
                best_x, best_f = pt_clipped.copy(), fp
            elif fp < second_best_f:
                second_best_x, second_best_f = pt_clipped.copy(), fp

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = f(best_x)

        # --- Phase 2: (1+1)-ES with 1/5 success rule (main refinement) ---
        # This is the core refinement machinery from the highest-scoring inspirations.
        rng = np.random.default_rng(0)
        x = best_x.copy()
        fx = best_f
        sigma = 2.0

        # Allocate remaining budget to ES
        remaining = budget - calls_used
        # Use 85% of remaining for main ES, 15% for potential restart
        main_budget = int(remaining * 0.85)
        restart_budget = remaining - main_budget

        for _ in range(main_budget):
            if calls_used >= budget or time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = f(y)
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
            else:
                sigma *= 0.87

        # --- Phase 3: Optional restart from second-best Sobol point ---
        # If we have budget left and the second-best point is promising,
        # try a short ES from there.
        if (restart_budget > 20 and second_best_x is not None 
                and time.time() < deadline and calls_used < budget):
            # Only restart if the second best is within a reasonable range
            # of the current best (to avoid wasting budget on poor regions)
            if second_best_f < fx + 50.0:  # threshold for promising region
                rng2 = np.random.default_rng(123)
                x2 = second_best_x.copy()
                fx2 = second_best_f
                sigma2 = 2.0

                for _ in range(restart_budget):
                    if calls_used >= budget or time.time() > deadline:
                        break
                    y2 = np.clip(x2 + sigma2 * rng2.standard_normal(dim), -5, 5)
                    fy2 = f(y2)
                    if fy2 < fx2:
                        x2, fx2 = y2, fy2
                        sigma2 *= 1.5
                    else:
                        sigma2 *= 0.87

                if fx2 < fx:
                    x, fx = x2, fx2

        return x, fx
# EVOLVE-BLOCK-END
