"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start local search with Nelder-Mead for multi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # We'll track all evaluated points and their objective values
        all_x = []
        all_f = []
        budget_used = 0

        def eval_point(x):
            nonlocal budget_used
            if budget_used >= self.budget:
                return None
            x = np.clip(x, lo, hi)
            f = problem(x)
            budget_used += 1
            all_x.append(x.copy())
            all_f.append(f.copy())
            return f

        # Phase 1: Random initialization - spread out in the space
        n_init = min(30, self.budget // 3)
        rng = np.random.default_rng(42)

        for _ in range(n_init):
            x = lo + span * rng.random(self.dim)
            eval_point(x)

        # Phase 2: Multi-start local search with Nelder-Mead
        # Use different scalarization weights to explore different parts of Pareto front
        n_local = min(25, self.budget - budget_used)

        # Generate diverse weight vectors
        weights_list = []
        # Uniform weights
        for _ in range(n_local // 2):
            w = rng.dirichlet(np.ones(self.n_obj))
            weights_list.append(w)
        # Add some extreme weights to find knee points
        for _ in range(n_local - len(weights_list)):
            w = rng.dirichlet(np.ones(self.n_obj) * 0.5)
            weights_list.append(w)

        # Select starting points for local search from initial population
        # Prefer points that are non-dominated
        if len(all_f) > 0:
            F_init = np.array(all_f)
            # Find non-dominated points
            n_dom = len(F_init)
            is_dominated = np.zeros(n_dom, dtype=bool)
            for i in range(n_dom):
                for j in range(n_dom):
                    if i == j:
                        continue
                    if np.all(F_init[j] <= F_init[i]) and np.any(F_init[j] < F_init[i]):
                        is_dominated[i] = True
                        break

            non_dom_idx = np.where(~is_dominated)[0]
            if len(non_dom_idx) == 0:
                non_dom_idx = np.arange(n_dom)

            # Select starting points from non-dominated set
            start_points = []
            for i in non_dom_idx:
                start_points.append(all_x[i])

            # If we have more local searches than start points, add some random ones
            while len(start_points) < n_local:
                x = lo + span * rng.random(self.dim)
                start_points.append(x)

            # Run Nelder-Mead local search
            for k in range(min(n_local, len(start_points))):
                if time.time() > deadline - 10:
                    break
                if budget_used >= self.budget:
                    break

                w = weights_list[k % len(weights_list)]
                x0 = start_points[k].copy()

                def objective(x):
                    if budget_used >= self.budget:
                        return np.inf
                    x_clipped = np.clip(x, lo, hi)
                    f = problem(x_clipped)
                    budget_used += 1
                    all_x.append(x_clipped.copy())
                    all_f.append(f.copy())
                    return np.dot(w, f)

                try:
                    result = minimize(
                        objective,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4}
                    )
                except Exception:
                    pass

        # Phase 3: If budget remains, do more random sampling in promising regions
        if budget_used < self.budget and time.time() < deadline - 5:
            # Find the best region based on hypervolume contribution
            if len(all_f) > 0:
                F = np.array(all_f)
                # Find points that are close to the Pareto front
                # Sample additional points near existing good solutions
                remaining = self.budget - budget_used
                n_extra = min(remaining, 20)

                for _ in range(n_extra):
                    if time.time() > deadline - 5:
                        break
                    # Pick a random existing point and perturb it
                    idx = rng.integers(len(all_x))
                    x = all_x[idx] + 0.05 * span * rng.standard_normal(self.dim)
                    eval_point(x)

        return all_x
# EVOLVE-BLOCK-END
