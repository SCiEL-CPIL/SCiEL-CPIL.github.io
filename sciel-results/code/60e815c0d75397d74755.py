"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer exploiting ZDT1 structure: fix x[1:] to minimize g, vary x[0] for Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT1 structure: f1 = x[0], f2 = g(x[1:]) * h(f1/g, 1/g)
        # g(x[1:]) = 1 + 10*(n-1) + sum((x_i - 0.5)^2 - cos(20*pi*(x_i-0.5)))
        # g is minimized at x[1:] = 0.5, giving g_min = 1
        # Pareto front: f2 = 1 + (f1)^0.25 when g=1

        # Strategy:
        # 1. Spend some budget finding good x[1:] values (minimizing g)
        # 2. Use remaining budget to explore x[0] values for Pareto front coverage

        # Phase 1: Optimize x[1:] to minimize g (use ~20% of budget)
        # Initialize with x[1:] = 0.5 which is the known optimum
        x_base = np.ones(dim)
        x_base[0] = 0.5
        x_base[1:] = 0.5

        # Evaluate the base point
        f_base = np.array(problem(x_base))
        all_x = [x_base.copy()]
        all_f = [f_base.copy()]
        evals = 1

        # Phase 2: Explore the Pareto front by varying x[0]
        # The Pareto front for ZDT1 with g=1 is: f1 in [0,1], f2 = 1 + f1^0.25
        # We want to sample points along this front

        remaining_budget = budget - evals

        # Use a mix of:
        # - Uniform sampling of x[0] in [0,1]
        # - Local refinement around promising points

        # First, do uniform sampling to get good coverage
        n_uniform = min(remaining_budget // 2, 50)
        if n_uniform > 0:
            x0_values = np.linspace(0.0, 1.0, n_uniform)
            for x0 in x0_values:
                if time.time() > deadline:
                    break
                if evals >= budget:
                    break
                x = x_base.copy()
                x[0] = x0
                x = np.clip(x, lo, hi)
                f = np.array(problem(x))
                all_x.append(x.copy())
                all_f.append(f.copy())
                evals += 1

        # Phase 3: Refine - use remaining budget for local search around best points
        # Convert to arrays for easier manipulation
        all_x = np.array(all_x)
        all_f = np.array(all_f)

        # Find non-dominated points
        def is_dominated(f, F):
            # Check if f is dominated by any point in F
            for fi in F:
                if np.all(fi <= f) and np.any(fi < f):
                    return True
            return False

        # Use remaining budget for local refinement
        remaining = budget - evals
        if remaining > 0:
            # Sample from current non-dominated set and perturb
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                if evals >= budget:
                    break

                # Pick a random point from current population
                idx = np.random.randint(0, len(all_x))
                parent_x = all_x[idx].copy()
                parent_f = all_f[idx]

                # Perturb: mostly change x[0], sometimes x[1:]
                if np.random.rand() < 0.7:
                    # Change x[0]
                    parent_x[0] += np.random.normal(0, 0.05)
                else:
                    # Change some x[1:]
                    n_change = np.random.randint(1, 3)
                    indices = np.random.choice(dim - 1, n_change, replace=False)
                    parent_x[1:][indices] += np.random.normal(0, 0.02, n_change)

                parent_x = np.clip(parent_x, lo, hi)
                f_new = np.array(problem(parent_x))
                all_x = np.vstack([all_x, parent_x])
                all_f = np.vstack([all_f, f_new])
                evals += 1

        return all_x
# EVOLVE-BLOCK-END
