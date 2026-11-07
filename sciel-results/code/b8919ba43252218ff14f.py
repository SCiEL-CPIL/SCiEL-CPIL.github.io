"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex with multi-start and adaptive restart for BBOB f24."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill

        rng = np.random.default_rng(123)

        best_x = None
        best_f = np.inf

        # Multi-start strategy: use several restarts
        # Budget allocation: give more to later starts if early ones stall
        n_starts = 5
        # Allocate budget roughly equally with slight bias to first start
        budgets_per_start = [int(budget * 0.35)] + [int(budget * 0.15)] * 4
        # Adjust to sum to budget
        while sum(budgets_per_start) > budget:
            idx = np.argmax(budgets_per_start)
            budgets_per_start[idx] -= 1
        while sum(budgets_per_start) < budget:
            idx = np.argmin(budgets_per_start)
            budgets_per_start[idx] += 1

        for start_i in range(n_starts):
            if time.time() > deadline:
                break

            alloc = budgets_per_start[start_i]
            if alloc < 10:
                continue

            if start_i == 0:
                # First start: center
                x = np.zeros(dim)
            else:
                # Subsequent starts: random points, biased toward center
                x = rng.uniform(-3, 3, dim)

            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Nelder-Mead simplex
            # Initialize simplex
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x
            fvals = np.zeros(dim + 1)
            fvals[0] = fx
            calls_used = 1

            # Initialize other vertices
            for i in range(1, dim + 1):
                if time.time() > deadline:
                    break
                if calls_used >= alloc:
                    break
                # Small perturbation in one direction
                step = 0.5 * (1.0 if start_i == 0 else 0.5)
                direction = np.zeros(dim)
                direction[i - 1] = step
                simplex[i] = np.clip(x + direction, -5, 5)
                fvals[i] = func(simplex[i])
                calls_used += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = simplex[i].copy()

            # Nelder-Mead parameters
            alpha = 1.0    # reflection
            gamma = 2.0    # expansion
            rho = 0.5      # contraction
            sigma = 0.5    # shrink

            max_iter = alloc - calls_used

            for it in range(max_iter):
                if time.time() > deadline:
                    break

                # Sort by function value
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                # Check convergence
                if np.std(fvals) < 1e-10:
                    break

                x_avg = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = x_avg + alpha * (x_avg - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls_used += 1
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()

                if f_r < fvals[0]:
                    # Expansion
                    x_e = x_avg + gamma * (x_r - x_avg)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls_used += 1
                    if f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        # Outside contraction
                        x_c = x_avg + rho * (x_r - x_avg)
                    else:
                        # Inside contraction
                        x_c = x_avg - rho * (simplex[-1] - x_avg)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls_used += 1
                    if f_c < best_f:
                        best_f = f_c
                        best_x = x_c.copy()
                    if f_c < f_r:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fvals[i] = func(simplex[i])
                            calls_used += 1
                            if fvals[i] < best_f:
                                best_f = fvals[i]
                                best_x = simplex[i].copy()
                        if time.time() > deadline:
                            break

        return best_x, best_f
# EVOLVE-BLOCK-END
