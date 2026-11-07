"""Optimizer using aggressive Nelder-Mead with adaptive convergence and time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Aggressive Nelder-Mead with adaptive convergence detection."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x_start, f_start, calls, deadline, rng):
        """Run Nelder-Mead from x_start, return (x_best, f_best, calls)."""
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        x_best = x_start.copy()
        f_best = f_start
        calls = calls

        # Initialize simplex with adaptive step size
        step = 0.5
        simplex = np.zeros((self.dim + 1, self.dim))
        simplex[0] = x_start
        f_values = np.zeros(self.dim + 1)
        f_values[0] = f_start

        for i in range(self.dim):
            x_i = x_start.copy()
            x_i[i] += step
            x_i = np.clip(x_i, -5, 5)
            simplex[i + 1] = x_i
            f_values[i + 1] = func(x_i)
            calls += 1
            if calls >= self.budget or time.time() > deadline:
                break

        if calls >= self.budget or time.time() > deadline:
            idx_min = np.argmin(f_values)
            if f_values[idx_min] < f_best:
                f_best = f_values[idx_min]
                x_best = simplex[idx_min].copy()
            return x_best, f_best, calls

        # Nelder-Mead iterations with early convergence
        max_iters = 100
        stagnant = 0
        last_best = f_best
        min_improve = 1e-10

        for _ in range(max_iters):
            if calls >= self.budget or time.time() > deadline:
                break

            order = np.argsort(f_values)
            simplex = simplex[order]
            f_values = f_values[order]

            # Convergence check
            if f_values[0] < last_best - min_improve:
                last_best = f_values[0]
                stagnant = 0
            else:
                stagnant += 1
                if stagnant > 15:
                    break

            # Check simplex size convergence
            simplex_size = np.max(np.abs(simplex - simplex[0]), axis=1)
            if np.mean(simplex_size) < 1e-6:
                break

            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_ref = centroid + alpha * (centroid - simplex[-1])
            x_ref = np.clip(x_ref, -5, 5)
            f_ref = func(x_ref)
            calls += 1
            if calls >= self.budget or time.time() > deadline:
                if f_ref < f_best:
                    f_best = f_ref
                    x_best = x_ref.copy()
                break

            if f_ref < f_values[0]:
                # Expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp = np.clip(x_exp, -5, 5)
                f_exp = func(x_exp)
                calls += 1
                if calls >= self.bbudget or time.time() > deadline:
                    if f_exp < f_best:
                        f_best = f_exp
                        x_best = x_exp.copy()
                    break

                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    f_values[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    f_values[-1] = f_ref
            elif f_ref < f_values[-2]:
                simplex[-1] = x_ref
                f_values[-1] = f_ref
            else:
                # Contraction
                if f_ref < f_values[-1]:
                    x_con = centroid + rho * (x_ref - centroid)
                else:
                    x_con = centroid - rho * (x_ref - centroid)

                x_con = np.clip(x_con, -5, 5)
                f_con = func(x_con)
                calls += 1
                if calls >= self.budget or time.time() > deadline:
                    if f_con < f_best:
                        f_best = f_con
                        x_best = x_con.copy()
                    break

                if f_con < min(f_ref, f_values[-1]):
                    simplex[-1] = x_con
                    f_values[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, self.dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_values[i] = func(simplex[i])
                        calls += 1
                        if calls >= self.budget or time.time() > deadline:
                            break
                    if calls >= self.budget or time.time() > deadline:
                        break

            # Update best
            idx_min = np.argmin(f_values)
            if f_values[idx_min] < f_best:
                f_best = f_values[idx_min]
                x_best = simplex[idx_min].copy()

        return x_best, f_best, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin (300s limit, 15s buffer)

        rng = np.random.default_rng(42)
        calls = 0

        # Phase 1: Random sampling to find good starting points
        n_samples = 20
        x_best = None
        f_best = np.inf

        for i in range(n_samples):
            if calls >= self.budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            f = func(x)
            calls += 1
            if f < f_best:
                f_best = f
                x_best = x.copy()

        if x_best is None:
            x_best = np.zeros(self.dim)
            f_best = func(x_best)
            calls += 1

        # Phase 2: Aggressive local refinement with Nelder-Mead
        # Run multiple NM instances from different starting points
        n_nm_runs = 15
        for i in range(n_nm_runs):
            if calls >= self.budget or time.time() > deadline:
                break

            # Choose starting point: 70% near best, 30% random
            if i == 0 or rng.random() < 0.7:
                noise_scale = 0.5 * (1 - i / n_nm_runs)  # Decreasing noise
                x_start = np.clip(x_best + rng.standard_normal(self.dim) * noise_scale, -5, 5)
            else:
                x_start = rng.uniform(-5, 5, self.dim)

            f_start = func(x_start)
            calls += 1
            if f_start < f_best:
                f_best = f_start
                x_best = x_start.copy()

            # Run Nelder-Mead
            x_best, f_best, calls = self._nelder_mead(
                func, x_best, f_best, calls, deadline, rng
            )

        return x_best, f_best
# EVOLVE-BLOCK-END
