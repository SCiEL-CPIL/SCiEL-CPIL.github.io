"""Optimizer for BBOB-style Schwefel f20 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize


class Optimizer:
    """Differential evolution with periodic-aware initialization + L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        call_count = 0

        def eval_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= self.budget:
                return best_f
            x = np.asarray(x, dtype=np.float64)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = np.array(x, copy=True)
            return fx

        bounds = [(-5.0, 5.0)] * self.dim

        # Phase 1: Coarse grid to estimate landscape structure
        # Use a small grid to find promising regions
        grid_points = []
        # Sample at key locations: center, axis points, and random
        grid_points.append(np.zeros(self.dim))
        for i in range(self.dim):
            for val in [-3.0, 0.0, 3.0]:
                p = np.zeros(self.dim)
                p[i] = val
                grid_points.append(p)
        # Add random points
        for _ in range(20):
            grid_points.append(rng.uniform(-4.5, 4.5, self.dim))

        # Evaluate grid points (limited budget)
        grid_budget = min(50, self.budget // 5)
        grid_results = []
        for p in grid_points:
            if call_count >= grid_budget:
                break
            if time.time() > deadline:
                break
            p = np.clip(p, -5, 5)
            f = eval_func(p)
            grid_results.append((f, p))

        # Sort by fitness
        grid_results.sort(key=lambda x: x[0])

        # Phase 2: Differential Evolution with periodic-aware seeding
        # Schwefel's function has period 4π ≈ 12.566 in each dimension
        # The global minimum is at 4.209687... in each dimension
        # With hidden shift, we use DE to find the shifted minimum

        de_budget = int(self.budget * 0.55)
        remaining_after_de = self.budget - de_budget

        # Create initial population from best grid points + periodic shifts
        pop_size = 30
        pop = []

        # Seed with best grid points
        for f, p in grid_results[:min(10, len(grid_results))]:
            pop.append(p.copy())

        # Add periodic-shifted versions of best points
        # Schwefel period is 4π ≈ 12.566, but domain is [-5,5], so shifts are limited
        # Instead, use the known structure: minima cluster around specific values
        # Add points near the known Schwefel minimum (4.2097) with random shifts
        for _ in range(pop_size - len(pop)):
            if time.time() > deadline:
                break
            # Mix of random and structured points
            if rng.random() < 0.3:
                # Near known Schwefel minimum region
                p = rng.uniform(2.0, 5.0, self.dim)
            elif rng.random() < 0.5:
                # Near negative region
                p = rng.uniform(-5.0, -2.0, self.dim)
            else:
                p = rng.uniform(-4.5, 4.5, self.dim)
            pop.append(np.clip(p, -5, 5))

        # Ensure population size
        while len(pop) < pop_size:
            pop.append(rng.uniform(-4.5, 4.5, self.dim))

        # Run differential evolution manually with budget tracking
        def de_obj(x):
            if call_count >= de_budget + grid_budget:
                return best_f
            return eval_func(np.clip(x, -5, 5))

        # Manual DE with (15/100/0.9) strategy
        F = 0.8  # mutation factor
        CR = 0.9  # crossover rate
        max_iter = max(1, (de_budget - call_count) // pop_size)

        pop = np.array(pop)
        pop_f = np.array([de_obj(p) for p in pop])

        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            if call_count >= self.budget:
                break

            # Sort population
            order = np.argsort(pop_f)
            pop = pop[order]
            pop_f = pop_f[order]

            for i in range(pop_size):
                if time.time() > deadline:
                    break
                if call_count >= self.budget:
                    break

                # Select three distinct random indices different from i
                r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)

                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                v = np.clip(v, -5, 5)

                # Crossover
                j_rand = rng.integers(self.dim)
                trial = np.where(rng.random(self.dim) <= CR, v, pop[i])
                trial[j_rand] = v[j_rand]
                trial = np.clip(trial, -5, 5)

                # Evaluate trial
                f_trial = de_obj(trial)

                # Selection
                if f_trial <= pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = f_trial

        # Phase 3: L-BFGS-B polish from best points
        if best_x is not None and call_count < self.budget:
            polish_budget = self.budget - call_count
            n_polish = min(6, max(1, polish_budget // 20))

            # Get top points from DE population
            order = np.argsort(pop_f)
            top_points = [pop[order[i]].copy() for i in range(min(n_polish, pop_size))]

            # Also add best_x
            if best_x not in top_points:
                top_points.append(best_x.copy())

            per_polish = max(5, polish_budget // len(top_points))
            bounds_list = [(-5.0, 5.0)] * self.dim

            for sp in top_points:
                if call_count >= self.budget:
                    break
                if time.time() > deadline:
                    break

                calls_this = [0]

                def lbfgs_obj(x):
                    nonlocal calls_this
                    if calls_this[0] >= per_polish:
                        return best_f
                    calls_this[0] += 1
                    return eval_func(np.clip(x, -5, 5))

                try:
                    res = minimize(
                        lbfgs_obj,
                        sp,
                        method='L-BFGS-B',
                        bounds=bounds_list,
                        options={
                            'maxiter': per_polish,
                            'ftol': 1e-10,
                            'gtol': 1e-8,
                        }
                    )
                except Exception:
                    pass

        # Phase 4: Final fine-tuning with small perturbations
        if best_x is not None and call_count < self.budget:
            remaining = self.budget - call_count
            for _ in range(min(remaining, 40)):
                if time.time() > deadline:
                    break
                if call_count >= self.budget:
                    break
                y = np.clip(best_x + 0.02 * rng.standard_normal(self.dim), -5, 5)
                fy = eval_func(y)

        return best_x, best_f
# EVOLVE-BLOCK-END
