"""Optimizer for BBOB f4 (Büche-Rastrigin) with multi-restart (1+1)-ES."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Track best overall
        best_x = None
        best_f = np.inf

        # Use Sobol-like quasi-random points for initial exploration
        # We'll generate a set of diverse starting points
        rng = np.random.default_rng(42)

        # Generate initial diverse points using stratified sampling + Sobol-like
        # For 5D, we can use a simple quasi-random sequence
        n_initial = min(20, budget // 10)
        initial_points = []

        # Simple quasi-random: use golden ratio stratification per dimension
        golden = (1 + np.sqrt(5)) / 2
        inv_golden = 1.0 / golden

        for i in range(n_initial):
            point = np.zeros(dim)
            for d in range(dim):
                # Stratified sampling with offset
                frac = (i * inv_golden + d * 0.618033988749895) % 1.0
                point[d] = -5 + 10 * frac
            initial_points.append(point)

        # Evaluate initial points
        for p in initial_points:
            if time.time() > deadline:
                break
            fp = func(p)
            if fp < best_f:
                best_f = fp
                best_x = p.copy()

        calls_made = n_initial

        # Multi-restart (1+1)-ES with adaptive sigma
        # Each restart runs for a limited number of iterations
        max_iters_per_restart = 150
        n_restarts = max(1, (budget - calls_made) // max_iters_per_restart)

        # Sort initial points by fitness for restart ordering
        evals = []
        for p in initial_points:
            if time.time() > deadline:
                break
            fp = func(p)
            evals.append((fp, p.copy()))
            calls_made += 1

        evals.sort(key=lambda x: x[0])

        # Restart from best points first
        for restart_idx in range(n_restarts):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break

            # Choose restart point
            if restart_idx < len(evals):
                x = evals[restart_idx][1].copy()
                fx = evals[restart_idx][0]
            else:
                x = best_x.copy() if best_x is not None else rng.uniform(-5, 5, dim)
                fx = best_f if best_f != np.inf else func(x)
                calls_made += 1

            # Adaptive (1+1)-ES
            sigma = 1.0
            sigma_min = 0.01
            sigma_max = 5.0
            stagnation = 0
            max_stagnation = 30

            for it in range(max_iters_per_restart):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    stagnation = 0
                    sigma = min(sigma * 1.2, sigma_max)
                else:
                    stagnation += 1
                    sigma = max(sigma * 0.8, sigma_min)

                    # If stagnated too long, break out and restart
                    if stagnation >= max_stagnation:
                        break

        return best_x, best_f
