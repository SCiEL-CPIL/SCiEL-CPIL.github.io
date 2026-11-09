"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Iterative multi-scale coordinate descent with adaptive convergence for Büche-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s limit

        best_x = None
        best_fx = float('inf')
        calls_made = 0

        def update_best(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        def time_exceeded():
            return time.time() > deadline

        # Phase 1: Smart initialization using knowledge of Büche-Rastrigin structure
        # The unshifted minimum is at (2,2,2,2,2). After shift+rotation, it's in [-4,4]^5.
        # Try a few candidate starting points near the expected region.
        
        # Start at origin (safe, central)
        x = np.zeros(self.dim)
        fx = func(x)
        calls_made += 1
        update_best(x, fx)

        # Also probe near x=2 (the known unshifted minimum)
        x_probe = np.full(self.dim, 2.0)
        x_probe = np.clip(x_probe, -5, 5)
        if not np.array_equal(x, x_probe):
            fx_probe = func(x_probe)
            calls_made += 1
            if fx_probe < fx:
                x = x_probe.copy()
                fx = fx_probe
                update_best(x, fx)

        # Phase 2: Iterative multi-scale coordinate descent
        # At each step size, repeatedly sweep all dimensions until no improvement
        # This allows navigating rotated landscapes by making multi-dimensional progress
        
        step_sizes = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 
                      0.015625, 0.0078125, 0.00390625, 0.001953125, 0.0009765625]

        for step in step_sizes:
            if calls_made >= self.budget or time_exceeded():
                break

            # Iterate until convergence or budget/time limit at this step size
            max_iterations = 50  # Cap iterations per step size to avoid infinite loops
            for iteration in range(max_iterations):
                if calls_made >= self.budget or time_exceeded():
                    break

                improved = False
                for d in range(self.dim):
                    if calls_made >= self.budget or time_exceeded():
                        break

                    # Try positive step
                    y = x.copy()
                    y[d] = np.clip(y[d] + step, -5, 5)
                    if not np.allclose(y, x):  # Skip if no change due to clipping
                        fy = func(y)
                        calls_made += 1
                        if fy < fx:
                            x = y
                            fx = fy
                            update_best(x, fx)
                            improved = True

                    # Try negative step
                    if calls_made >= self.budget or time_exceeded():
                        break
                    y = x.copy()
                    y[d] = np.clip(y[d] - step, -5, 5)
                    if not np.allclose(y, x):
                        fy = func(y)
                        calls_made += 1
                        if fy < fx:
                            x = y
                            fx = fy
                            update_best(x, fx)
                            improved = True

                if not improved:
                    break  # Converged at this step size

        # Phase 3: Fine-tuning with random perturbations if budget remains
        if calls_made < self.budget and not time_exceeded():
            rng = np.random.default_rng(42)
            sigma = 0.01
            remaining = self.budget - calls_made
            for _ in range(remaining):
                if time_exceeded():
                    break
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x = y
                    fx = fy
                    update_best(x, fx)
                    sigma = min(sigma * 1.3, 0.1)
                else:
                    sigma *= 0.92
                sigma = max(sigma, 0.001)

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_fx = func(best_x)
            calls_made += 1

        return best_x, best_fx
# EVOLVE-BLOCK-END
