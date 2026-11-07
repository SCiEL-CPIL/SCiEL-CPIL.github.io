"""Simulated Annealing optimizer for BBOB-style f20 (Schwefel)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Simulated Annealing with geometric cooling and adaptive step size."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initialize
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # SA parameters
        t0 = 10.0  # Initial temperature
        t_end = 1e-4
        # Geometric cooling: T_k = T0 * (T_end/T0)^(k/(N-1))
        # We'll use a per-iteration cooling factor
        if budget > 1:
            alpha = (t_end / t0) ** (1.0 / (budget - 1))
        else:
            alpha = t_end / t0

        temperature = t0
        step_size = 2.0  # Initial step size

        calls_made = 1

        for i in range(budget - 1):
            # Check deadline
            if time.time() > deadline:
                break

            # Generate candidate
            direction = rng.standard_normal(dim)
            y = x + step_size * direction
            y = np.clip(y, -5, 5)

            fy = func(y)
            calls_made += 1

            # Accept or reject
            if fy < fx:
                # Always accept improvement
                x = y
                fx = fy
                # Adapt step size: increase slightly on improvement
                step_size = min(step_size * 1.1, 5.0)
            else:
                # Metropolis criterion
                delta = fy - fx
                if delta < 0:
                    accept = True
                else:
                    accept = np.log(rng.uniform(1e-10, 1.0)) > -delta / temperature

                if accept:
                    x = y
                    fx = fy
                    # Keep step size on accepted worse move
                else:
                    # Reject: decrease step size
                    step_size = max(step_size * 0.95, 0.01)

            # Update best
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

            # Cool down
            temperature *= alpha

        return best_x, best_fx
# EVOLVE-BLOCK-END
