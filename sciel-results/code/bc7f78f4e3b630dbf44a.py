"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step-size and local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        # Initial point
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Phase 1: (1+1)-ES with improved sigma adaptation
        sigma = 2.0
        calls_used = 1
        success = 0
        # Use a success-count based adaptation for more stable sigma
        for _ in range(self.budget - 50):
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_used += 1

            if fy < fx:
                step_norm = np.linalg.norm(y - x)
                if step_norm > 1e-12:
                    # Adapt sigma: target step size ~ 0.5*sqrt(dim)
                    target_step = 0.5 * np.sqrt(self.dim)
                    ratio = target_step / max(step_norm, 1e-12)
                    # Logarithmic adjustment for stability
                    sigma *= 1.0 + 0.3 * np.log(max(ratio, 1e-12))
                    sigma = np.clip(sigma, 1e-6, 10.0)
                x, fx = y, fy
                success += 1
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            else:
                sigma *= 0.7
                sigma = max(sigma, 1e-8)

        # Phase 2: Local refinement using coordinate descent or small perturbations
        # Use remaining budget for fine-tuning
        remaining_calls = self.budget - calls_used
        if remaining_calls > 20 and time.time() < deadline - 5.0:
            # Try small perturbations in each coordinate
            current_x = best_x.copy()
            current_fx = best_fx
            local_sigma = 0.1
            for _ in range(min(remaining_calls - 5, 100)):
                if time.time() > deadline:
                    break
                # Random coordinate perturbation
                d = rng.integers(self.dim)
                direction = rng.choice([-1, 1])
                step = local_sigma * direction
                y = current_x.copy()
                y[d] += step
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_used += 1

                if fy < current_fx:
                    current_x = y
                    current_fx = fy
                    if current_fx < best_fx:
                        best_fx = current_fx
                        best_x = current_x.copy()
                    local_sigma *= 1.2
                else:
                    local_sigma *= 0.8
                    local_sigma = max(local_sigma, 1e-6)

        return best_x, best_fx
# EVOLVE-BLOCK-END
