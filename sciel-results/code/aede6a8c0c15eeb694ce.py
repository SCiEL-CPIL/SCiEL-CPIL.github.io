"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Gradient-aware optimizer for linear slope functions."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Estimate gradient direction using a small set of points
        # Start at origin (center of box) for symmetric sampling
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls = 1

        # Sample along each axis to estimate gradient
        eps = 1.0
        grad_est = np.zeros(dim)
        for i in range(dim):
            xi = x0.copy()
            xi[i] += eps
            fi = func(xi)
            calls += 1
            grad_est[i] = (fi - f0) / eps

        # Normalize gradient
        grad_norm = np.linalg.norm(grad_est)
        if grad_norm > 1e-10:
            direction = -grad_est / grad_norm  # negative gradient = descent
        else:
            direction = np.zeros(dim)

        # Phase 2: Walk along the gradient direction to find the minimum
        # The minimum of a linear function on a box is at a vertex
        # We walk from x0 in the descent direction until we hit a boundary
        x = x0.copy()
        fx = f0

        # Try large steps first to find the boundary quickly
        step = 5.0  # try to jump to boundary
        for _ in range(50):  # limited exploration steps
            if time.time() > deadline:
                break
            y = x + step * direction
            # Clip to box
            y = np.clip(y, -5, 5)
            if np.allclose(y, x):
                # At boundary, try to find which dimension is constrained
                break
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                step *= 0.5  # reduce step when making progress
            else:
                step *= 0.5
                if step < 0.01:
                    break

        # Phase 3: Fine-tune around the current best point
        # Use a local search with adaptive step size
        sigma = 0.5
        best_x = x.copy()
        best_f = fx

        remaining = budget - calls
        for i in range(remaining):
            if time.time() > deadline:
                break
            # Generate perturbation
            noise = rng.standard_normal(dim)
            y = best_x + sigma * noise
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls += 1
            if fy < best_f:
                best_x, best_f = y, fy
                sigma *= 1.2  # expand step on success
            else:
                sigma *= 0.8  # shrink step on failure

        return best_x, best_f
# EVOLVE-BLOCK-END
