"""Derivative-free Broyden (BFGS-like) with coarse global search for BBOB f6."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase optimizer: coarse CMA-ES-like global search, then BFGS-like local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        # --- Phase 1: Coarse global search using a simple population-based method ---
        # Use a small population with restarts to find a good starting point
        rng = np.random.default_rng(42)

        # We'll spend about 60% of budget on global search
        global_budget = int(budget * 0.6)
        local_budget = budget - global_budget

        # Simple population-based search: keep best, generate mutants
        pop_size = 10
        x = np.zeros(dim)
        fx = np.inf

        # Initialize with a few random points
        candidates = []
        for i in range(pop_size):
            xi = rng.uniform(-4.5, 4.5, dim)
            fxi = func(xi)
            candidates.append((fxi, xi.copy()))
            if fxi < fx:
                fx = fxi
                x = xi.copy()

        # Simple hill-climbing with larger steps for global exploration
        sigma = 2.0
        calls_made = pop_size
        for it in range(global_budget - pop_size):
            if time.time() > deadline:
                break
            # Generate a mutant from the best
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
            else:
                sigma *= 0.9

            # Occasionally try a random restart
            if it % 50 == 49 and it > 0:
                xr = rng.uniform(-4.5, 4.5, dim)
                fxr = func(xr)
                calls_made += 1
                if fxr < fx:
                    x, fx = xr, fxr
                    sigma = 2.0

        # --- Phase 2: BFGS-like local refinement using finite differences ---
        # Build approximate inverse Hessian from finite-difference gradients
        # We'll use a simplified Broyden update with finite differences

        # Compute initial gradient using central differences
        def compute_gradient(x0, h=1e-4):
            grad = np.zeros(dim)
            for i in range(dim):
                xp = x0.copy()
                xm = x0.copy()
                xp[i] = min(5.0, xp[i] + h)
                xm[i] = max(-5.0, xm[i] - h)
                fp = func(xp)
                fm = func(xm)
                grad[i] = (fp - fm) / (xp[i] - xm[i])
            return grad

        # Check time
        if time.time() > deadline:
            return x, fx

        # BFGS-like iteration
        # H is the inverse Hessian approximation, start as identity
        H = np.eye(dim)
        grad = compute_gradient(x)
        calls_made += 2 * dim  # 2 function calls per gradient component

        # Track for Broyden update
        x_prev = x.copy()
        grad_prev = grad.copy()

        for it in range(local_budget):
            if time.time() > deadline:
                break

            # Compute search direction
            d = -H @ grad

            # Line search: try several step sizes
            best_step = 0.0
            best_fx = fx
            best_x = x.copy()
            best_grad = grad.copy()

            for alpha in [0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.001]:
                x_trial = np.clip(x + alpha * d, -5, 5)
                fx_trial = func(x_trial)
                calls_made += 1
                if fx_trial < best_fx:
                    best_fx = fx_trial
                    best_x = x_trial.copy()
                    best_step = alpha

            if best_step == 0.0:
                # No improvement, try smaller steps or break
                break

            # Compute new gradient
            grad_new = compute_gradient(best_x)
            calls_made += 2 * dim

            # Broyden update: H_new = H + (s - H*y) * s^T / (s^T * y)
            s = best_x - x
            y = grad_new - grad
            sy = s @ y
            if abs(sy) > 1e-12:
                Hy = H @ y
                H = H + (s - Hy) @ s / sy

            # Check convergence
            if np.linalg.norm(grad_new) < 1e-6:
                break
            if np.linalg.norm(s) < 1e-8:
                break

            x = best_x
            fx = best_fx
            grad = grad_new

            # Check budget
            if calls_made >= budget:
                break

        return x, fx
# EVOLVE-BLOCK-END
