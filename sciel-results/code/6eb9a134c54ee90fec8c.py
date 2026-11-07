"""Optimizer for BBOB f9 (Rosenbrock rotated) using gradient-based local search."""
import numpy as np
import time


class Optimizer:
    """Gradient-based optimizer with central-difference gradient estimation."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track best solution
        best_x = None
        best_f = np.inf

        # Multi-start with gradient descent
        # We'll use a few starting points and do gradient descent from each
        # Central finite differences for gradient

        def estimate_gradient(x, eps=1e-5):
            """Central finite difference gradient."""
            grad = np.zeros(dim)
            for i in range(dim):
                xp = x.copy()
                xm = x.copy()
                xp[i] += eps
                xm[i] -= eps
                # Clip to box
                xp = np.clip(xp, -5, 5)
                xm = np.clip(xm, -5, 5)
                fp = func(xp)
                fm = func(xm)
                grad[i] = (fp - fm) / (2 * eps)
            return grad

        def gradient_descent(x0, fx0, max_iters, eps=1e-5, lr_init=0.05):
            """Gradient descent with backtracking line search."""
            nonlocal best_x, best_f, budget
            x = x0.copy()
            fx = fx0
            lr = lr_init
            iters = 0

            for _ in range(max_iters):
                if time.time() > deadline - 5:
                    break
                if budget <= 0:
                    break

                # Estimate gradient (2*dim function calls)
                if budget < 2 * dim + 1:
                    break

                grad = estimate_gradient(x, eps)
                budget -= 2 * dim

                # Check if we're at a minimum
                gnorm = np.linalg.norm(grad)
                if gnorm < 1e-8:
                    break

                # Backtracking line search
                step_dir = -grad / gnorm  # normalize direction
                alpha = lr_init
                improved = False

                for _ in range(30):
                    if budget < 1:
                        break
                    x_new = np.clip(x + alpha * step_dir, -5, 5)
                    fx_new = func(x_new)
                    budget -= 1

                    if fx_new < fx:
                        x, fx = x_new, fx_new
                        improved = True
                        # Increase learning rate slightly if we made good progress
                        alpha *= 1.2
                        break
                    else:
                        alpha *= 0.5

                if not improved:
                    # Try smaller step or break
                    if alpha < 1e-10:
                        break

                # Update best
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

                iters += 1

            return x, fx

        # Multi-start strategy: try several starting points
        # First, do a coarse random search to find a good starting region
        n_starts = 5
        calls_per_start = (budget - n_starts) // n_starts

        for s in range(n_starts):
            if time.time() > deadline - 10:
                break
            if budget < 1:
                break

            # Generate starting point
            if s == 0:
                x0 = np.zeros(dim)
            else:
                x0 = rng.uniform(-4, 4, dim)

            fx0 = func(x0)
            budget -= 1

            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()

            # Run gradient descent from this point
            if budget > 2 * dim:
                x, fx = gradient_descent(x0, fx0, max_iters=200, eps=1e-5, lr_init=0.05)

                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

            # If we have budget left, do a local refinement with smaller eps
            if budget > 2 * dim + 10 and best_x is not None:
                if time.time() > deadline - 10:
                    break
                x_ref, fx_ref = gradient_descent(best_x, best_f, max_iters=100, eps=1e-6, lr_init=0.01)
                if fx_ref < best_f:
                    best_f = fx_ref
                    best_x = x_ref.copy()

        # Final local refinement if budget remains
        if budget > 2 * dim + 5 and best_x is not None:
            if time.time() > deadline - 5:
                pass
            else:
                x_final, fx_final = gradient_descent(best_x, best_f, max_iters=50, eps=1e-7, lr_init=0.005)
                if fx_final < best_f:
                    best_f = fx_final
                    best_x = x_final.copy()

        if best_x is None:
            best_x = np.zeros(dim)

        return best_x, best_f
