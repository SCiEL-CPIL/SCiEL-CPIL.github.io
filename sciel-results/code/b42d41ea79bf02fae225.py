"""Black-box continuous optimizer for BBOB-style f9 (Rosenbrock rotated)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start stochastic BFGS with finite-difference gradients."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 270  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f, best_x = fx, x.copy()
            return fx

        def finite_diff_gradient(x, h=1e-5):
            """Estimate gradient using central differences."""
            g = np.zeros(dim)
            for i in range(dim):
                xp = x.copy()
                xm = x.copy()
                xp[i] = min(xp[i] + h, 5.0)
                xm[i] = max(xm[i] - h, -5.0)
                fp = safe_call(xp)
                fm = safe_call(xm)
                actual_h = xp[i] - xm[i]
                if actual_h > 1e-12:
                    g[i] = (fp - fm) / actual_h
                else:
                    g[i] = 0.0
            return g

        def bfgs_local(x0, fx0, max_iter, fd_h=1e-5):
            """Run BFGS from x0 with initial f-value fx0."""
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget or time.time() > deadline - 5:
                return x0, fx0

            x = x0.copy()
            fx = fx0
            H = np.eye(dim)  # inverse Hessian approximation
            g = finite_diff_gradient(x, fd_h)

            for it in range(max_iter):
                if calls_made >= budget or time.time() > deadline - 5:
                    break

                # Check convergence
                if np.linalg.norm(g) < 1e-6:
                    break

                # Search direction
                d = -H @ g
                # Ensure descent direction
                if np.dot(d, g) > 0:
                    H = np.eye(dim)
                    d = -g

                # Line search (backtracking)
                alpha = 1.0
                c1 = 1e-4
                improved = False
                for ls in range(20):
                    x_new = np.clip(x + alpha * d, -5, 5)
                    fx_new = safe_call(x_new)
                    if fx_new < fx + c1 * alpha * np.dot(g, d):
                        improved = True
                        break
                    alpha *= 0.5

                if not improved:
                    # Try smaller step
                    x_new = np.clip(x + 0.01 * d, -5, 5)
                    fx_new = safe_call(x_new)
                    if fx_new >= fx:
                        break
                    improved = True

                if not improved:
                    break

                # Update gradient
                g_new = finite_diff_gradient(x_new, fd_h)

                # BFGS update
                s = x_new - x
                y = g_new - g
                sy = np.dot(s, y)
                if sy > 1e-12:
                    Hy = H @ y
                    rho = 1.0 / sy
                    H = H + (rho * np.outer(s, s) * (1 + rho * np.dot(y, Hy)) - rho * (np.outer(s, Hy) + np.outer(Hy, s)))
                    # Regularize to maintain positive definiteness
                    H = 0.5 * (H + H.T)
                    # Clip extreme values
                    H = np.clip(H, -100, 100)

                x = x_new
                fx = fx_new
                g = g_new

            return x, fx

        # Multi-start strategy: diverse initializations
        n_starts = 8
        calls_per_start = max(10, (budget - n_starts) // n_starts)

        # Generate diverse starting points
        starts = []
        # Center point
        starts.append(np.zeros(dim))
        # Points along axes
        for i in range(dim):
            p = np.zeros(dim)
            p[i] = rng.uniform(-3, 3)
            starts.append(p)
        # Random points
        while len(starts) < n_starts:
            starts.append(rng.uniform(-4, 4, dim))

        for si, x0 in enumerate(starts):
            if calls_made >= budget or time.time() > deadline - 10:
                break

            x0 = np.clip(x0, -5, 5)
            fx0 = safe_call(x0)

            # Adaptive max iterations based on remaining budget
            remaining = budget - calls_made
            # Each BFGS iteration costs ~2*dim+1 calls for gradient + line search
            est_calls_per_iter = 2 * dim + 5
            max_iter = min(15, max(3, remaining // est_calls_per_iter))

            x_local, fx_local = bfgs_local(x0, fx0, max_iter)

            # If BFGS didn't help, try a small local search
            if fx_local >= fx0 and calls_made < budget:
                # Simple local perturbation
                for _ in range(min(5, budget - calls_made)):
                    if time.time() > deadline - 5:
                        break
                    x_try = np.clip(x0 + 0.1 * rng.standard_normal(dim), -5, 5)
                    fx_try = safe_call(x_try)

        return best_x, best_f
# EVOLVE-BLOCK-END
