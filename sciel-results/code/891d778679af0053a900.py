"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic quasi-Newton (BFGS) with finite-difference gradient estimation."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        # Initial point: center of the box
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # BFGS Hessian approximation H (starts as identity)
        H = np.eye(dim)

        # Finite-difference step size
        h = 0.01

        def compute_gradient(x):
            """Compute gradient using central finite differences."""
            nonlocal calls
            grad = np.zeros(dim)
            for i in range(dim):
                if time.time() > deadline:
                    return None
                xp = x.copy()
                xp[i] += h
                xp = np.clip(xp, -5, 5)
                calls += 1
                f_p = func(xp)
                xm = x.copy()
                xm[i] -= h
                xm = np.clip(xm, -5, 5)
                calls += 1
                f_m = func(xm)
                denom = xp[i] - xm[i]
                if denom < 1e-12:
                    denom = 2 * h
                grad[i] = (f_p - f_m) / denom
            return grad

        # Line search parameters
        c1 = 1e-4
        rho = 0.5
        max_ls = 20

        for iteration in range(budget - 1):
            if time.time() > deadline:
                break
            if calls >= budget:
                break

            # Compute gradient
            g = compute_gradient(x)
            if g is None:
                break

            # Check for convergence
            if np.linalg.norm(g) < 1e-6:
                break

            # Compute search direction: d = -H @ g
            d = -H @ g

            # Ensure descent direction
            if np.dot(d, g) > 0:
                # Reset H if not descent
                H = np.eye(dim)
                d = -g

            # Backtracking line search
            alpha = 1.0
            found = False
            for _ in range(max_ls):
                if time.time() > deadline:
                    break
                if calls >= budget:
                    break
                x_new = np.clip(x + alpha * d, -5, 5)
                calls += 1
                f_new = func(x_new)

                if f_new <= fx + c1 * alpha * np.dot(g, d):
                    found = True
                    break
                alpha *= rho

            if not found:
                # Try smaller step or restart
                alpha = 0.01
                x_new = np.clip(x + alpha * d, -5, 5)
                calls += 1
                f_new = func(x_new)
                if f_new < fx:
                    found = True
                else:
                    # No progress, shrink H and try again
                    H = 0.5 * H
                    continue

            # BFGS update
            s = x_new - x
            g_new = compute_gradient(x_new)
            if g_new is None:
                break

            # Update H using BFGS formula
            ys = np.dot(y := (g_new - g), s)
            if ys > 1e-10:
                Hy = H @ (g_new - g)
                H = H - np.outer(Hy, s) / np.dot(s, Hy) + np.outer(s, s) / ys

            x, fx = x_new, f_new
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

            # Adaptive h: shrink when close to minimum
            if np.linalg.norm(g) < 1e-3:
                h = max(h * 0.5, 1e-4)

        return best_x, best_fx
# EVOLVE-BLOCK-END
