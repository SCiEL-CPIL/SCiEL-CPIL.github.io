import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = float('inf')
        self.start_time = None
        self.calls_made = 0
        self.max_time = 320.0

    def __call__(self, func):
        self.start_time = time.time()
        self.calls_made = 0

        def safe_func(x):
            if time.time() - self.start_time > self.max_time:
                return float('inf')
            if self.calls_made >= self.budget:
                return float('inf')
            self.calls_made += 1
            fx = func(x)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = np.array(x, copy=True)
            return fx

        def finite_diff_gradient(x, fval, eps=1e-5):
            grad = np.zeros(self.dim)
            for i in range(self.dim):
                xp = np.array(x, copy=True)
                xm = np.array(x, copy=True)
                xp[i] += eps
                xm[i] -= eps
                fp = safe_func(xp)
                fm = safe_func(xm)
                if np.isinf(fp) or np.isinf(fm):
                    return None
                grad[i] = (fp - fm) / (2 * eps)
            return grad

        def bfgs_local(x0, max_calls, tol=1e-7):
            """BFGS with finite-difference gradients and backtracking line search."""
            x = np.array(x0, copy=True)
            fx = safe_func(x)
            if np.isinf(fx):
                return x, fx

            # Initialize inverse Hessian approximation
            H = np.eye(self.dim)
            calls_used = 1  # count the initial evaluation

            for iteration in range(max_calls):
                if time.time() - self.start_time > self.max_time:
                    break
                if self.calls_made >= self.budget:
                    break

                # Compute gradient
                grad = finite_diff_gradient(x, fx)
                if grad is None:
                    break
                calls_used += 2 * self.dim  # 2 calls per dimension

                if np.linalg.norm(grad) < tol:
                    break

                # BFGS direction: p = -H @ grad
                p = -H @ grad

                # Check for negative curvature
                if np.dot(p, grad) > 0:
                    # Reset H to identity
                    H = np.eye(self.dim)
                    p = -grad

                # Backtracking line search
                alpha = 1.0
                c1 = 1e-4
                max_backtracks = 20
                for _ in range(max_backtracks):
                    x_new = np.clip(x + alpha * p, -5, 5)
                    fx_new = safe_func(x_new)
                    calls_used += 1

                    if np.isinf(fx_new):
                        alpha *= 0.5
                        continue

                    if fx_new <= fx + c1 * alpha * np.dot(grad, p):
                        break
                    alpha *= 0.5
                else:
                    break

                # Update BFGS
                s = x_new - x
                y = None
                # Recompute gradient at new point for BFGS update
                grad_new = finite_diff_gradient(x_new, fx_new)
                if grad_new is None:
                    break
                calls_used += 2 * self.dim

                y = grad_new - grad
                sy = np.dot(s, y)
                if sy > 1e-12:
                    # Standard BFGS inverse Hessian update
                    rho = 1.0 / sy
                    I = np.eye(self.dim)
                    H = (I - rho * np.outer(s, y.T)) @ H @ (I - rho * np.outer(y, s.T)) + rho * np.outer(s, s)
                else:
                    # Reset to identity
                    H = np.eye(self.dim)

                x = x_new
                fx = fx_new

                if np.linalg.norm(s) < 1e-8:
                    break

            return x, fx

        # Multi-start strategy with diverse points
        rng = np.random.default_rng(42)
        starts = []
        # Center
        starts.append(np.zeros(self.dim))
        # Diagonals
        starts.append(np.ones(self.dim) * 2.0)
        starts.append(-np.ones(self.dim) * 2.0)
        # Random points in the box
        for _ in range(6):
            starts.append(rng.uniform(-4, 4, self.dim))

        # Allocate budget: use ~70% for multi-start BFGS, ~30% for polish
        n_starts = len(starts)
        # Give more budget to earlier starts (they're more likely to be good)
        # But cap each start's budget
        total_budget = self.budget
        polish_budget = int(total_budget * 0.25)
        search_budget = total_budget - polish_budget
        calls_per_start = max(10, search_budget // n_starts)

        # Run BFGS from each start
        for x0 in starts:
            if time.time() - self.start_time > self.max_time:
                break
            if self.calls_made >= self.budget - polish_budget:
                break

            # Limit calls for this start
            remaining_for_start = min(calls_per_start, self.budget - polish_budget - self.calls_made)
            if remaining_for_start < 10:
                break

            x_opt, f_opt = bfgs_local(x0, remaining_for_start)

            if f_opt < self.best_f:
                self.best_f = f_opt
                self.best_x = np.array(x_opt, copy=True)

        # Final polish: run BFGS from the best point found with tighter tolerance
        if self.best_x is not None and self.calls_made < self.budget:
            remaining = self.budget - self.calls_made
            if remaining > 10:
                x_polished, f_polished = bfgs_local(self.best_x, remaining, tol=1e-9)
                if f_polished < self.best_f:
                    self.best_f = f_polished
                    self.best_x = np.array(x_polished, copy=True)

        return self.best_x, self.best_f
