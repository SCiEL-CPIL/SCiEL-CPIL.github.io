"""Stochastic quasi-Newton optimizer with random restarts for Bent Cigar (f12)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    L-BFGS with finite-difference gradients and random restarts.
    
    Strategy:
    - Use multiple random restarts to handle the hidden shift/rotation.
    - Within each restart, use L-BFGS with finite-difference gradients to
      adaptively scale steps along the narrow valley of the Bent Cigar function.
    - Track elapsed time to stay within the 330s wall-clock limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = time.time() + 320  # 10s safety margin

    def _finite_diff_gradient(self, func, x, h=1e-6):
        """Compute finite-difference gradient."""
        grad = np.zeros(self.dim)
        for i in range(self.dim):
            xp = x.copy()
            xp[i] += h
            xm = x.copy()
            xm[i] -= h
            grad[i] = (func(np.clip(xp, -5, 5)) - func(np.clip(xm, -5, 5)) / (2 * h))
        return grad

    def _lbfgs_step(self, func, x, fx, m=5, max_iter=50):
        """
        L-BFGS with finite-difference gradients.
        Returns (x_best, fx_best, calls_used).
        """
        calls = 0
        s_list = []  # previous step vectors
        y_list = []  # previous gradient differences
        rho_list = []
        g = self._finite_diff_gradient(func, x)
        calls += 2 * self.dim

        for it in range(max_iter):
            if time.time() > self.deadline:
                break

            # Two-loop recursion to compute search direction
            q = g.copy()
            alpha = np.zeros(len(s_list))
            for i in range(len(s_list) - 1, -1, -1):
                alpha[i] = rho_list[i] * np.dot(s_list[i], q)
                q -= alpha[i] * y_list[i]

            # Initial Hessian approximation (inverse)
            if len(s_list) > 0:
                yy = np.dot(y_list[-1], y_list[-1])
                sy = np.dot(s_list[-1], y_list[-1])
                gamma = sy / yy if yy > 1e-12 else 1.0
            else:
                gamma = 1.0

            r = gamma * q

            for i in range(len(s_list)):
                beta = rho_list[i] * np.dot(y_list[i], r)
                r += (alpha[i] - beta) * s_list[i]

            # Line search (backtracking)
            direction = -r
            step = 1.0
            x_new = x + step * direction
            x_new = np.clip(x_new, -5, 5)
            fx_new = func(x_new)
            calls += 1

            # Backtracking line search
            c1 = 1e-4
            max_backtracks = 10
            for bt in range(max_backtracks):
                if time.time() > self.deadline:
                    break
                if fx_new < fx + c1 * step * np.dot(g, direction):
                    break
                step *= 0.5
                x_new = x + step * direction
                x_new = np.clip(x_new, -5, 5)
                fx_new = func(x_new)
                calls += 1

            # Update history
            s = step * direction
            g_new = self._finite_diff_gradient(func, x_new)
            calls += 2 * self.dim
            y = g_new - g
            sy = np.dot(s, y)
            yy = np.dot(y, y)

            if sy > 1e-12:
                rho = 1.0 / sy
                s_list.append(s)
                y_list.append(y)
                rho_list.append(rho)
                if len(s_list) > m:
                    s_list.pop(0)
                    y_list.pop(0)
                    rho_list.pop(0)

            x, fx = x_new, fx_new
            g = g_new

        return x, fx, calls

    def __call__(self, func):
        rng = np.random.default_rng(42)
        best_x = None
        best_fx = np.inf
        calls_used = 0

        # Number of restarts: adapt based on remaining budget
        # Each L-BFGS phase uses roughly 50 * (2*dim + 1 + some backtracking) calls
        # Let's use up to 10 restarts, each with a sub-budget
        n_restarts = min(10, self.budget // 50)
        budget_per_restart = self.budget // n_restarts

        for restart in range(n_restarts):
            if time.time() > self.deadline:
                break

            # Random start within [-4, 4]^5 (minimizer is in this range)
            x0 = rng.uniform(-4, 4, self.dim)
            fx0 = func(x0)
            calls_used += 1

            if fx0 < best_fx:
                best_fx = fx0
                best_x = x0.copy()

            # Run L-BFGS from this start
            x_opt, fx_opt, phase_calls = self._lbfgs_step(
                func, x0, fx0, m=5, max_iter=min(30, budget_per_restart // (2 * self.dim + 2))
            )
            calls_used += phase_calls

            if fx_opt < best_fx:
                best_fx = fx_opt
                best_x = x_opt.copy()

            # If we're doing well, use remaining budget for a final polish
            if calls_used < self.budget and time.time() < self.deadline:
                remaining = self.budget - calls_used
                if remaining > 2 * self.dim + 2:
                    x_final, fx_final, final_calls = self._lbfgs_step(
                        func, x_opt, fx_opt, m=5, max_iter=min(20, remaining // (2 * self.dim + 2))
                    )
                    calls_used += final_calls
                    if fx_final < best_fx:
                        best_fx = fx_final
                        best_x = x_final.copy()

        # If we still have budget, do one more local refinement
        if best_x is not None and calls_used < self.budget and time.time() < self.deadline:
            remaining = self.budget - calls_used
            if remaining > 2 * self.dim + 2:
                x_final, fx_final, final_calls = self._lbfgs_step(
                    func, best_x, best_fx, m=5, max_iter=min(20, remaining // (2 * self.dim + 2))
                )
                calls_used += final_calls
                if fx_final < best_fx:
                    best_fx = fx_final
                    best_x = x_final.copy()

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_fx = func(best_x)

        return best_x, best_fx
# EVOLVE-BLOCK-END
