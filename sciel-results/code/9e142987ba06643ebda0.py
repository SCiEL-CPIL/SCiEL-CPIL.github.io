import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = None
        self.best_x = None
        self.best_f = np.inf

    def __call__(self, func):
        self.start_time = time.time()
        d = self.dim
        N = self.budget
        deadline = self.start_time + 280.0

        def time_check():
            return time.time() < deadline

        def safe_eval(x):
            if self.calls >= N or not time_check():
                return np.inf
            self.calls += 1
            try:
                x = np.asarray(x, dtype=float).flatten()
                x = np.clip(x, -5.0, 5.0)
                val = float(func(x))
                return val
            except Exception:
                return np.inf

        def update_best(x, f):
            if f < self.best_f:
                self.best_f = f
                self.best_x = np.asarray(x, dtype=float).copy()

        # Initial evaluation at origin
        x0 = np.zeros(d)
        f0 = safe_eval(x0)
        update_best(x0, f0)

        rng = np.random.default_rng(42)

        # Generate diverse starting points
        start_points = [x0.copy()]

        # Random points
        for _ in range(8):
            start_points.append(rng.uniform(-4, 4, d))

        # Points along axes
        for i in range(d):
            for sign in [-1, 1]:
                for mag in [2.0, 3.5]:
                    pt = np.zeros(d)
                    pt[i] = mag * sign
                    start_points.append(pt)

        # Phase 1: Multi-restart L-BFGS-B
        # Each restart gets a limited budget to explore different basins
        max_evals_per_run = 40

        for sp in start_points:
            if self.calls >= N - 50 or not time_check():
                break

            try:
                res = minimize(
                    safe_eval,
                    sp,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * d,
                    options={'maxiter': 80, 'maxfun': max_evals_per_run, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if np.isfinite(res.fun) and res.fun < self.best_f:
                    update_best(res.x, res.fun)
            except Exception:
                pass

        # Phase 2: Stochastic quasi-Newton with adaptive step sizing
        # This implements a custom L-BFGS-like method that adapts to the
        # anisotropic landscape of Bent Cigar by maintaining a low-rank
        # Hessian approximation and using line search.
        
        if self.calls < N - 20 and time_check():
            x_current = self.best_x.copy()
            f_current = self.best_f

            # L-BFGS memory
            m = min(5, d)  # memory size
            s_history = []  # step vectors
            y_history = []  # gradient difference vectors
            rho_history = []  # 1/(s^T y)

            # Finite difference gradient
            eps_fd = 1e-6

            def finite_diff_gradient(x):
                if self.calls >= N or not time_check():
                    return None
                grad = np.zeros(d)
                for i in range(d):
                    if self.calls >= N or not time_check():
                        return None
                    x_plus = x.copy()
                    x_plus[i] += eps_fd
                    x_minus = x.copy()
                    x_minus[i] -= eps_fd
                    f_plus = safe_eval(x_plus)
                    f_minus = safe_eval(x_minus)
                    if not np.isfinite(f_plus) or not np.isfinite(f_minus):
                        return None
                    grad[i] = (f_plus - f_minus) / (2 * eps_fd)
                return grad

            # Two-loop recursion for L-BFGS inverse Hessian approximation
            def lbfgs_direction(g, s_hist, y_hist, rho_hist):
                if not s_hist:
                    return -g
                n = len(s_hist)
                q = g.copy()
                alpha = np.zeros(n)
                for i in range(n - 1, -1, -1):
                    alpha[i] = rho_hist[i] * np.dot(s_hist[i], q)
                    q -= alpha[i] * y_hist[i]
                r = q
                for i in range(n):
                    beta = rho_hist[i] * np.dot(y_hist[i], r)
                    r += (alpha[i] - beta) * s_hist[i]
                return r

            # Line search (backtracking)
            def line_search(x, f, direction, alpha_init=1.0, c1=1e-4):
                if self.calls >= N or not time_check():
                    return None, None, None
                alpha = alpha_init
                g = finite_diff_gradient(x)
                if g is None:
                    return None, None, None
                slope = np.dot(g, direction)
                if slope >= 0:
                    return None, None, None
                for _ in range(30):
                    if self.calls >= N or not time_check():
                        return None, None, None
                    x_trial = np.clip(x + alpha * direction, -5.0, 5.0)
                    f_trial = safe_eval(x_trial)
                    if not np.isfinite(f_trial):
                        alpha *= 0.5
                        continue
                    if f_trial <= f + c1 * alpha * slope:
                        return x_trial, f_trial, g
                    alpha *= 0.5
                return None, None, None

            # Run stochastic quasi-Newton iterations
            max_iters = 200
            for it in range(max_iters):
                if self.calls >= N - 10 or not time_check():
                    break

                g = finite_diff_gradient(x_current)
                if g is None:
                    break

                # Check convergence
                if np.linalg.norm(g, np.inf) < 1e-8:
                    break

                # Compute search direction using L-BFGS
                direction = lbfgs_direction(g, s_history, y_history, rho_history)

                # Ensure descent direction
                if np.dot(direction, g) > 0:
                    direction = -g

                # Normalize direction for step size control
                dir_norm = np.linalg.norm(direction)
                if dir_norm > 0:
                    direction /= dir_norm

                # Adaptive initial step size based on gradient magnitude
                g_norm = np.linalg.norm(g)
                alpha_init = min(1.0, 0.1 / (g_norm + 1e-8))

                x_new, f_new, g_new = line_search(x_current, f_current, direction, alpha_init)

                if x_new is None:
                    # Try smaller step
                    alpha_init *= 0.1
                    x_new, f_new, g_new = line_search(x_current, f_current, direction, alpha_init)
                    if x_new is None:
                        break

                # Update L-BFGS history
                s = x_new - x_current
                y = g_new - g
                sy = np.dot(s, y)
                if sy > 1e-10:
                    rho = 1.0 / sy
                    s_history.append(s)
                    y_history.append(y)
                    rho_history.append(rho)
                    if len(s_history) > m:
                        s_history.pop(0)
                        y_history.pop(0)
                        rho_history.pop(0)

                x_current = x_new
                f_current = f_new
                update_best(x_current, f_current)

        # Phase 3: Final polish with L-BFGS-B
        if self.calls < N - 5 and time_check():
            try:
                res = minimize(
                    safe_eval,
                    self.best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * d,
                    options={'maxiter': 500, 'maxfun': N - self.calls - 2, 'ftol': 1e-16, 'gtol': 1e-12}
                )
                if np.isfinite(res.fun) and res.fun < self.best_f:
                    update_best(res.x, res.fun)
            except Exception:
                pass

        return self.best_x, self.best_f
