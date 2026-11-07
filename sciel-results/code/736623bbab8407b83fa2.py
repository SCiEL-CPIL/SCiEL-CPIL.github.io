"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Stochastic L-BFGS with finite-difference gradients and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 300s limit

        # Counting wrapper
        call_count = [0]

        def f(x):
            call_count[0] += 1
            return float(func(x))

        def finite_diff_gradient(x, h):
            """Compute gradient estimate using central differences."""
            g = np.zeros(dim)
            for i in range(dim):
                xp = x.copy()
                xm = x.copy()
                xp[i] = min(5.0, xp[i] + h)
                xm[i] = max(-5.0, xm[i] - h)
                # If clipped, use one-sided
                if xp[i] == x[i]:
                    g[i] = (f(xp) - f(x)) / (xp[i] - x[i] + 1e-15)
                elif xm[i] == x[i]:
                    g[i] = (f(x) - f(xm)) / (x[i] - xm[i] + 1e-15)
                else:
                    g[i] = (f(xp) - f(xm)) / (xp[i] - xm[i])
            return g

        def lbfgs_step(x, fx, m_hist, h_param):
            """
            One L-BFGS step using finite-difference gradient.
            Returns new_x, new_fx, updated history.
            """
            if call_count[0] >= budget:
                return x, fx, m_hist

            # Compute gradient
            g = finite_diff_gradient(x, h_param)

            # L-BFGS two-loop recursion
            n_hist = min(len(m_hist), 5)
            if n_hist == 0:
                # No history: gradient descent step
                step = -g
                # Normalize to reasonable step size
                gn = np.linalg.norm(g)
                if gn > 1e-10:
                    step = step / gn * min(h_param * 2.0, 1.0)
                new_x = np.clip(x + step, -5.0, 5.0)
                new_fx = f(new_x)
                # Update history
                s = new_x - x
                y = finite_diff_gradient(new_x, h_param) - g
                if call_count[0] < budget:
                    m_hist.append((s, y, np.dot(s, y)))
                    if len(m_hist) > 5:
                        m_hist.pop(0)
                return new_x, new_fx, m_hist

            # Two-loop recursion
            rho = [1.0 / max(h[2], 1e-10) for h in m_hist[-n_hist:]]
            alphas = np.zeros(n_hist)
            q_vec = g.copy()

            for i in range(n_hist - 1, -1, -1):
                s_i, y_i, _ = m_hist[-n_hist + i]
                alphas[i] = rho[i] * np.dot(s_i, q_vec)
                q_vec = q_vec - alphas[i] * y_i

            # Initial Hessian approximation
            # Use inverse Hessian from last pair
            s_last, y_last, _ = m_hist[-1]
            sy = np.dot(s_last, y_last)
            if sy > 1e-10:
                y_norm = np.dot(y_last, y_last)
                H0 = (sy / max(y_norm, 1e-10)) * np.eye(dim)
            else:
                H0 = np.eye(dim)

            r = H0 @ q_vec

            for i in range(n_hist):
                s_i, y_i, _ = m_hist[-n_hist + i]
                beta = rho[i] * np.dot(y_i, r)
                r = r + (alphas[i] - beta) * s_i

            # Line search (backtracking)
            step = -r
            step_norm = np.linalg.norm(step)
            if step_norm < 1e-12:
                step = -g
                step_norm = np.linalg.norm(step)
                if step_norm < 1e-12:
                    return x, fx, m_hist
                step = step / step_norm * min(h_param, 0.5)

            # Backtracking line search
            alpha = 1.0
            c1 = 1e-4
            for _ in range(10):
                new_x = np.clip(x + alpha * step, -5.0, 5.0)
                new_fx = f(new_x)
                if new_fx <= fx + c1 * alpha * np.dot(g, step):
                    break
                alpha *= 0.5

            # Update history
            s_new = new_x - x
            y_new = finite_diff_gradient(new_x, h_param) - g
            if call_count[0] < budget:
                m_hist.append((s_new, y_new, np.dot(s_new, y_new)))
                if len(m_hist) > 5:
                    m_hist.pop(0)

            return new_x, new_fx, m_hist

        # Multi-start strategy
        rng = np.random.default_rng(42)
        n_starts = 3
        starts = []
        # First start: center
        starts.append(np.zeros(dim))
        # Additional starts: random
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-3, 3, dim))

        best_x = None
        best_fx = np.inf
        total_calls_per_start = max(1, budget // n_starts)

        for start_idx in range(n_starts):
            if call_count[0] >= budget:
                break
            if time.time() > deadline - 30:
                break

            x = starts[start_idx].copy()
            fx = f(x)
            m_hist = []
            h_param = 0.5  # gradient perturbation size
            no_improve = 0

            # Adaptive gradient step size
            for step_i in range(total_calls_per_start):
                if call_count[0] >= budget:
                    break
                if time.time() > deadline - 10:
                    break

                # Adapt h_param: shrink as we get closer
                h_param = max(0.01, 0.5 * (0.95 ** step_i))

                new_x, new_fx, m_hist = lbfgs_step(x, fx, m_hist, h_param)

                if new_fx < fx - 1e-8:
                    x, fx = new_x, new_fx
                    no_improve = 0
                else:
                    no_improve += 1
                    # Shrink gradient perturbation if stuck
                    h_param = max(0.005, h_param * 0.5)

                if no_improve > 20:
                    # Restart from current best with perturbation
                    x = x + 0.1 * rng.standard_normal(dim)
                    x = np.clip(x, -5.0, 5.0)
                    fx = f(x)
                    m_hist = []
                    no_improve = 0

            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        # Final polish: Nelder-Mead on best point if budget remains
        if best_x is not None and call_count[0] < budget and time.time() < deadline - 5:
            # Simple coordinate-wise local search
            for _ in range(20):
                if call_count[0] >= budget or time.time() > deadline - 2:
                    break
                improved = False
                for i in range(dim):
                    if call_count[0] >= budget:
                        break
                    for direction in [-1, 1]:
                        x_try = best_x.copy()
                        x_try[i] += direction * 0.05
                        x_try = np.clip(x_try, -5.0, 5.0)
                        f_try = f(x_try)
                        if f_try < best_fx:
                            best_x = x_try
                            best_fx = f_try
                            improved = True
                            break
                if not improved:
                    break

        return best_x, best_fx
# EVOLVE-BLOCK-END
