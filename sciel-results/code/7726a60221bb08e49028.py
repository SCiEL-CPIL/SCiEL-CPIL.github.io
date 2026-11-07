"""Nelder-Mead simplex with restarts for BBOB Weierstrass f16."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Nelder-Mead simplex with restarts for the BBOB Weierstrass (f16) task."""

    def __init__(self, budget, dim):
        self.budget, self.dim = int(budget), int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)
        deadline = time.time() + 280.0  # safety margin below the 330s kill

        n_calls = [0]

        def safe_func(x):
            if n_calls[0] >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            n_calls[0] += 1
            return float(func(x))

        best_x = np.zeros(dim)
        best_f = np.inf

        # Nelder-Mead parameters
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink

        def nelder_mead(start, max_iter=200):
            nonlocal best_x, best_f
            n = dim
            # Initialize simplex
            simplex = np.zeros((n + 1, n))
            simplex[0] = start
            for i in range(n):
                simplex[i + 1] = start.copy()
                simplex[i + 1, i] += 0.1 if start[i] >= 0 else -0.1
                simplex[i + 1] = np.clip(simplex[i + 1], -5.0, 5.0)

            # Evaluate initial simplex
            fvals = np.array([safe_func(simplex[i]) for i in range(n + 1)])

            for it in range(max_iter):
                if n_calls[0] >= budget or time.time() > deadline:
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                # Update global best
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()

                # Check convergence
                if np.max(fvals) - np.min(fvals) < 1e-10:
                    break

                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = safe_func(x_r)

                if fvals[0] <= f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                elif f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = safe_func(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - simplex[-1])
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = safe_func(x_c)

                    if f_c < min(f_r, fvals[-1]):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            fvals[i] = safe_func(simplex[i])

                # Update global best after each iteration
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_f:
                    best_f = fvals[min_idx]
                    best_x = simplex[min_idx].copy()

            return best_x, best_f

        # Phase 1: Initial Nelder-Mead from origin
        nelder_mead(np.zeros(dim), max_iter=150)

        # Phase 2: Restart from diverse points if budget remains
        if n_calls[0] < budget and time.time() < deadline:
            # Try a few diverse starting points
            restart_starts = []
            for scale in [1.0, 2.0, 3.0]:
                restart_starts.append(np.full(dim, scale))
                restart_starts.append(np.full(dim, -scale))
            for _ in range(10):
                restart_starts.append(rng.uniform(-4.0, 4.0, dim))

            for s in restart_starts:
                if n_calls[0] >= budget or time.time() > deadline:
                    break
                # Quick local search from this point
                old_best_f = best_f
                old_best_x = best_x.copy()
                nelder_mead(s, max_iter=50)
                # If no improvement, continue; if improved, keep going

        # Phase 3: Final polish with very small steps around the best
        if n_calls[0] < budget and time.time() < deadline:
            polish_sigma = 0.01
            polish_window = []
            polish_window_size = 5

            while n_calls[0] < budget and time.time() < deadline:
                candidate = best_x + polish_sigma * rng.standard_normal(dim)
                candidate = np.clip(candidate, -5.0, 5.0)
                fx = safe_func(candidate)

                if fx < best_f:
                    best_x = candidate.copy()
                    best_f = fx
                    polish_window.append(1)
                    polish_sigma *= 0.7
                else:
                    polish_window.append(0)
                    polish_sigma *= 0.95

                if len(polish_window) >= polish_window_size:
                    success_rate = sum(polish_window) / polish_window_size
                    if success_rate > 0.4:
                        polish_sigma *= 1.3
                    elif success_rate < 0.15:
                        polish_sigma *= 0.75
                    polish_window = polish_window[-polish_window_size // 2:]

                polish_sigma = max(polish_sigma, 1e-8)
                polish_sigma = min(polish_sigma, 1.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
