"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with adaptive step size and multi-start."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 270.0  # safety margin before 330s kill

        best_x = None
        best_f = np.inf
        calls = 0

        def eval_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            calls += 1
            try:
                val = float(func(x))
            except:
                return np.inf
            if np.isnan(val) or np.isinf(val):
                return np.inf
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Initial exploration - evaluate several random points
        n_initial = min(10, budget // 10)
        for i in range(n_initial):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = eval_func(x)

        # Phase 2: Main optimization using a population-based CMA-ES-like approach
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls += 1

        # Use CMA-ES with population size 4 for dim=5
        pop_size = 4
        sigma = 2.0
        c1 = (4 + np.log(dim)) / ((dim + 4) ** 2 + 4)
        cmu = (4 + np.log(dim / 2)) / ((dim + 4) ** 2 + 4)
        cs = (dim + 1.3) / (dim + 5.5)
        ds = 1 + 2 * max(0, np.sqrt((cs - 1) / (cs + 1)) - 1)
        cmu = min(cmu, 2 * c1)

        mean = best_x.copy()
        sigma = 2.0
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        max_iter = budget - calls - 5
        for it in range(max_iter):
            if calls >= budget - 5 or time.time() > deadline:
                break

            # Generate population
            ys = []
            fs = []
            for i in range(pop_size):
                y = mean + sigma * (B @ (D * rng.standard_normal(dim)))
                y = np.clip(y, -5, 5)
                fy = eval_func(y)
                if calls >= budget - 5 or time.time() > deadline:
                    break
                ys.append(y)
                fs.append(fy)

            if len(ys) < pop_size:
                break

            # Sort by fitness
            order = np.argsort(fs)
            ys = [ys[i] for i in order]
            fs = [fs[i] for i in order]

            # Update mean with top half
            mu = pop_size // 2
            weights = np.ones(mu)
            weights /= weights.sum()
            new_mean = sum(w * y for w, y in zip(weights, ys[:mu]))

            # Update covariance matrix (simplified)
            for i in range(mu):
                diff = ys[i] - mean
                C = (B @ (D * D) @ B.T)
                C += 2 * cmu * weights[i] * np.outer(diff, diff)
                # Recompute B and D (simplified via eigen-decomposition)
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                D = np.sqrt(eigvals)
                B = eigvecs

            mean = new_mean

            # Update sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (B @ (D @ (mean - mean) / sigma))
            # Simplified: just use success rate
            n_success = sum(1 for y in ys[:mu] if np.allclose(y, mean, atol=1e-6))
            if n_success > 0:
                sigma *= 1.1
            else:
                sigma *= 0.9
            sigma = np.clip(sigma, 1e-4, 10.0)

            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = ys[0].copy()

        # Phase 3: Final local refinement with (1+1)-ES
        if best_x is not None and calls < budget - 3 and time.time() < deadline:
            x0 = best_x.copy()
            fx0 = eval_func(x0)
            sigma = 0.3
            for _ in range(min(30, budget - calls - 3)):
                if calls >= budget - 3 or time.time() > deadline:
                    break
                y = np.clip(x0 + sigma * rng.standard_normal(dim), -5, 5)
                fy = eval_func(y)
                if fy < fx0:
                    x0, fx0 = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.8
                    if sigma < 1e-6:
                        break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
