"""Optimized black-box continuous optimizer for BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with finite-difference gradient polishing and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_time():
            return time.time() >= deadline

        def update_best(x, f):
            nonlocal best_x, best_f, calls_made
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # Phase 1: Multi-start CMA-ES with limited budget per restart
        # Weierstrass is multimodal; multiple starts help find global basin
        n_restarts = 3
        budget_per_restart = budget // n_restarts

        for restart in range(n_restarts):
            if check_time():
                break
            if calls_made >= budget:
                break

            # Initial point: spread starts across the domain
            if restart == 0:
                x = rng.uniform(-3, 3, dim)
            else:
                # Perturb previous best or random
                if best_x is not None and restart == 1:
                    x = best_x + rng.uniform(-2, 2, dim)
                else:
                    x = rng.uniform(-4, 4, dim)
                x = np.clip(x, -5, 5)

            fx = func(x)
            calls_made += 1
            update_best(x, fx)

            # CMA-ES parameters
            sigma = 1.0
            pop_size = max(4, int(4 + 2 * np.log(dim)))
            # CMA-ES weights
            mu = pop_size // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            c_mu = 2 / ((dim + 1) ** 2 + 10)
            c_sigma = (dim + 1.3) / (dim + 3 + mu)
            c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu)
            c_mu = 2 / ((dim + 2) ** 2 + mu)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1)) - 1) + c_sigma
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)

            for gen in range(budget_per_restart - 1):
                if check_time():
                    break
                if calls_made + pop_size > budget:
                    # Do partial generation
                    remaining = budget - calls_made
                    if remaining <= 0:
                        break
                    pop_size_partial = remaining
                else:
                    pop_size_partial = pop_size

                # Generate population
                offspring = []
                offspring_f = []
                for i in range(pop_size_partial):
                    if check_time():
                        break
                    B = np.linalg.cholesky(C)
                    z = rng.standard_normal(dim)
                    y = x + sigma * (B @ z)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    offspring.append(y)
                    offspring_f.append(fy)
                    update_best(y, fy)

                if len(offspring) == 0:
                    break

                # Sort by fitness
                idx = np.argsort(offspring_f)
                offspring = [offspring[i] for i in idx]
                offspring_f = [offspring_f[i] for i in idx]

                # Update mean
                x_new = sum(weights[i] * offspring[i] for i in range(min(mu, len(offspring))))

                # Update covariance
                diff = np.array([offspring[i] - x for i in range(min(mu, len(offspring)))])
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) - C) + \
                    c_mu * (diff.T @ (weights * diff))

                # Ensure positive semi-definite
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update step size control
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu) / sigma * (B @ (x_new - x))
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < \
                        (1.4 + 1 / dim) * chi_n:
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                    sigma = np.clip(sigma, 1e-10, 10.0)

                # Update path
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu) / sigma * (x_new - x)

                x = x_new
                if len(offspring_f) > 0:
                    fx = offspring_f[0]

            # After CMA-ES, do gradient polishing if time allows
            if not check_time() and calls_made < budget:
                x, fx = self._gradient_polish(func, x, fx, rng, dim, budget - calls_made, check_time, update_best)
                calls_made = budget - (budget - calls_made)  # approximate; tracked internally

        # Phase 2: Final gradient polishing on best point
        if best_x is not None and not check_time() and calls_made < budget:
            # Re-evaluate best to get current value
            fb = func(best_x)
            calls_made += 1
            update_best(best_x, fb)
            best_x, best_f = self._gradient_polish(func, best_x, best_f, rng, dim,
                                                    budget - calls_made, check_time, update_best)

        return best_x, best_f

    def _gradient_polish(self, func, x, fx, rng, dim, remaining_budget, check_time, update_best):
        """Finite-difference gradient descent to polish the solution."""
        nonlocal calls_made
        x = x.copy()
        fx = func(x)
        calls_made += 1
        update_best(x, fx)

        lr = 0.5
        fd_eps = 1e-4
        max_iter = min(remaining_budget - 1, 50)

        for it in range(max_iter):
            if check_time() or calls_made >= self.budget:
                break

            # Compute gradient via central differences
            grad = np.zeros(dim)
            for i in range(dim):
                if check_time() or calls_made >= self.budget:
                    break
                xp = x.copy()
                xp[i] += fd_eps
                xp = np.clip(xp, -5, 5)
                fp = func(xp)
                calls_made += 1

                xm = x.copy()
                xm[i] -= fd_eps
                xm = np.clip(xm, -5, 5)
                fm = func(xm)
                calls_made += 1

                grad[i] = (fp - fm) / (2 * fd_eps)

            if calls_made >= self.budget:
                break

            # Check if gradient is small
            if np.linalg.norm(grad) < 1e-6:
                break

            # Gradient step
            x_new = x - lr * grad
            x_new = np.clip(x_new, -5, 5)

            # Line search: try a few step sizes
            improved = False
            for factor in [1.0, 0.5, 0.25, 0.1]:
                if check_time() or calls_made >= self.budget:
                    break
                x_try = x - lr * factor * grad
                x_try = np.clip(x_try, -5, 5)
                f_try = func(x_try)
                calls_made += 1
                if f_try < fx:
                    x = x_try
                    fx = f_try
                    update_best(x, fx)
                    improved = True
                    break

            if not improved:
                lr *= 0.5
                if lr < 1e-6:
                    break
            else:
                lr = min(lr * 1.1, 2.0)

        return x, fx
# EVOLVE-BLOCK-END
