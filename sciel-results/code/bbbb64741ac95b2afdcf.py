"""Combined optimizer: CMA-ES (rotation-aware) + restart + (1+1)-ES fine-tune for Rastrigin f3."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restart + (1+1)-ES fine-tuning for Rastrigin f3 under rotation."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 270  # safety margin before 300s kill
        rng = np.random.default_rng(7)

        calls = [0]

        def call_func(x):
            calls[0] += 1
            return func(x)

        def time_ok():
            return time.time() < deadline and calls[0] < budget

        # ---------- Phase 1: Fast random start (seed early progress) ----------
        # Evaluate a few random points immediately to get a decent f_best early.
        x_best = None
        f_best = np.inf
        for _ in range(5):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, size=dim)
            fx = call_func(x)
            if fx < f_best:
                f_best = fx
                x_best = x.copy()

        # ---------- Phase 2: CMA-ES from best random point ----------
        # CMA-ES adapts covariance, so it handles the hidden rotation.
        # We implement a compact CMA-ES that respects the call budget.

        def cma_es(x0, f0, max_calls):
            """Compact CMA-ES. Returns (best_x, best_f, calls_used)."""
            nonlocal calls
            d = dim
            pop = max(4 + int(3 * np.log(d)), 8)  # population size
            sigma = 1.0
            x_mean = x0.astype(float).copy()
            f_mean = f0
            C = np.eye(d)
            # Inverse sqrt of C for sampling
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            # Evolution paths
            p_sigma = np.zeros(d)
            p_c = np.zeros(d)
            # Parameters
            mu = pop // 2
            w = np.log(pop / 2 + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)
            c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
            c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
            # Track best
            best_x = x0.copy()
            best_f = f0
            local_calls = 0

            for gen in range(200):
                if local_calls >= max_calls or not time_ok():
                    break
                # Sample population
                for i in range(pop):
                    if local_calls >= max_calls or not time_ok():
                        break
                    z = rng.standard_normal(d)
                    x_i = x_mean + sigma * (C_inv_sqrt @ z)
                    x_i = np.clip(x_i, -5, 5)
                    f_i = call_func(x_i)
                    local_calls += 1
                    if f_i < best_f:
                        best_f = f_i
                        best_x = x_i.copy()
                    # Store for recombination (use first mu)
                    if i < mu:
                        if not hasattr(cma_es, '_buf'):
                            cma_es._buf = []
                        cma_es._buf.append((f_i, x_i))

                if len(getattr(cma_es, '_buf', [])) < mu:
                    break
                cma_es._buf = sorted(cma_es._buf, key=lambda t: t[0])[:mu]
                cma_es._buf = [t[1] for t in cma_es._buf]

                # Recombine
                x_new = np.zeros(d)
                for i in range(mu):
                    x_new += w[i] * cma_es._buf[i]

                # Update sigma (sigma rule)
                y = (x_new - x_mean) / max(sigma, 1e-12)
                h_sigma = (
                    np.sqrt((1 - np.exp(-2 / (gen + 1))) *
                            (1 - np.exp(-2 / (gen + 1 + 1 / c_sigma)))) / c_sigma
                ) if gen < 100 else 1.0
                p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Update C (rank-mu + rank-1)
                y_c = (x_new - x_mean) / max(sigma, 1e-12)
                h_c = (
                    np.sqrt((1 - np.exp(-2 / (gen + 1))) *
                            (1 - np.exp(-2 / (gen + 1 + 1 / c_c)))) / c_c
                ) if gen < 100 else 1.0
                p_c = (1 - c_c) * p_c + h_c * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
                delta_h = (1 + h_c) * np.sqrt(1 - np.exp(-2 * (gen + 1) / (d + 1))) - 1
                C = (1 + c_c * (2 - c_c) - c_c * mu_eff) * C + \
                    c_c * mu_eff * np.outer(y_c, y_c) + \
                    c_c * delta_h * (np.outer(p_c, p_c) - C)
                # Update mean
                x_mean = x_new
                # Refresh C_inv_sqrt periodically (expensive)
                if gen % 5 == 0:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                if sigma < 1e-8:
                    break

            # Clean up
            if hasattr(cma_es, '_buf'):
                del cma_es._buf
            return best_x, best_f, local_calls

        # Run Phase 2 CMA-ES
        cma_budget1 = min(200, budget - calls[0] - 50)
        if cma_budget1 > 10:
            x2, f2, _ = cma_es(x_best, f_best, cma_budget1)
            if f2 < f_best:
                f_best = f2
                x_best = x2.copy()

        # ---------- Phase 3: CMA-ES restart from perturbed best ----------
        # Escape the local basin by restarting CMA-ES from a perturbed point.
        if time_ok() and calls[0] < budget - 30:
            # Perturb the best point moderately to escape the basin
            x_restart = x_best + 0.3 * rng.standard_normal(dim)
            x_restart = np.clip(x_restart, -5, 5)
            f_restart = call_func(x_restart)
            if f_restart < f_best:
                f_best = f_restart
                x_best = x_restart.copy()

            cma_budget2 = min(150, budget - calls[0] - 30)
            if cma_budget2 > 10:
                x3, f3, _ = cma_es(x_restart, f_restart, cma_budget2)
                if f3 < f_best:
                    f_best = f3
                    x_best = x3.copy()

        # ---------- Phase 4: (1+1)-ES fine-tuning with remaining budget ----------
        if time_ok():
            sigma = 0.05
            for _ in range(budget - calls[0]):
                if not time_ok():
                    break
                step = sigma * rng.standard_normal(dim)
                y = np.clip(x_best + step, -5, 5)
                fy = call_func(y)
                if fy < f_best:
                    f_best = fy
                    x_best = y.copy()
                    sigma *= 1.05
                else:
                    sigma *= 0.9
                if sigma < 0.0003:
                    break

        return x_best, f_best
# EVOLVE-BLOCK-END
