"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES (population) + (1+1)-ES polish, with a wall-clock deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 280.0  # safety margin below the 330 s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(2024)

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        # --- Phase 1: CMA-ES ---
        # Initial point: center of the box with small perturbation
        x0 = rng.uniform(-0.5, 0.5, dim)
        best_x = x0.copy()
        best_f = func(clip(best_x))
        calls = 1

        # CMA-ES parameters (standard, well-tuned)
        lam = 4 * dim + 2  # 22 for dim=5
        mu = lam // 2      # 11
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0.0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        x_mean = x0.copy()
        sigma = 2.0
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)

        # Reserve some budget for polish phase
        reserve = max(30, budget // 8)
        cma_budget = budget - reserve

        while calls < cma_budget:
            if time.time() > deadline - 10.0:
                break

            # Generate population
            zs = rng.standard_normal((lam, dim))
            # y_i = x_mean + sigma * B @ (D * z_i)
            ys = x_mean + sigma * (zs @ D) @ B.T
            ys = np.clip(ys, -5.0, 5.0)

            fs = np.empty(lam)
            for i in range(lam):
                if calls >= budget:
                    break
                if time.time() > deadline - 10.0:
                    break
                fs[i] = func(ys[i])
                calls += 1
                if fs[i] < best_f:
                    best_f = fs[i]
                    best_x = ys[i].copy()

            if calls >= budget:
                break

            # Sort by fitness
            order = np.argsort(fs)
            ys = ys[order]
            fs = fs[order]

            x_old = x_mean.copy()
            x_mean = np.sum(w[:, None] * ys[:mu], axis=0)
            y = (x_mean - x_old) / sigma

            # Update ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B @ (D * y))

            # Update pc and C
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 1 / dim) * np.sqrt(dim):
                pc = (1 - c_c) * pc + c_c * y
                C = ((1 - c_1 - c_mu) * C
                     + c_1 * (np.outer(pc, pc) + (c_c - c_c ** 2) * np.eye(dim))
                     + c_mu * np.sum(w[:mu, None, None] *
                                    np.outer((ys[:mu] - x_old) / sigma, (ys[:mu] - x_old) / sigma),
                                    axis=0))
                C = (C + C.T) / 2
                ev_vals, ev_vecs = np.linalg.eigh(C)
                ev_vals = np.abs(ev_vals)
                idx = ev_vals.argsort()[::-1]
                ev_vals = ev_vals[idx]
                ev_vecs = ev_vecs[:, idx]
                D = np.sqrt(np.maximum(ev_vals, 1e-20))
                B = ev_vecs

            # Adapt sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
            sigma = min(max(sigma, 1e-10), 20.0)

        # --- Phase 2: (1+1)-ES polish ---
        x = best_x.copy()
        fx = best_f
        s = 0.3
        while calls < budget:
            if time.time() > deadline:
                break
            y = clip(x + s * rng.standard_normal(dim))
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                s *= 1.15
            else:
                s *= 0.85
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            if s < 1e-8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
