"""Combined optimizer: periodic basin hopping + CMA-ES + (1+1)-ES fine-tuning for Rastrigin f3."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Periodic basin hopping with CMA-ES refinement and (1+1)-ES fine-tuning."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285
        rng = np.random.default_rng(42)

        calls = [0]

        def call_func(x):
            calls[0] += 1
            return func(x)

        def time_ok():
            return time.time() < deadline and calls[0] < budget

        best_x = None
        best_f = np.inf

        # ---------- Phase 1: Basin exploration using periodic structure ----------
        # Rastrigin: local minima at integer points (after shift).
        # Sample integer lattice points in [-4, 4]^5 to find the best basin.
        # Use a structured sampling: random subset of integer points.
        n_explore = 50
        for i in range(n_explore):
            if not time_ok():
                break
            # Random integer point in [-4, 4]^5
            x = (rng.integers(-4, 5, size=dim)).astype(float)
            x = np.clip(x, -5, 5)
            fx = call_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Also try origin and axis-aligned neighbors
        for x_cand in [np.zeros(dim)] + [
            np.array([0.0] * dim) + np.eye(dim)[axis] * delta
            for axis in range(dim) for delta in [-1, 1]
        ]:
            if not time_ok():
                break
            x_cand = np.clip(x_cand, -5, 5)
            fx = call_func(x_cand)
            if fx < best_f:
                best_f = fx
                best_x = x_cand.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = call_func(best_x)

        # ---------- Phase 2: CMA-ES ----------
        def cma_es(x0, f0, budget_cma, sigma_init=0.5):
            dim_local = self.dim
            pop_size = 4 + int(3 * np.log(dim_local))
            mu = pop_size // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)

            cc = (4 + dim_local / dim_local) / (dim_local + 4)
            cs = (4 + dim_local / 3) / (dim_local + 4)
            c1 = (2 + 2 / (dim_local + 1)) / ((dim_local + 2) ** 2 + 2)
            cmu = min(1 - c1,
                      2 * (mu_eff - 2 + 1 / mu_eff) / ((dim_local + 2) ** 2 + mu_eff))

            sigma = sigma_init
            x = x0.copy()
            f_best = f0
            C = np.eye(dim_local)
            C_sqrt = np.eye(dim_local)
            C_sqrt_inv = np.eye(dim_local)
            ps = np.zeros(dim_local)
            pc = np.zeros(dim_local)

            eigvals = np.ones(dim_local)
            eigvecs = np.eye(dim_local)

            for gen in range(budget_cma // pop_size + 1):
                if not time_ok():
                    break

                if gen % 2 == 0 or gen == 0:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C_sqrt = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
                    C_sqrt_inv = (eigvecs / np.sqrt(eigvals)) @ eigvecs.T

                offspring = []
                for i in range(pop_size):
                    if not time_ok():
                        break
                    z = rng.standard_normal(dim_local)
                    y = np.clip(x + sigma * (C_sqrt @ z), -5, 5)
                    fy = call_func(y)
                    offspring.append((fy, y))

                if not offspring:
                    break

                offspring.sort(key=lambda t: t[0])

                for fy, y in offspring:
                    if fy < f_best:
                        f_best = fy
                        x = y.copy()
                        break

                x_mean = np.zeros(dim_local)
                for i in range(mu):
                    x_mean += w[i] * offspring[i][1]

                d_sigma = 1 + 2 * max(0, (cc + 1) / (cc + dim_local / 2) - 1)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (
                    C_sqrt_inv @ (x_mean - x)) / sigma

                hsig = (np.linalg.norm(ps) /
                        np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))))
                if hsig < (1.4 + 2 / (dim_local + 1)) * np.sqrt(mu_eff + 1):
                    sigma *= np.exp(cs / d_sigma * (
                        np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))

                pc = (1 - cc) * pc + cc * (x_mean - x) / (
                    sigma * np.sqrt(cc * (2 - cc) * mu_eff))

                delta_sigma = (np.linalg.norm(pc) /
                               np.sqrt(1 - (1 - cc) ** (gen + 1)) >
                               1.4 + 2 / (dim_local + 1))
                rank_c = np.outer(pc, pc)
                if delta_sigma:
                    rank_c += (1 / cc - 1) * (2 - cc) * np.eye(dim_local)

                H = np.zeros((dim_local, dim_local))
                for i in range(mu):
                    H += w[i] * np.outer(
                        offspring[i][1] - x, offspring[i][1] - x)

                C = (1 + c1 * (rank_c - np.eye(dim_local)) + cmu * H) * C
                C = (C + C.T) / 2

                eigvals_reg, eigvecs_reg = np.linalg.eigh(C)
                eigvals_reg = np.maximum(eigvals_reg, 1e-10)
                C = (eigvecs_reg * eigvals_reg) @ eigvecs_reg.T

                x = x_mean

            return x, f_best

        remaining = budget - calls[0]
        cma_budget_1 = int(remaining * 0.7)
        if cma_budget_1 > 20:
            best_x, best_f = cma_es(best_x, best_f, cma_budget_1, sigma_init=0.5)

        # ---------- Phase 2b: Restart from nearby basin ----------
        if time_ok() and budget - calls[0] > 30:
            basin_center = np.round(best_x).astype(float)
            basin_center = np.clip(basin_center, -5, 5)
            f_center = call_func(basin_center)

            restart_candidates = [basin_center]
            for axis in range(dim):
                for delta in [-1, 1]:
                    cand = basin_center.copy()
                    cand[axis] += delta
                    cand = np.clip(cand, -5, 5)
                    restart_candidates.append(cand)

            best_restart_x = basin_center
            best_restart_f = f_center
            for cand in restart_candidates:
                if not time_ok():
                    break
                fc = call_func(cand)
                if fc < best_restart_f:
                    best_restart_f = fc
                    best_restart_x = cand.copy()

            if best_restart_f < best_f:
                best_f = best_restart_f
                best_x = best_restart_x.copy()

            cma_budget_2 = int((budget - calls[0]) * 0.25)
            if cma_budget_2 > 15:
                x2, f2 = cma_es(best_x, best_f, cma_budget_2, sigma_init=0.3)
                if f2 < best_f:
                    best_f = f2
                    best_x = x2.copy()

        # ---------- Phase 3: (1+1)-ES fine-tuning ----------
        if time_ok():
            sigma = 0.1
            for _ in range(budget - calls[0]):
                if not time_ok():
                    break
                step = sigma * rng.standard_normal(dim)
                y = np.clip(best_x + step, -5, 5)
                fy = call_func(y)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    sigma *= 1.05
                else:
                    sigma *= 0.92
                if sigma < 0.001:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
