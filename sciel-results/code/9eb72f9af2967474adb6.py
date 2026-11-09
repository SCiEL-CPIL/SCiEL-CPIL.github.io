"""Rewritten optimizer using CMA-ES for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based continuous optimizer for the BBOB f4 benchmark."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= self.budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = float(func(x))
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Initial exploration with random sampling
        n_init = min(20, self.budget // 25)
        candidates = []
        for _ in range(n_init):
            if time.time() > deadline or calls_made >= self.budget:
                break
            x = rng.uniform(-4.5, 4.5, self.dim)
            f = safe_call(x)
            candidates.append((f, x.copy()))

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])

        # Phase 2: CMA-ES from the best few starting points
        remaining = self.budget - calls_made
        if remaining > 10:
            n_starts = min(3, len(candidates), max(1, remaining // 80))
            budget_per_cma = max(20, remaining // max(1, n_starts))

            for i in range(n_starts):
                if time.time() > deadline or calls_made >= self.budget:
                    break

                f0, x0 = candidates[i]
                self._run_cma_es(safe_call, x0, f0, budget_per_cma, deadline, rng,
                                  best_x_holder := {'x': best_x, 'f': best_f})

                # Update local best from holder
                if best_x_holder['f'] < best_f:
                    best_f = best_x_holder['f']
                    best_x = best_x_holder['x']

        # Phase 3: Final local polish with adaptive random steps
        if best_x is not None and calls_made < self.budget:
            remaining = self.budget - calls_made
            step_sizes = [0.3, 0.1, 0.05, 0.02, 0.01, 0.005]
            for step_size in step_sizes:
                if time.time() > deadline or calls_made >= self.budget:
                    break
                n_steps = min(remaining, 15)
                for _ in range(n_steps):
                    if time.time() > deadline or calls_made >= self.budget:
                        break
                    step = step_size * rng.standard_normal(self.dim)
                    x_try = np.clip(best_x + step, -5.0, 5.0)
                    f_try = safe_call(x_try)

        if best_x is None:
            best_x = np.zeros(self.dim)
            safe_call(best_x)

        return best_x, best_f

    def _run_cma_es(self, safe_call, x0, f0, budget, deadline, rng, best_holder):
        """Run CMA-ES starting from x0 with given budget."""
        dim = self.dim
        calls_used = 0

        # CMA-ES parameters
        sigma = 1.0
        pop_size = max(4, int(4 + 3 * np.log(dim)))
        c1 = (2 + 2 / (dim + 1)) / ((dim + 1.3) ** 2 + 2)
        cmu = min(0.1, c1 * 0.5)
        cs = (dim + 1.7) / (dim + 9.2)
        damps = 1 + 2 * max(0, np.sqrt((dim + 1) / (dim + 7) - 1)) + cs
        c_sigma = cs / damps
        c_c = (4 + dim / dim) / (dim + 4)

        mean = x0.copy()
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        def compute_c_sqrt_inv(C):
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            return eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T

        C_sqrt_inv = compute_c_sqrt_inv(C)

        for gen in range(budget // pop_size + 1):
            if time.time() > deadline or calls_used >= budget:
                break

            # Generate population
            offspring = []
            for i in range(pop_size):
                if time.time() > deadline or calls_used >= budget:
                    break
                z = rng.standard_normal(dim)
                x = mean + sigma * C_sqrt_inv @ z
                f = safe_call(x)
                calls_used += 1
                offspring.append((f, x.copy()))

            if len(offspring) < 2:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[0])

            # Update best holder
            if offspring[0][0] < best_holder['f']:
                best_holder['f'] = offspring[0][0]
                best_holder['x'] = offspring[0][1].copy()

            # Update evolution paths
            lam = max(2, pop_size // 2)
            mu = lam
            weights = np.log(mu + 1) - np.log(1 + np.arange(mu))
            weights /= weights.sum()
            mu_eff = 1 / np.sum(weights ** 2)

            # Update ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * \
                 C_sqrt_inv @ (weights @ np.array([offspring[i][1] for i in range(mu)]) / sigma)

            # Check hsig
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) / chi_n < 1.4 + 2 / (dim + 1)

            # Update pc
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * \
                 (weights @ np.array([offspring[i][1] for i in range(mu)]))

            # Update covariance matrix
            delta_sigma = (np.linalg.norm(ps) / chi_n - 1) * c_sigma * (2 - c_sigma)
            sigma *= np.exp(0.3 * delta_sigma)

            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * c_c * (2 - c_c) * C) + \
                cmu / mu_eff * np.sum([weights[i] * np.outer(offspring[i][1] - mean,
                                                              offspring[i][1] - mean)
                                       for i in range(mu)], axis=0)

            C = (C + C.T) / 2

            # Update mean
            mean = mean + sigma * (weights @ np.array([offspring[i][1] for i in range(mu)]))

            # Recompute C^{-1/2} periodically
            if gen % 5 == 0:
                try:
                    C_sqrt_inv = compute_c_sqrt_inv(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim) * np.mean(np.diag(C))
                    C_sqrt_inv = np.eye(dim) / np.sqrt(np.mean(np.diag(C)))
# EVOLVE-BLOCK-END
