"""Optimizer for BBOB f23 (Katsuura) using CMA-ES with restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer with multiple restarts for Katsuura."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, budget, rng, deadline):
        """Run CMA-ES from x0 with sigma0 for at most budget evaluations."""
        d = self.dim
        n_evals = 0
        x = x0.copy()
        sigma = sigma0
        fx = func(x)
        n_evals += 1

        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(d))  # population size
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)

        # Adaptation rates
        cc = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        cs = (mu_eff + 2) / (d + mu_eff + 5)
        c1 = 2 / ((d + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))

        # State variables
        pc = np.zeros(d)
        ps = np.zeros(d)
        B = np.eye(d)
        D = np.ones(d)
        C = np.eye(d)
        D_inv = 1.0 / D
        D_sqrt = np.sqrt(D)
        D_inv_sqrt = 1.0 / D_sqrt

        # Precompute for efficiency
        chi_n = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))

        best_x = x.copy()
        best_f = fx

        while n_evals < budget:
            if time.time() > deadline:
                break

            # Generate population
            offspring = np.zeros((lambda_, d))
            for i in range(lambda_):
                z = rng.standard_normal(d)
                offspring[i] = x + sigma * (D_inv_sqrt @ (B.T @ z))

            # Clip to box
            offspring = np.clip(offspring, -5, 5)

            # Evaluate
            fitness = np.array([func(offspring[i]) for i in range(lambda_)])
            n_evals += lambda_

            # Sort by fitness
            idx = np.argsort(fitness)
            offspring = offspring[idx]
            fitness = fitness[idx]

            # Update best
            if fitness[0] < best_f:
                best_f = fitness[0]
                best_x = offspring[0].copy()

            # Weighted mean
            m_new = np.sum(weights[:, None] * offspring[:mu], axis=0)

            # Update evolution paths
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * (D_inv_sqrt @ (B.T @ (D_inv_sqrt * ((m_new - x) / sigma))))
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (C @ ((m_new - x) / sigma))

            # Update mean
            x = m_new

            # Update covariance matrix
            delta_mu = 1 + 2 * max(0, (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**n_evals) - chi_n) / d)
            rank_one = np.outer(ps, ps)
            rank_mu = np.sum(weights[:, None, None] * np.outer(offspring[:mu] - m_new, offspring[:mu] - m_new), axis=0) / sigma**2

            C = (1 + c1 * (delta_mu - 1) - c1 - cmu) * C + c1 * (np.outer(pc, pc) - C) + cmu * rank_mu

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            try:
                B, D = np.linalg.eigh(C)
                D = np.maximum(D, 1e-12)
            except:
                break

            D_inv = 1.0 / D
            D_sqrt = np.sqrt(D)
            D_inv_sqrt = 1.0 / D_sqrt

            # Update sigma
            sigma *= np.exp(cs / (5 * d) * (np.linalg.norm(ps) / chi_n - 1))

            # Clip sigma to reasonable range
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Termination check
            if sigma < 1e-10:
                break

        return best_x, best_f, n_evals

    def __call__(self, func):
        d = self.dim
        B = self.budget
        start_time = time.time()
        deadline = start_time + 280  # Safety margin: 300s limit, leave 20s

        rng = np.random.default_rng(42)

        # Strategy: multiple restarts with different initial points and sigmas
        # Katsuura is multimodal with many local minima
        # Use a combination of:
        # 1. Start near origin (common for BBOB)
        # 2. Random starts
        # 3. Structured starts

        best_x = None
        best_f = np.inf
        total_evals = 0

        # Define restart configurations
        # Each restart gets a portion of the budget
        # Use adaptive allocation: give more budget to promising restarts

        # First, do a quick exploration phase
        # Then exploit the best region

        # Phase 1: Quick exploration with small budgets
        n_explore = 5
        explore_budget = max(20, B // 10)

        candidates = []
        for i in range(n_explore):
            if time.time() > deadline:
                break
            if total_evals >= B:
                break

            if i == 0:
                x0 = np.zeros(d)
                sigma0 = 1.0
            elif i == 1:
                x0 = rng.uniform(-2, 2, d)
                sigma0 = 2.0
            elif i == 2:
                x0 = rng.uniform(-4, 4, d)
                sigma0 = 3.0
            else:
                x0 = rng.uniform(-5, 5, d)
                sigma0 = 2.0

            x_best, f_best, evals_used = self._cma_es(func, x0, sigma0, explore_budget, rng, deadline)
            total_evals += evals_used
            candidates.append((f_best, x_best))

        if not candidates:
            # Fallback: just do a single run
            x0 = np.zeros(d)
            x_best, f_best, _ = self._cma_es(func, x0, 2.0, B, rng, deadline)
            return x_best, f_best

        # Sort by fitness
        candidates.sort(key=lambda t: t[0])
        best_f = candidates[0][0]
        best_x = candidates[0][1].copy()

        # Phase 2: Exploit the best region with remaining budget
        remaining_budget = B - total_evals
        if remaining_budget > 20 and time.time() < deadline:
            # Refine the best candidate
            x_refined, f_refined, evals_used = self._cma_es(
                func, best_x, 0.5, remaining_budget, rng, deadline
            )
            total_evals += evals_used
            if f_refined < best_f:
                best_f = f_refined
                best_x = x_refined.copy()

        # Phase 3: If we still have budget, try a few more restarts from near best
        remaining_budget = B - total_evals
        if remaining_budget > 30 and time.time() < deadline:
            n_extra = min(3, remaining_budget // 30)
            for i in range(n_extra):
                if time.time() > deadline:
                    break
                if total_evals >= B:
                    break
                # Perturb the best solution
                x0 = best_x + 0.5 * rng.standard_normal(d)
                x0 = np.clip(x0, -5, 5)
                budget_i = remaining_budget // (n_extra - i)
                x_best, f_best, evals_used = self._cma_es(func, x0, 0.3, budget_i, rng, deadline)
                total_evals += evals_used
                if f_best < best_f:
                    best_f = f_best
                    best_x = x_best.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
