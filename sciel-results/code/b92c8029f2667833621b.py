"""Optimizer for BBOB f6 (Attractive Sector) using population-based CMA-ES."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Population-based CMA-ES with full covariance adaptation for anisotropic landscapes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 270.0  # Safety margin before 300s kill

        rng = np.random.default_rng(42)

        # CMA-ES parameters
        lam = 6  # population size - small for frequent early updates
        mu = lam // 2  # number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        cc = (4.0 + mueff / dim) / ((dim + 2.0) ** 2 + mueff / 2.0)
        cs = (mueff + 2.0) / (dim + mueff + 5.0)
        c1 = (2.0 / ((dim + 1.8) ** 2 + mueff)) * (2.0 - cc + cs)
        cmu = min(1.0 - c1, (2.0 * (mueff - 2.0 + 1.0 / mueff)) /
                 ((dim + 2.0) ** 2 + mueff))
        chiN = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim ** 2))

        # State
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        calls = 0
        best_x = None
        best_f = np.inf
        t = 0

        def sample():
            """Sample from N(0, C) using Cholesky decomposition."""
            try:
                L = np.linalg.cholesky(C + 1e-14 * np.eye(dim))
                z = rng.standard_normal((lam, dim))
                return z @ L.T  # (lam, dim)
            except np.linalg.LinAlgError:
                # Fallback: eigenvalue decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-14, None)
                z = rng.standard_normal((lam, dim))
                return z @ (eigvecs * np.sqrt(eigvals)).T

        def update_C():
            """Update covariance matrix C and recompute B, D."""
            nonlocal C, B, D
            C = (1.0 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1.0 - hsig) * cc * (2.0 - cc) * C)
            C += cmu * np.sum(weights[:mu][:, None, None] * np.outer(y_w - x_mean, y_w - x_mean), axis=0) / (sigma ** 2 + 1e-300)
            # Symmetrize
            C = 0.5 * (C + C.T)
            # Compute B, D from C via Cholesky
            try:
                L = np.linalg.cholesky(C + 1e-14 * np.eye(dim))
                Q, R = np.linalg.qr(L.T)
                D = np.diag(R).copy()
                B = Q.T
                # Ensure positive diagonal
                signs = np.sign(D)
                signs[signs == 0] = 1.0
                B = B * signs
                D = np.abs(D)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                B = np.eye(dim)
                D = np.ones(dim)

        # Main loop
        while calls < budget and time.time() < deadline:
            t += 1

            # Check if we can fit a full population
            if calls + lam > budget:
                # Use remaining budget as a partial population
                lam_eff = budget - calls
                if lam_eff < 2:
                    break
            else:
                lam_eff = lam

            # Generate population: y_i = x_mean + sigma * B @ diag(D) @ z_i
            z = rng.standard_normal((lam_eff, dim))
            steps = (z * D) @ B.T  # (lam_eff, dim)
            pop = x_mean + sigma * steps
            pop = np.clip(pop, -5.0, 5.0)

            # Evaluate population
            f_vals = np.empty(lam_eff)
            for i in range(lam_eff):
                f_vals[i] = float(func(pop[i]))
                calls += 1
                if f_vals[i] < best_f:
                    best_f = f_vals[i]
                    best_x = pop[i].copy()
                if calls >= budget or time.time() > deadline:
                    break

            # If we didn't complete the population, break
            if calls >= budget or time.time() > deadline:
                break

            # Sort by fitness
            order = np.argsort(f_vals)
            pop_sorted = pop[order]
            f_sorted = f_vals[order]

            # Number of parents
            mu_eff = min(mu, lam_eff)
            if mu_eff < 2:
                break

            w_eff = weights[:mu_eff].copy()
            w_eff = w_eff / w_eff.sum()
            mueff_eff = 1.0 / np.sum(w_eff ** 2)

            # Weighted mean
            y_w = np.sum(w_eff[:, None] * pop_sorted[:mu_eff], axis=0)

            # Update ps
            ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff_eff) * (
                np.sum(w_eff[:, None] * (pop_sorted[:mu_eff] - x_mean), axis=0) / (sigma + 1e-300)
            )

            # hsig condition
            norm_ps = np.linalg.norm(ps)
            hsig = (norm_ps / np.sqrt(1.0 - (1.0 - cs) ** (2 * t))) < (1.4 + 2.0 / (dim + 1)) * chiN

            # Update pc
            pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc)) * (y_w - x_mean) / (sigma + 1e-300)

            # Update C
            C = (1.0 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1.0 - hsig) * cc * (2.0 - cc) * C)
            C += cmu * np.sum(w_eff[:, None, None] * np.outer(pop_sorted[:mu_eff] - x_mean, pop_sorted[:mu_eff] - x_mean), axis=0) / (sigma ** 2 + 1e-300)
            C = 0.5 * (C + C.T)

            # Recompute B, D
            try:
                L = np.linalg.cholesky(C + 1e-14 * np.eye(dim))
                Q, R = np.linalg.qr(L.T)
                D_new = np.diag(R).copy()
                B_new = Q.T
                signs = np.sign(D_new)
                signs[signs == 0] = 1.0
                B_new = B_new * signs
                D_new = np.abs(D_new)
                B = B_new
                D = D_new
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                B = np.eye(dim)
                D = np.ones(dim)

            # Update sigma
            sigma *= np.exp(cs / chiN * (norm_ps / chiN - 1.0))
            if not np.isfinite(sigma) or sigma <= 0:
                sigma = 1.0
            if sigma < 1e-12:
                break

            # Update mean
            x_mean = y_w

        # Ensure valid return
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = float(func(best_x))

        return best_x, best_f
# EVOLVE-BLOCK-END
