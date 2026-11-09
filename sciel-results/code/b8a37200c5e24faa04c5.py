"""Optimizer for the BBOB-style Schwefel (f20) task.

Black-box continuous minimizer on [-5,5]^5 with a 500-evaluation budget.
Uses a CMA-ES (Covariance Matrix Adaptation) search with a population,
which is substantially stronger than the seed (1+1)-ES for the multimodal,
rotated Schwefel landscape.  A wall-clock deadline guard guarantees the
program returns the best result found before the evaluator's limit.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES minimizer with adaptive sigma and a deadline guard."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin well under the 300 s / 330 s limits

        # Per-instance entropy so each of the 3 instances is explored
        # independently (the seed used a fixed seed for all, which was a
        # weakness).  Derive a stable seed from the object identity.
        seed = (id(self) ^ 0x9E3779B97F4A7C15) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)

        # ---------- Initialization: cheap random scan to locate the basin ----------
        # Schwefel has a known global structure; a handful of random starts
        # lets us pick the best as the CMA-ES center.
        n_init = min(16, max(4, budget // 30))
        init = rng.uniform(-5.0, 5.0, size=(n_init, dim))
        init_vals = np.array([func(x) for x in init], dtype=float)
        i0 = int(np.argmin(init_vals))
        x = init[i0].copy()
        f_best = float(init_vals[i0])

        # ---------- CMA-ES parameters ----------
        # Population size, step size, and adaptation constants.
        lam = max(4, dim + int(round(3 * np.log(dim))))
        mu = lam // 2
        weights = np.log(mu + 1.0) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(
            1 - c_1,
            2 * ((mu_eff + 2) / (dim + 2) ** 2 - 1),
        )
        c_mu = max(0.0, c_mu)
        d_sigma = 1 + 2 * max(0.0, (mu_eff - 1) / (dim + 1)) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        sigma = 2.0  # broad initial step for a 5D box
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        invsqrtC = np.eye(dim)
        calls = n_init
        best_x = x.copy()

        # ---------- Main evolution loop ----------
        while calls < budget:
            # Deadline guard: stop early and return best found.
            if time.time() > deadline:
                break

            # Generate population.
            B = rng.standard_normal((lam, dim))
            # Inverse square root of C (eigendecomposition each generation).
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-20, None)
            invsqrtC = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            pop = x[None, :] + sigma * (B @ invsqrtC)
            pop = np.clip(pop, -5.0, 5.0)

            # Evaluate population.
            pop_vals = np.empty(lam, dtype=float)
            for j in range(lam):
                if calls >= budget:
                    pop_vals[j] = f_best
                    break
                if time.time() > deadline:
                    pop_vals[j] = f_best
                    break
                pop_vals[j] = func(pop[j])
                calls += 1
                if pop_vals[j] < f_best:
                    f_best = float(pop_vals[j])
                    best_x = pop[j].copy()

            if calls >= budget:
                break

            # Sort and take the mu best.
            order = np.argsort(pop_vals)
            sel = order[:mu]
            y = pop[sel]
            w = weights

            # New mean (weighted recombination).
            new_x = (w @ y)
            new_x = np.clip(new_x, -5.0, 5.0)

            # Update evolution paths.
            y_w = invsqrtC.T @ ((y - x) / sigma)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (w @ y_w)
            norm_ps = np.linalg.norm(ps)
            hsig = (norm_ps / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - n_init + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (w @ y_w)

            # Covariance matrix update.
            delta_h = (1 - hsig) * c_c * (2 - c_c)
            outer = np.outer(pc, pc)
            delta_outer = delta_h * (C - outer)
            C = (1 + c_1 + c_mu - c_1 - c_mu) * C  # keep base
            C = (1 + c_1 * (2 - c_1) - c_1) * C + c_1 * (outer - C)
            # The standard CMA-ES rank-mu update:
            C = C + c_mu * np.sum([w[i] * np.outer(y_w[i], y_w[i]) for i in range(mu)], axis=0)
            # Re-orthonormalize to keep C positive definite and symmetric.
            C = 0.5 * (C + C.T)
            # Clip eigenvalues to keep the matrix well-conditioned.
            ev, evec = np.linalg.eigh(C)
            ev = np.clip(ev, 1e-12, 1e6)
            C = (evec * ev) @ evec.T

            # Adapt sigma.
            sigma *= np.exp((c_sigma / d_sigma) * (norm_ps / chi_n - 1))
            sigma = np.clip(sigma, 1e-6, 50.0)

            x = new_x

        return best_x, f_best
# EVOLVE-BLOCK-END
