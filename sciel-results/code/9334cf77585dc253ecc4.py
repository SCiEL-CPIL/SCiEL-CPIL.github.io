"""Fresh CMA-ES design for BBOB f4 (Büche-Rastrigin), dim 5, budget 500."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES (global, large initial spread) with a wall-clock deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 270.0          # safety margin vs 300 s / 330 s limits
        dim = self.dim
        budget = self.budget
        lam = max(4, int(np.ceil(4 + 3 * np.log(dim))))   # 14 for dim 5

        # Large initial spread: CMA-ES is global, so a wide sigma helps it
        # escape the many Rastrigin local minima and find the shifted optimum.
        x = np.zeros(dim)
        sigma = 4.0
        sigma_min = 1e-4
        sigma_max = 10.0
        c1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 1)**2 + 2)
        cmu = min(1 - c1,
                  (2 + (2 / (dim + 1) - 1)) / ((dim + 2)**2 + 2))
        cs = (dim + 1.7 - 1) / (dim + 9 + 1.7)
        ds = 1 + 2 * max(0.0, np.sqrt((lam - 1) / (dim + 1)) - 1) + cs
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        mu_eff = 2.0 / np.sum(1.0 / np.arange(1, lam + 1))
        H = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        def evaluate(z):
            return float(func(np.asarray(z, dtype=float)))

        # --- Initial population: wide spread around the origin ---
        pop = x + sigma * np.random.randn(lam, dim)
        pop = np.clip(pop, -5.0, 5.0)
        vals = np.array([evaluate(row) for row in pop])
        order = np.argsort(vals, kind="stable")
        best_x = pop[order[0]].copy()
        best_f = vals[order[0]]

        # --- Main CMA-ES loop ---
        while True:
            if time.time() > deadline:
                break
            # mean update (weighted average of selected points)
            mean = np.average(pop[order[:lam]], axis=0,
                              weights=1.0 / np.arange(1, lam + 1))
            # step-size control
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) \
                * (H @ (pop[order[:lam]] - x).T).T[:, 0] / sigma
            if np.linalg.norm(ps) / chi < (1.4 + 1 / cs) * chi:
                sigma *= np.exp((cs / ds) * (np.linalg.norm(ps) / chi - 1))
            sigma = min(max(sigma, sigma_min), sigma_max)
            # covariance rank updates
            if np.linalg.norm(ps) / chi < (1.4 + 1 / cs) * chi:
                pc = (1 - cs) * pc + cs * (mean - x) / sigma
            diff = (pop[order[:lam]] - x) / sigma
            delta_H = (c1 * np.outer(pc, pc) +
                       (2 * (cmu - c1)) *
                       np.sum(np.outer(diff[:lam - 2], diff[:lam - 2]), axis=0)
                       / (lam - 2))
            H = (1 + c1 * (2 - c1 - cmu)) * H + delta_H
            # rescale to keep determinant 1
            sign, logdet = np.linalg.slogdet(H)
            if sign > 0 and np.isfinite(logdet):
                H = H * (1.0 / np.exp(logdet / (2 * dim)))
            x = mean
            # new population
            pop = x + sigma * np.random.randn(lam, dim)
            pop = np.clip(pop, -5.0, 5.0)
            vals = np.array([evaluate(row) for row in pop])
            order = np.argsort(vals, kind="stable")
            if vals[order[0]] < best_f:
                best_f = vals[order[0]]
                best_x = pop[order[0]].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
