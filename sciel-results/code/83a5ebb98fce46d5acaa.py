"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time guard for sharp ridge optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        rng = np.random.default_rng(7)
        dim = self.dim
        budget = self.budget

        # Initial exploration
        best_x = None
        best_f = np.inf
        n_init = min(8, max(2, budget // 25))
        for _ in range(n_init):
            x0 = rng.uniform(-5, 5, dim)
            fx0 = func(x0)
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        calls = n_init

        # CMA-ES setup
        pop = 4 + int(3 * np.log(dim))
        pop = max(2, min(pop, max(2, budget // 12)))
        mu = pop // 2
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)

        cs = (2 + 2 / (dim + 1)) / (dim + 2 + 2 / (dim + 1))
        cc = (4 + dim / 3) / (dim + 4 + 2)
        ccov = (2 + 2 / (mueff + 1) - (dim + 2) / (mueff + 1)) / (dim ** 2 + 19)
        dcov = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1) - 1) - 1) + ccov
        chin = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        x = best_x.copy()
        sigma = 1.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        stall = 0
        best_run_f = best_f

        while calls < budget:
            if time.time() > deadline:
                break

            # Sample population
            off = []
            offf = []
            for _ in range(pop):
                z = rng.standard_normal(dim)
                y = np.clip(x + sigma * z, -5, 5)
                fy = func(y)
                calls += 1
                off.append(y)
                offf.append(fy)

            if time.time() > deadline:
                break

            # Sort
            idx = np.argsort(offf)
            off = [off[i] for i in idx]
            offf = [offf[i] for i in idx]

            # Update best
            if offf[0] < best_run_f:
                best_run_f = offf[0]
                best_x = off[0].copy()
                stall = 0
            else:
                stall += 1

            # Restart on stall
            if stall > 35:
                x = best_x.copy()
                sigma *= 0.5
                sigma = max(sigma, 0.02)
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                stall = 0
                continue

            # Update mean
            xnew = np.zeros(dim)
            for i in range(mu):
                xnew += w[i] * off[i]

            # Update sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (xnew - x)
            nps = np.linalg.norm(ps)
            if nps > 1e-30:
                sigma *= np.exp(cs / chin * (nps / chin - 1))
            sigma = np.clip(sigma, 1e-10, 30.0)

            # Update covariance
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * (xnew - x) / sigma
            d_h = 1.0 if np.linalg.norm(pc) / chin < (1.4 + 2 / (dim + 1)) else 0.0

            C = (1 + ccov * (2 - ccov + dcov) - ccov) * C
            C += ccov * np.outer(pc, pc)
            C += ccov * (1 - d_h) * cc * (2 - cc) * C
            for i in range(mu):
                C += ccov * w[i] * np.outer(off[i] - x, off[i] - x) / (sigma ** 2)

            # Symmetrize and ensure PSD
            C = (C + C.T) / 2
            ev, evec = np.linalg.eigh(C)
            ev = np.maximum(ev, 1e-14)
            C = evec @ np.diag(ev) @ evec.T

            x = xnew

            if calls >= budget:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
