"""Rewritten optimizer for BBOB f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multi-start and polish for BBOB f14 (Different powers)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 285
        rng = np.random.default_rng(42)
        d = self.dim
        max_calls = self.budget

        best_x = None
        best_f = float('inf')
        calls_used = 0

        def try_eval(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= max_calls:
                return None
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Multi-start evaluation
        starts = [np.zeros(d)]
        for i in range(d):
            for val in [-4, 4]:
                pt = np.zeros(d)
                pt[i] = val
                starts.append(pt)
        for _ in range(5):
            starts.append(rng.uniform(-4, 4, d))

        for s in starts:
            fx = try_eval(np.clip(s, -5, 5))
            if fx is None:
                break

        if calls_used >= max_calls or time.time() > deadline - 15:
            return best_x, best_f

        # Phase 2: CMA-ES
        x = best_x.copy()
        sigma = 2.0
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        ccs = (1 + 3 / (d + 1)) / (2 + (d + 3) / (d + 1))
        cmu = 2 / ((d + 1.3) ** 2 + d)
        c1 = (2 + 2 / (d + 1.3)) / ((d + 2) ** 2 + 2)
        c_sigma = (2 + 2 / (d + 1.3)) / (d + 4)
        chi_d = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))
        lam = max(4, int(np.round(4 + 3 * np.log(d))))  # population size

        while calls_used < max_calls:
            if time.time() > deadline - 10:
                break

            # Generate population
            pop = []
            for _ in range(lam):
                z = rng.standard_normal(d)
                y = x + sigma * (C @ z)
                pop.append((np.clip(y, -5, 5), z))

            # Evaluate
            evals = []
            for px, _ in pop:
                fy = try_eval(px)
                if fy is None:
                    break
                evals.append((fy, px))

            if len(evals) < 2:
                break

            # Sort by fitness
            evals.sort(key=lambda t: t[0])
            mu = max(1, lam // 2)
            w = np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = 1 / w.sum()

            # Update mean
            x_new = sum(w[i] * evals[i][1] for i in range(mu))

            # Update covariance
            diff = [(evals[i][1] - x) / sigma for i in range(mu)]
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - (1 - ccs) ** 2) * C) \
                + cmu * sum(w[i] * np.outer(diff[i], diff[i]) for i in range(mu))

            # Symmetrize
            C = (C + C.T) / 2

            # Update path
            pc = (1 - ccs) * pc + ccs * (x_new - x) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (x_new - x) / sigma

            # Update sigma
            sigma *= np.exp(c_sigma / chi_d * (np.linalg.norm(ps) / chi_d - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            x = x_new

            # Termination: sigma too small or no progress
            if sigma < 1e-10:
                break

            # Check for stagnation
            if calls_used > 100 and best_f > -1e-6:
                # Try a random restart
                if time.time() > deadline - 10:
                    break
                new_x = rng.uniform(-4, 4, d)
                new_fy = try_eval(new_x)
                if new_fy is None:
                    break
                if new_fy < best_f:
                    x = new_x
                    sigma = 1.5
                    C = np.eye(d)
                    pc = np.zeros(d)
                    ps = np.zeros(d)
                else:
                    x = best_x.copy()
                    sigma = 0.5
                    C = np.eye(d)
                    pc = np.zeros(d)
                    ps = np.zeros(d)

        # Phase 3: Polish with (1+1)-ES
        if calls_used < max_calls and time.time() < deadline - 3:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            for _ in range(min(30, max_calls - calls_used)):
                if time.time() > deadline - 2:
                    break
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = try_eval(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 0.9
                else:
                    sigma *= 0.8

        return best_x, best_f
# EVOLVE-BLOCK-END
