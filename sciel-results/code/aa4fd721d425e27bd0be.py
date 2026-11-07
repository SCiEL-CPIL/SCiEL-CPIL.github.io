"""Optimizer for BBOB f15 (Rastrigin rotated) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Structure-aware search: Rastrigin lattice init + CMA-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin under 300s
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        # Rastrigin: f(x) = 10*d + sum(x_i^2 - 10*cos(2*pi*x_i))
        # Global minima at x_i in 2*Z (even integers). Within [-5,5]: {-4,-2,0,2,4}
        # The hidden shift+rotation moves the minimizer; evaluate the lattice to
        # establish a strong low baseline immediately (early progress counts).
        lattice = [-4.0, -2.0, 0.0, 2.0, 4.0]
        candidates = []
        for c0 in lattice:
            for c1 in lattice:
                for c2 in lattice:
                    for c3 in lattice:
                        for c4 in lattice:
                            x = np.array([c0, c1, c2, c3, c4], dtype=float)
                            candidates.append((func(x), x))
        candidates.sort(key=lambda t: t[0])

        best_x = candidates[0][1].copy()
        best_f = candidates[0][0]
        calls = len(candidates)  # 3125

        # Keep a pool of the best distinct lattice points as seeds for local search.
        seeds = [c[1] for c in candidates[:24]]

        # CMA-ES style local refinement around promising seeds, with adaptive
        # step sizes and a population to handle the hidden rotation/shift.
        def cmaes(center, f0, max_calls, sigma0):
            nonlocal best_x, best_f, calls
            pop = 4 + 2 * int(np.log(dim))  # 4 + 2*2 = 8
            mu = pop // 2
            w = np.full(pop, 1.0 / (2 * mu))
            w[:mu] *= 1.0 + 0.5 * np.log(mu / (np.arange(mu) + 1))
            w /= w.sum()
            cmu = w.sum()
            sigma = sigma0
            C = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            cs = (1 + 2 * np.sqrt((dim - 1) / (dim + 3))) / (dim + 2)
            ds = cs + 2 * np.max((0, np.sqrt((cs + 1) / (cs - 1) - 1)) - 1)
            cc = (4 + dim / (dim + 4)) / ((dim + 2) + 4 + dim / (dim + 4))
            chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            mem = []
            for _ in range(max_calls):
                if calls >= budget or time.time() > deadline - 5:
                    break
                # sample population
                ys = []
                fs = []
                for _ in range(pop):
                    z = rng.standard_normal(dim)
                    # sample from N(0, C)
                    y = center + sigma * (C @ z)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls += 1
                    ys.append(y)
                    fs.append(fy)
                order = np.argsort(fs)
                # update best
                if fs[order[0]] < best_f:
                    best_f = fs[order[0]]
                    best_x = ys[order[0]].copy()
                # recombination
                yw = np.zeros(dim)
                for i in range(mu):
                    yw += w[i] * ys[order[i]]
                mem.append(yw)
                if len(mem) > 2 * dim:
                    mem.pop(0)
                # step-size adaptation (1/5 rule on the selected steps)
                steps = np.array([np.linalg.norm(ys[order[i]] - center) for i in range(mu)])
                avg_step = steps.mean()
                if avg_step > 0:
                    sigma *= np.exp((1 / 5 - 1) * (avg_step / (sigma * chi) - 1) * 0.33)
                    sigma = min(max(sigma, 1e-6), 5.0)
                # covariance adaptation
                hs = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (len(mem) + 1)))) < (1.4 + 2 / (dim + 1)) * chi
                ps = (1 - cs) * ps + cs * ((1 / sigma) * (yw - center))
                pc = (1 - cc) * pc + cc * hs * ((yw - center) / sigma)
                rank_one = np.outer(ps, ps)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    di = (ys[order[i]] - center) / sigma
                    rank_mu += w[i] * np.outer(di, di)
                C = (1 - cs * ds * (2 - cc) - cmu) * C + cs * ds * (pc @ pc + (1 - cc) * cc) + cmu * rank_mu
                # recenter
                center = np.clip(yw, -5, 5)
            return center

        # Refine the top seeds with local CMA-ES, distributing remaining budget.
        remaining = budget - calls
        if remaining > 20:
            per_seed = max(20, remaining // len(seeds))
            for s in seeds:
                if calls >= budget or time.time() > deadline - 5:
                    break
                # sigma based on how promising the seed is relative to best
                sigma0 = 0.6 if s[0] < best_f + 5 else 0.3
                cmaes(s.copy(), func(s), per_seed, sigma0)

        # Final polish: a few (1+1) steps from the best point with a small step.
        if calls < budget and time.time() < deadline - 5:
            x = best_x.copy()
            fx = best_f
            sigma = 0.15
            while calls < budget and time.time() < deadline - 5:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.7
            if fx < best_f:
                best_f, best_x = fx, x

        return best_x, best_f
# EVOLVE-BLOCK-END
