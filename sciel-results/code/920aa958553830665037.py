# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-start CMA-ES for the anisotropic Discus (f11) function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.monotonic() + 300.0  # safety margin vs 330s kill

        state = {"count": 0, "best_x": None, "best_f": np.inf}

        def evaluate(x):
            if state["count"] >= budget:
                return None
            x = np.asarray(x, dtype=float).reshape(-1)
            if x.shape != (dim,):
                return None
            x = np.clip(x, -5.0, 5.0)
            state["count"] += 1
            try:
                f = float(func(x))
            except Exception:
                return None
            if not np.isfinite(f):
                return None
            if f < state["best_f"]:
                state["best_f"] = f
                state["best_x"] = x.copy()
            return f

        # Multi-start: a few short CMA-ES runs from distinct seeds.
        n_starts = 4
        shares = [0.5, 0.2, 0.15, 0.15]
        seeds = [11, 23, 37, 53]

        for s in range(n_starts):
            if state["count"] >= budget:
                break
            if time.monotonic() > deadline - 5.0:
                break
            rng = np.random.default_rng(seeds[s])
            x0 = rng.uniform(-4.0, 4.0, dim)
            self._cmaes(evaluate, x0, sigma0=1.5,
                        max_iter=max(1, int(budget * shares[s])),
                        rng=rng, deadline=deadline, state=state)

        return state["best_x"], state["best_f"]

    def _cmaes(self, evaluate, x0, sigma0, max_iter, rng, deadline, state):
        dim = self.dim
        budget = self.budget
        x = np.array(x0, dtype=float)
        sigma = float(sigma0)
        C = np.eye(dim)
        mu = max(2, dim // 2)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w * w)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0.0, (mueff - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * ((mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff)))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        lam = 2 * mu

        for _ in range(max_iter):
            if state["count"] >= budget:
                break
            if time.monotonic() > deadline - 2.0:
                break
            B, D = self._eigh(C)
            points = []
            for _ in range(lam):
                z = rng.standard_normal(dim)
                points.append(x + sigma * (B @ (D * z)))
            fvals = []
            for p in points:
                f = evaluate(p)
                fvals.append(f if f is not None else np.inf)
            fvals = np.array(fvals)
            order = np.argsort(fvals)
            new_mean = np.sum(w[None, :mu] * np.array(points)[order[:mu]], axis=1)
            old_sigma = sigma
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) / sigma * (new_mean - x)
            cs = np.linalg.norm(pc) / chi_n
            sigma = sigma * np.exp(c_sigma / d_sigma * (cs / chi_n - 1))
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (new_mean - x) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (state["count"] + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
            if hsig:
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * (new_mean - x) / sigma
            diff = [np.array(points)[order[i]] - new_mean for i in range(mu)]
            rank1 = np.outer(ps, ps)
            rankmu = np.zeros((dim, dim))
            for i in range(mu):
                rankmu += w[i] * np.outer(diff[i], diff[i])
            C = (1 + c_1 * (2 - c_1) - c_1 - c_mu) * C + c_1 * rank1 + c_mu * rankmu
            C = 0.5 * (C + C.T)
            if np.isfinite(fvals[order[0]]):
                x = new_mean
            sigma = float(np.clip(sigma, 1e-9, 20.0))

    def _eigh(self, C):
        try:
            evals, evecs = np.linalg.eigh(C)
        except np.linalg.LinAlgError:
            return np.eye(self.dim), np.ones(self.dim)
        evals = np.clip(evals, 1e-12, None)
        return evecs, np.sqrt(evals)
# EVOLVE-BLOCK-END
