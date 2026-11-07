import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, calls, max_gens, sigma_init=0.5, pop_size=8):
        """Efficient CMA-ES for local refinement."""
        d = self.dim
        m = np.array(x0, dtype=float)
        sigma = sigma_init
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        best_x = m.copy()
        best_f = func(m)

        mu = max(2, pop_size // 2)
        w = np.log(mu / 2 + 1) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)

        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))

        rng = np.random.default_rng(123)

        for gen in range(max_gens):
            if calls[0] >= self.budget - 5:
                break
            if time.time() > self._deadline - 15:
                break

            xs = []
            fs = []
            for k in range(pop_size):
                z = rng.standard_normal(d)
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C += 1e-4 * np.eye(d)
                    L = np.linalg.cholesky(C)
                x = m + sigma * (L @ z)
                xs.append(x)
                fs.append(func(x))

            idx = np.argsort(fs)
            xs = [xs[i] for i in idx]
            fs = [fs[i] for i in idx]

            if fs[0] < best_f:
                best_f = fs[0]
                best_x = xs[0].copy()

            if sigma < 1e-6:
                break

            m_new = sum(w[i] * xs[i] for i in range(mu))
            sigma_new = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(2 * d) - 1))

            v = [(xs[i] - m) / sigma for i in range(mu)]
            C_mu = sum(w[i] * np.outer(v[i], v[i]) for i in range(mu))
            pc_new = (1 - c_c) * pc + c_c * ((m_new - m) / sigma)
            C = (1 - (c_1 + c_mu)) * C + c_1 * (1 + (1 - c_sigma) * d_sigma) * np.outer(pc, pc) - c_1 * c_sigma * np.outer(pc, pc) + c_mu * (C_mu - C)
            C = (C + C.T) / 2 + 1e-10 * np.eye(d)

            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * ((m_new - m) / sigma)

            m = m_new
            sigma = sigma_new
            pc = pc_new

        return best_x, best_f

    def _nelder_mead(self, func, x0, calls, max_iters=30, step0=0.3):
        """Nelder-Mead for final polish."""
        d = self.dim
        x = np.array(x0, dtype=float)
        f0 = func(x)
        simplex = [x.copy()]
        fs = [f0]
        for i in range(d):
            xi = x.copy()
            xi[i] += step0
            simplex.append(xi)
            fs.append(func(xi))
        best_x = x.copy()
        best_f = f0

        for _ in range(max_iters):
            if calls[0] >= self.budget:
                break
            order = np.argsort(fs)
            simplex = [simplex[i] for i in order]
            fs = [fs[i] for i in order]
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = simplex[0].copy()

            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + (centroid - simplex[-1])
            fr = func(xr)
            if fr < fs[0]:
                xe = centroid + 2.0 * (centroid - simplex[-1])
                fe = func(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fs[-1] = fe
                else:
                    simplex[-1] = xr
                    fs[-1] = fr
                continue
            if fr < fs[-2]:
                simplex[-1] = xr
                fs[-1] = fr
                continue

            # Contraction
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = func(xc)
            if fc < fs[-1]:
                simplex[-1] = xc
                fs[-1] = fc
                continue

            # Shrink
            for i in range(1, d + 1):
                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                fs[i] = func(simplex[i])

        return best_x, best_f

    def __call__(self, func):
        t0 = time.time()
        self._deadline = t0 + 280.0
        dim = self.dim
        calls = [0]
        budget = self.budget

        def safe_func(x):
            if calls[0] >= budget:
                return np.inf
            calls[0] += 1
            try:
                return float(func(np.asarray(x, dtype=float)))
            except Exception:
                return np.inf

        best_x = np.zeros(dim)
        best_f = np.inf
        top_k = []  # list of (f, x) tuples

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = np.array(x, dtype=float)
            top_k.append((f, np.array(x, dtype=float)))
            top_k.sort(key=lambda t: t[0])
            if len(top_k) > 10:
                top_k.pop()

        rng = np.random.default_rng(42)

        # Phase 1: Lattice points for early Rastrigin-aware baseline
        # Evaluate origin and its immediate neighbors
        priority_points = [np.zeros(dim)]
        for i in range(dim):
            for sign in (-1, 1):
                p = np.zeros(dim)
                p[i] = sign
                priority_points.append(p)

        for p in priority_points:
            if calls[0] >= budget or time.time() > self._deadline - 30:
                break
            f_val = safe_func(p)
            record(p, f_val)

        # Phase 2: Sobol exploration for global coverage
        n_explore = 80
        sampler = qmc.Sobol(d=dim, scramble=True)
        sobol_points = sampler.random(n_explore)
        explore_points = sobol_points * 10.0 - 5.0

        for p in explore_points:
            if calls[0] >= budget or time.time() > self._deadline - 30:
                break
            f_val = safe_func(p)
            record(p, f_val)

        if calls[0] >= budget:
            return best_x, best_f

        # Phase 3: Multi-start CMA-ES from top points
        # Run CMA-ES from the best 2-3 points found
        remaining = budget - calls[0]
        if remaining > 30:
            # Sort top_k by fitness
            top_k.sort(key=lambda t: t[0])
            
            # Run CMA-ES from best point
            if remaining > 20:
                xc, fc = self._cma_es(safe_func, top_k[0][1], calls, max_gens=20, sigma_init=0.5, pop_size=8)
                record(xc, fc)
            
            # Run CMA-ES from 2nd best if different and budget allows
            if len(top_k) > 1 and np.linalg.norm(top_k[0][1] - top_k[1][1]) > 0.5:
                if calls[0] < budget - 20:
                    xc, fc = self._cma_es(safe_func, top_k[1][1], calls, max_gens=15, sigma_init=0.5, pop_size=6)
                    record(xc, fc)

        # Phase 4: Nelder-Mead polish on the best point
        if calls[0] < budget - 10 and time.time() < self._deadline - 10:
            xc, fc = self._nelder_mead(safe_func, best_x, calls, max_iters=30, step0=0.3)
            record(xc, fc)

        return best_x, best_f
