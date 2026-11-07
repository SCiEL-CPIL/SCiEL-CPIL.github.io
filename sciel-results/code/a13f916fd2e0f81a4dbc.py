import numpy as np
import time


class Optimizer:
    """Hybrid CMA-ES + coordinate refinement + Nelder-Mead polish for Rastrigin-rotated f15."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, calls, max_iters=50, step0=0.5):
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
                # Extension
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

    def _coordinate_refine(self, func, x0, calls, iters=40, step0=2.0):
        """Rastrigin-aware coordinate descent: shrink step on failure,
        which snaps toward the nearest period-2 lattice valley."""
        d = self.dim
        x = np.array(x0, dtype=float)
        fx = func(x)
        step = np.full(d, step0)
        for _ in range(iters):
            if calls[0] >= self.budget:
                break
            improved = False
            for i in range(d):
                for sign in (1, -1):
                    xt = x.copy()
                    xt[i] += sign * step[i]
                    ft = func(xt)
                    if ft < fx:
                        x = xt
                        fx = ft
                        step[i] *= 1.2
                        improved = True
                        break
            if not improved:
                step *= 0.5
                if np.max(step) < 1e-4:
                    break
        return x, fx

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 275.0  # margin under 330s kill
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

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = np.array(x, dtype=float)

        rng = np.random.default_rng(42)

        # ---- Phase 1: CMA-ES for global exploration ----
        # Parameters
        lambda_ = 10  # Population size
        mu = 5        # Number of parents
        # Weights
        w = np.log(mu / 2 + 1) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)

        # Initial state
        m = np.zeros(dim)
        sigma = 1.5
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # Step size control parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))

        # Pre-compute Cholesky (will update)
        L = np.linalg.cholesky(C)

        # CMA-ES Loop
        # Reserve ~100 calls for final polishing
        cma_budget = budget - 100

        for gen in range(50):
            if calls[0] >= cma_budget or time.time() > deadline - 40:
                break

            # Sample population
            xs = []
            fs = []
            for k in range(lambda_):
                z = rng.standard_normal(dim)
                x = m + sigma * (L @ z)
                xs.append(x)
                fs.append(safe_func(x))

            # Sort
            idx = np.argsort(fs)
            xs = [xs[i] for i in idx]
            fs = [fs[i] for i in idx]

            # Update best
            for x, f in zip(xs, fs):
                record(x, f)

            # Update mean
            m_new = sum(w[i] * xs[i] for i in range(mu))

            # Update covariance matrix
            # Delta sigma
            sigma_new = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(2 * dim) - 1))

            # Update C
            # v_k = (x_k - m) / sigma
            v = [(xs[i] - m) / sigma for i in range(mu)]
            # Rank-mu update
            C_mu = sum(w[i] * np.outer(v[i], v[i]) for i in range(mu))
            # Rank-1 update
            pc_new = (1 - c_c) * pc + c_c * ((m_new - m) / sigma)
            C_1 = (1 - (c_1 + c_mu)) * C + c_1 * (1 + (1 - c_sigma) * d_sigma * np.outer(pc, pc) - c_sigma * np.outer(pc, pc)) + c_mu * (C_mu - C)

            # Symmetrize and ensure positive definite
            C = (C_1 + C_1.T) / 2
            C += 1e-10 * np.eye(dim)

            # Update ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * ((m_new - m) / sigma)

            # Update L
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = C + 1e-3 * np.eye(dim)
                L = np.linalg.cholesky(C)

            m = m_new
            sigma = sigma_new
            pc = pc_new

            # Stopping criterion
            if sigma < 1e-6:
                break

        # ---- Phase 2: Coordinate refinement on best points from CMA-ES ----
        # Take the top few candidates from CMA-ES and refine them
        if calls[0] < budget - 50:
            # Generate a few diverse starting points around the current best
            starts = [best_x.copy()]
            for _ in range(3):
                starts.append(best_x + rng.normal(0, 1.0, dim))

            for s in starts:
                if calls[0] >= budget - 30 or time.time() > deadline - 30:
                    break
                xc, fc = self._coordinate_refine(safe_func, s, calls, iters=30, step0=2.0)
                record(xc, fc)

        # ---- Phase 3: Nelder-Mead polish the best point ----
        if calls[0] < budget - 10:
            px, pf = self._nelder_mead(safe_func, best_x, calls, max_iters=50, step0=0.5)
            record(px, pf)

        return best_x, best_f
