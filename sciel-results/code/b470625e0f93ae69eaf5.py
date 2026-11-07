import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style f9 (Rosenbrock rotated)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes(self, func, x0, sigma0, budget, rng, deadline):
        dim = self.dim
        lam = 4 + int(3 * np.log(dim))  # ~12 for dim=5
        mu = lam // 2
        weights = np.log(np.arange(mu, 0, -1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        damps = 1 + 2 * max(0, (mu_eff - 2) / (dim + 2) - 1) + cs

        x = x0.copy()
        sigma = sigma0
        C = np.eye(dim)
        B, D = np.linalg.eigh(C)[1], np.sqrt(np.diag(np.linalg.eigh(C)[0]))
        # Properly decompose
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-20)
        B = eigvecs
        D = np.sqrt(eigvals)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        best_x = x.copy()
        best_fx = func(x)
        calls = 1
        no_improve = 0
        threshold = 1e-10

        for gen in range(budget // lam):
            if time.time() > deadline:
                break
            if calls >= budget:
                break

            pop = []
            fvals = []
            for i in range(lam):
                if calls >= budget:
                    break
                z = rng.standard_normal(dim)
                y = x + sigma * (B * D) @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                pop.append(y)
                fvals.append(fy)
                calls += 1

            if len(pop) == 0:
                break

            idx = np.argsort(fvals)
            pop = [pop[i] for i in idx]
            fvals = np.array(fvals)[idx]

            if fvals[0] < best_fx:
                best_fx = fvals[0]
                best_x = pop[0].copy()
                no_improve = 0
            else:
                no_improve += 1

            if no_improve > 15 and sigma < 1e-4:
                break

            mu_use = min(mu, len(pop))
            m = np.sum(weights[:mu_use, None] * np.array(pop[:mu_use]), axis=0)

            # Update step size
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (B.T @ (m - x))
            hsig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 100.0)

            # Update evolution path
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) / sigma * (B.T @ (m - x))

            # Update covariance
            delta_hs = (1 - hsig) * cc * (2 - cc)
            outer = np.outer(pc, pc)
            C = (1 + c1 * delta_hs - c1 - cmu) * C + c1 * outer
            for i in range(mu_use):
                zi = (np.array(pop[i]) - x) / sigma
                C += cmu * weights[i] * np.outer(zi, zi)

            # Ensure symmetry and positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            B = eigvecs
            D = np.sqrt(eigvals)

            x = m

        return best_x, best_fx, calls

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

        rng = np.random.default_rng(42)

        # Phase 1: Coarse global exploration with large initial sigma
        # Try multiple starting points to find a good basin
        candidates = [
            np.zeros(dim),
            np.array([2.0, 2.0, 2.0, 2.0, 2.0]),
            np.array([-2.0, -2.0, -2.0, -2.0, -2.0]),
            np.array([1.0, -1.0, 1.0, -1.0, 1.0]),
            np.array([-1.0, 1.0, -1.0, 1.0, -1.0]),
        ]

        # Evaluate initial candidates
        best_x = None
        best_fx = np.inf
        calls = 0
        for c in candidates:
            if calls >= budget:
                break
            fc = func(c)
            calls += 1
            if fc < best_fx:
                best_fx = fc
                best_x = c.copy()

        # Phase 2: Local CMA-ES refinement from best starting point
        if calls < budget - 10:
            sigma0 = 1.5
            bx, bf, extra_calls = self._cmaes(func, best_x, sigma0, budget - calls, rng, deadline)
            calls += extra_calls
            if bf < best_fx:
                best_fx = bf
                best_x = bx.copy()

        # Phase 3: If we have budget left, try a few more restarts from random points
        remaining = budget - calls
        if remaining > 20 and time.time() < deadline - 10:
            for attempt in range(3):
                if remaining < 15 or time.time() > deadline - 5:
                    break
                # Random restart point
                rp = rng.uniform(-3, 3, size=dim)
                fr = func(rp)
                remaining -= 1
                if fr < best_fx:
                    best_fx = fr
                    best_x = rp.copy()
                    # Refine from this new best
                    if remaining > 15:
                        bx2, bf2, ec = self._cmaes(func, best_x, 1.0, remaining, rng, deadline)
                        remaining -= ec
                        if bf2 < best_fx:
                            best_fx = bf2
                            best_x = bx2.copy()

        return best_x, best_fx
