"""CMA-ES optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES with population, adaptation, and time-safe budgeting."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before 300s kill
        dim = self.dim
        budget = self.budget

        # Population size
        lambda_ = int(4 + 3 * np.log(dim))
        mu = lambda_ // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 / (dim + 1.3) ** 2 + 2 / (dim + mu_eff)) * (1 - c_c)
        c_mu = (2 / (dim + 1.3) ** 2 + 2 / (dim + mu_eff)) * (
            min(1, (mu_eff - 2 + np.prod(w)) / (dim + 2))
        )
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial point and sigma
        rng = np.random.default_rng(0)
        x = np.zeros(dim)
        sigma = 2.0
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(z):
            nonlocal best_x, best_f, calls
            z = clip(z)
            f = func(z)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = z.copy()
            return f

        # Initial population
        while calls < budget:
            # Check time
            if time.time() > deadline - 5.0:
                break

            # Generate population
            pop = []
            for i in range(lambda_):
                if calls >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break
                z = x + sigma * rng.multivariate_normal(np.zeros(dim), C)
                f = evaluate(z)
                pop.append((z, f))
            if len(pop) == 0:
                break

            # Sort by fitness
            pop.sort(key=lambda t: t[1])
            selected = pop[:mu]
            yk = [s[0] for s in selected]

            # Weighted mean
            x_new = np.sum(np.array([w[i] * yk[i] for i in range(mu)]), axis=0)

            # Evolution path for sigma
            y_sigma = np.zeros(dim)
            for i in range(mu):
                y_sigma += w[i] * (yk[i] - x) / (sigma + 1e-30)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (calls + 1))) / chi_n < (1.4 + 2 / (dim + 1))
            sigma *= np.exp((c_sigma / chi_n) * (np.linalg.norm(p_sigma) / chi_n - 1) * (1.0 if hsig else 1.0))

            # Evolution path for C
            p_c = (1 - c_c) * p_c + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / sigma
            x = x_new

            # Covariance update
            delta_h = (1 - hsig) * c_c * (2 - c_c)
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(p_c, p_c) + delta_h * np.eye(dim))
            for i in range(mu):
                diff = (yk[i] - x) / sigma
                C += c_mu * w[i] * np.outer(diff, diff)

            # Symmetrize
            C = 0.5 * (C + C.T)
            sigma = float(np.clip(sigma, 1e-12, 100.0))

            if calls >= budget or time.time() > deadline - 5.0:
                break

        # Final polish with Nelder-Mead on the best point found
        if best_x is not None and calls < budget and time.time() < deadline - 2.0:
            best_x = self._nelder_mead_polish(func, best_x, dim, calls, budget, deadline, evaluate)
            if best_x is not None and best_f > np.inf:
                best_f = evaluate(best_x)

        return best_x, best_f

    def _nelder_mead_polish(self, func, x0, dim, calls, budget, deadline, evaluate):
        """Nelder-Mead local refinement with budget and time guards."""
        if calls >= budget or time.time() > deadline - 2.0:
            return None

        # Initialize simplex
        alpha = 1.0
        beta = 0.5
        eps = 1e-8
        simplex = [x0.copy()]
        for i in range(dim):
            pt = x0.copy()
            step = alpha if abs(x0[i]) > eps else alpha
            pt[i] += step
            simplex.append(pt)

        # Evaluate initial simplex
        fvals = []
        for pt in simplex:
            if calls >= budget or time.time() > deadline - 2.0:
                return None
            f = evaluate(pt)
            fvals.append(f)

        # Nelder-Mead iterations
        max_iter = min(200, (budget - calls) // (dim + 1))
        for _ in range(max_iter):
            if calls >= budget or time.time() > deadline - 2.0:
                break

            # Sort by fitness
            idx = np.argsort(fvals)
            simplex = [simplex[i] for i in idx]
            fvals = [fvals[i] for i in idx]

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = evaluate(xr)

            if fr < fvals[0]:
                # Expansion
                xe = centroid + (alpha + 1) * (centroid - simplex[-1])
                fe = evaluate(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    fc = evaluate(xc)
                    if fc <= fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            if calls >= budget or time.time() > deadline - 2.0:
                                break
                            fvals[i] = evaluate(simplex[i])
                        break
                else:
                    xc = centroid - 0.5 * (simplex[-1] - centroid)
                    fc = evaluate(xc)
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            if calls >= budget or time.time() > deadline - 2.0:
                                break
                            fvals[i] = evaluate(simplex[i])
                        break

        # Return best point from simplex
        if fvals:
            best_idx = np.argmin(fvals)
            return simplex[best_idx]
        return None
# EVOLVE-BLOCK-END
