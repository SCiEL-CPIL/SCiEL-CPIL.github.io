"""Improved BBOB f22 optimizer: CMA-ES + (1+1)-ES + Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer for BBOB f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 280.0

        best_x = np.zeros(self.dim)
        best_f = np.inf
        calls_made = 0

        def check_time():
            return (time.time() - start_time) < time_limit

        def safe_func(x):
            nonlocal calls_made, best_x, best_f
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(7)

        # Phase 1: Random exploration to find good basins
        n_random = min(40, max(10, self.budget // 12))
        random_points = []
        for i in range(n_random):
            if not check_time() or calls_made >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = safe_func(x)
            random_points.append((fx, x))

        random_points.sort(key=lambda t: t[0])

        # CMA-ES implementation
        def cma_es(x0, f0, max_evals=150):
            nonlocal best_x, best_f, calls_made

            if max_evals <= 0 or not check_time() or (self.budget - calls_made) <= 0:
                return

            dim = self.dim
            n = dim
            # Population size
            lambda_ = max(4, int(4 + 3 * np.log(n)))
            # Weighting
            mu = lambda_ // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1 / np.sum(weights ** 2)
            # Strategy parameters
            c_sigma = (mueff + 2) / (n + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # State
            mean = x0.copy()
            sigma = 0.5
            B = np.eye(n)
            D = np.ones(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            evals = 0
            no_improve = 0

            # Initial population
            pop = []
            pop_f = []
            for i in range(lambda_):
                if not check_time() or (self.budget - calls_made) <= 0:
                    return
                z = rng.standard_normal(n)
                y = mean + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                fy = safe_func(y)
                pop.append(y)
                pop_f.append(fy)
                evals += 1

            while evals < max_evals and check_time() and (self.budget - calls_made) > 0:
                # Sort by fitness
                idx = np.argsort(pop_f)
                pop = [pop[i] for i in idx]
                pop_f = [pop_f[i] for i in idx]

                # Update best
                if pop_f[0] < best_f:
                    best_f = pop_f[0]
                    best_x = pop[0].copy()

                # Weighted recombination
                y_w = np.zeros(n)
                for i in range(mu):
                    y_w += weights[i] * (pop[i] - mean)

                # Update evolution paths
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * y_w
                hs = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2) / chi_n
                pc = (1 - c_c) * pc + c_c / sigma * y_w

                # Update covariance matrix
                delta_d = 1 if hs < (1.4 + 2 / (n + 1)) * chi_n else 0
                delta_outer = (delta_d - 1) * c_1 * (c_1 + c_mu)
                rank_one = np.outer(pc, pc)
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    zi = (pop[i] - mean) / sigma
                    rank_mu += weights[i] * np.outer(zi, zi)

                C = B @ (D ** 2) @ B.T
                C = (1 + c_1 * delta_outer + c_1 + c_mu - c_1 - c_mu) * C
                C += c_1 * (rank_one - C)
                C += c_mu * rank_mu

                # Ensure positive semi-definite
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update B and D
                eigvals, eigvecs = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(eigvals, 1e-20))
                B = eigvecs

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Update mean
                mean = mean + y_w
                mean = np.clip(mean, -5, 5)

                # Stopping criteria
                if np.linalg.norm(y_w) / (sigma + 1e-10) < 1e-8:
                    no_improve += 1
                else:
                    no_improve = 0
                if no_improve > 10 or sigma < 1e-8:
                    break

                # Generate new population
                new_pop = []
                new_pop_f = []
                for i in range(lambda_):
                    if not check_time() or (self.budget - calls_made) <= 0:
                        return
                    z = rng.standard_normal(n)
                    y = mean + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    fy = safe_func(y)
                    new_pop.append(y)
                    new_pop_f.append(fy)
                    evals += 1
                pop = new_pop
                pop_f = new_pop_f

        # (1+1)-ES with 1/5 success rule
        def one_plus_one_es(x0, f0, max_steps=60):
            nonlocal best_x, best_f, calls_made

            if max_steps <= 0 or not check_time() or (self.budget - calls_made) <= 0:
                return

            dim = self.dim
            sigma = 0.3
            successes = 0
            trials = 0

            for _ in range(max_steps):
                if not check_time() or (self.budget - calls_made) <= 0:
                    break

                y = np.clip(x0 + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_func(y)
                trials += 1

                if fy < f0:
                    x0, f0 = y, fy
                    successes += 1
                    if f0 < best_f:
                        best_f = f0
                        best_x = x0.copy()

                # 1/5 success rule: adjust sigma
                if trials >= 5:
                    if successes / trials > 0.2:
                        sigma *= 1.2
                    else:
                        sigma *= 0.85
                    successes = 0
                    trials = 0

                sigma = np.clip(sigma, 1e-6, 5.0)

                if sigma < 1e-5:
                    break

        # Nelder-Mead local search
        def nelder_mead(x0, f0, max_steps=40):
            nonlocal best_x, best_f, calls_made

            if max_steps <= 0 or not check_time() or (self.budget - calls_made) <= 0:
                return

            dim = self.dim
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_shrink = 0.5

            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += 0.03 * (1 if xi[i] < 0 else -1)
                xi = np.clip(xi, -5, 5)
                simplex.append(xi)

            simplex_f = [f0]
            for i in range(1, dim + 1):
                if not check_time() or (self.budget - calls_made) <= 0:
                    return
                fx = safe_func(simplex[i])
                simplex_f.append(fx)
                if fx < best_f:
                    best_f = fx
                    best_x = simplex[i].copy()

            for step in range(max_steps):
                if not check_time() or (self.budget - calls_made) <= 0:
                    break

                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                if simplex_f[0] < best_f:
                    best_f = simplex_f[0]
                    best_x = simplex[0].copy()

                centroid = np.mean(simplex[:-1], axis=0)

                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fxr = safe_func(xr)
                if not check_time() or (self.budget - calls_made) <= 0:
                    break

                if fxr < simplex_f[0]:
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fxe = safe_func(xe)
                    if not check_time() or (self.budget - calls_made) <= 0:
                        break
                    if fxe < fxr:
                        simplex[-1] = xe
                        simplex_f[-1] = fxe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fxr
                elif fxr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fxr
                else:
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fxc = safe_func(xc)
                    if not check_time() or (self.budget - calls_made) <= 0:
                        break
                    if fxc < simplex_f[-1]:
                        simplex[-1] = xc
                        simplex_f[-1] = fxc
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if not check_time() or (self.budget - calls_made) <= 0:
                                return
                            simplex_f[i] = safe_func(simplex[i])

                if np.max(simplex_f) - np.min(simplex_f) < 1e-12:
                    break

        # Phase 2: CMA-ES from best random point
        if random_points and check_time() and (self.budget - calls_made) > 50:
            x_start = random_points[0][1].copy()
            f_start = random_points[0][0]
            cma_es(x_start, f_start, max_evals=150)

        # Phase 3: (1+1)-ES refinement from current best
        remaining = self.budget - calls_made
        if remaining > 30 and best_x is not None and check_time():
            one_plus_one_es(best_x.copy(), best_f, max_steps=60)

        # Phase 4: Nelder-Mead polish
        remaining = self.budget - calls_made
        if remaining > 15 and best_x is not None and check_time():
            nelder_mead(best_x.copy(), best_f, max_steps=40)

        # Phase 5: Final CMA-ES if budget allows
        remaining = self.budget - calls_made
        if remaining > 80 and best_x is not None and check_time():
            cma_es(best_x.copy(), best_f, max_evals=min(100, remaining - 10))

        # Phase 6: Final (1+1)-ES polish
        remaining = self.budget - calls_made
        if remaining > 20 and best_x is not None and check_time():
            one_plus_one_es(best_x.copy(), best_f, max_steps=min(40, remaining - 5))

        # Phase 7: Final Nelder-Mead
        remaining = self.budget - calls_made
        if remaining > 10 and best_x is not None and check_time():
            nelder_mead(best_x.copy(), best_f, max_steps=min(20, remaining - 5))

        return best_x, best_f
# EVOLVE-BLOCK-END
