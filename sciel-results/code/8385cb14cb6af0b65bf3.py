import numpy as np
import time

class Optimizer:
    """
    Hybrid optimizer for BBOB f17 (Schaffers F7):
    - Phase 1: Sobol quasi-random global exploration
    - Phase 2: CMA-ES with adaptive restarts from best points
    - Phase 3: Nelder-Mead polishing
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill
        n = self.dim
        budget = self.budget

        calls_made = [0]

        def safe_func(x):
            if time.time() > deadline:
                return float('inf')
            calls_made[0] += 1
            try:
                val = func(np.asarray(x, dtype=float))
                if np.isfinite(val):
                    return float(val)
                return float('inf')
            except Exception:
                return float('inf')

        # Phase 1: Global exploration with Sobol
        n_global = min(budget, max(50, int(budget * 0.25)))
        
        try:
            from scipy.stats import qmc
            sobol = qmc.Sobol(d=n, scramble=True)
            points_unit = sobol.random(n_global)
            # Map to [-4.5, 4.5] to cover likely minimum region
            points = points_unit * 9.0 - 4.5
        except Exception:
            points = np.random.uniform(-4.5, 4.5, (n_global, n))

        best_x = None
        best_f = float('inf')
        all_points = []
        all_fvals = []

        for p in points:
            if time.time() > deadline:
                break
            f_val = safe_func(p)
            all_points.append(p)
            all_fvals.append(f_val)
            if f_val < best_f:
                best_f = f_val
                best_x = p.copy()

        # Sort points by fitness
        if len(all_points) > 0:
            sorted_idx = np.argsort(all_fvals)
            all_points = [all_points[i] for i in sorted_idx]
            all_fvals = [all_fvals[i] for i in sorted_idx]

        # Phase 2: CMA-ES from multiple starting points
        remaining = budget - calls_made[0]
        if remaining > 20 and best_x is not None:
            # Select top K starting points for CMA-ES
            n_starts = min(5, max(1, len(all_points) // 10))
            start_points = all_points[:n_starts]
            start_fvals = all_fvals[:n_starts]

            for sp_idx, sp in enumerate(start_points):
                if time.time() > deadline:
                    break
                if calls_made[0] >= budget:
                    break

                # Allocate remaining budget among starts
                budget_left = budget - calls_made[0]
                if budget_left < 10:
                    break

                # Sigma based on distance to best
                if best_x is not None:
                    dist = np.linalg.norm(sp - best_x)
                    sigma0 = max(0.1, min(1.0, dist * 0.5))
                else:
                    sigma0 = 1.0

                # Run CMA-ES
                cma_best_x, cma_best_f, cma_calls = self._cma_es(
                    safe_func, sp, sigma0, budget_left, 
                    np.random.default_rng(sp_idx * 1000 + 42),
                    deadline, best_f
                )

                if cma_best_f < best_f:
                    best_f = cma_best_f
                    best_x = cma_best_x

        # Phase 3: Nelder-Mead polishing
        remaining = budget - calls_made[0]
        if remaining > 5 and best_x is not None:
            try:
                from scipy.optimize import minimize
                max_iter = max(5, remaining // (n + 1))
                res = minimize(
                    safe_func, best_x, method='Nelder-Mead',
                    options={
                        'maxiter': max_iter,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f

    def _cma_es(self, func, x0, sigma0, budget, rng, deadline, incumbent_f):
        """
        Run CMA-ES with given budget.
        Returns (best_x, best_f, calls_used).
        """
        n = self.dim
        x = np.array(x0, dtype=float).copy()
        sigma = sigma0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)

        # CMA-ES parameters
        lambda_ = min(4 + int(3 * np.log(n)), budget // 2)
        lambda_ = max(4, lambda_)
        mu = lambda_ // 2

        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        # Learning rates
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

        # Step size control
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_sigma = (1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1)) * c_c / 2

        # h_sigma threshold
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        h_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1)

        # Initial simplex-like population
        best_x = x.copy()
        best_f = func(x)
        calls_used = 1

        # Population
        pop = []
        pop_f = []

        # Generate initial population
        for i in range(lambda_):
            if time.time() > deadline or calls_used >= budget:
                break
            z = rng.standard_normal(n)
            # Transform using C (initially identity)
            y = z  # C @ z, but C starts as identity
            xi = x + sigma * y
            fi = func(xi)
            calls_used += 1
            pop.append(xi)
            pop_f.append(fi)

            if fi < best_f:
                best_f = fi
                best_x = xi.copy()

        # Sort population
        if len(pop) > 0:
            order = np.argsort(pop_f)
            pop = [pop[i] for i in order]
            pop_f = [pop_f[i] for i in order]

        # Main loop
        while calls_used < budget and time.time() < deadline:
            # Compute mean of best mu
            if len(pop) < mu:
                break

            y_w = np.zeros(n)
            for i in range(mu):
                y_w += w[i] * (pop[i] - x)

            # Update evolution paths
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * y_w
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (y_w / sigma)

            # Update covariance
            delta_hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_used + 1) / (n * c_sigma))) > (1 + 2 / (n + 1)) * chi_n)
            C = ((1 - c_1 - c_mu + c_1 * delta_hsig) * C +
                 c_1 * (np.outer(pc, pc) - delta_hsig * C) +
                 c_mu / sigma ** 2 * np.sum(w[i] * np.outer(pop[i] - x, pop[i] - x) for i in range(mu)))

            # Symmetrize
            C = (C + C.T) / 2

            # Update step size
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

            # Update mean
            x = x + y_w

            # Generate new population
            new_pop = []
            new_pop_f = []
            for i in range(lambda_):
                if time.time() > deadline or calls_used >= budget:
                    break
                z = rng.standard_normal(n)
                # Transform using Cholesky of C
                try:
                    L = np.linalg.cholesky(C)
                    y = L @ z
                except np.linalg.LinAlgError:
                    # Fallback: use eigen decomposition
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-20)
                    y = eigvecs @ (np.sqrt(eigvals) * z)
                xi = x + sigma * y
                fi = func(xi)
                calls_used += 1
                new_pop.append(xi)
                new_pop_f.append(fi)

                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()

            # Replace population
            pop = new_pop
            pop_f = new_pop_f

            # Sort
            if len(pop) > 0:
                order = np.argsort(pop_f)
                pop = [pop[i] for i in order]
                pop_f = [pop_f[i] for i in order]

        return best_x, best_f, calls_used
