"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with Nelder-Mead polish for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        # Wrap func to count calls and clip
        call_count = 0

        def wrapped_func(x):
            nonlocal call_count
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            val = float(func(x))
            call_count += 1
            return val

        # CMA-ES implementation
        def cma_es(func, x0, sigma0, max_iter, max_time):
            nonlocal call_count
            n = x0.shape[0]
            x0 = x0.copy()
            sigma = sigma0
            popsize = int(4 + 3 * np.log(n))
            lam = popsize

            # Step size control
            cs = 1 / (np.sqrt(n) + 3)
            ds = 1 + 2 * max(0, np.sqrt((lam + 1) / (n + 3)) - 1) + cs
            c_sigma = (ds + 1) / ((n + 1.3) ** 2 + ds)
            c_c = (4 + lam / n) / ((n + 4) ** 2 + lam / n)
            c_1 = 2 / ((n + 1.3) ** 2 + lam / n)
            c_mu = min(1 - c_1, (mu_eff + 2) / ((n + 2) ** 2 + mu_eff))
            mu = lam // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1 / np.sum(weights ** 2)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            C = np.eye(n)
            Cinv = np.eye(n)
            Cinv_sqrt = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            best_x = x0.copy()
            best_f = func(x0)
            call_count = 0

            for it in range(max_iter):
                if time.time() > max_time:
                    break
                if call_count >= budget:
                    break

                # Generate population
                Y = np.zeros((lam, n))
                for i in range(lam):
                    Y[i] = x0 + sigma * Cinv_sqrt @ rng.standard_normal(n)
                Y = np.clip(Y, -5.0, 5.0)

                # Evaluate
                F = np.zeros(lam)
                for i in range(lam):
                    if call_count >= budget:
                        F[i] = best_f
                    else:
                        F[i] = func(Y[i])
                        call_count += 1

                # Sort
                order = np.argsort(F)
                Y = Y[order]
                F = F[order]

                # Update best
                if F[0] < best_f:
                    best_f = F[0]
                    best_x = Y[0].copy()

                # Mean update
                x_m = np.sum(weights[:, None] * Y[:mu], axis=0)
                y = (x_m - x0) / sigma

                # Update p_sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * Cinv_sqrt @ y

                # Update p_c and C
                hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1)))) < (1.4 + 2 / (n + 1)) * chi_n
                p_c = (1 - c_c) * p_c + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y
                rank_mu = np.outer(weights @ Y[:mu] - x0, weights @ Y[:mu] - x0) / sigma ** 2
                rank_one = np.outer(p_c, p_c)
                C = (1 + c_1 * rank_one + c_mu * rank_mu - c_1) * C
                C = 0.5 * (C + C.T)

                # Update Cinv_sqrt
                Cinv_sqrt = self._update_cinv_sqrt(C, Cinv_sqrt)
                Cinv = Cinv_sqrt @ Cinv_sqrt.T

                # Update sigma
                sigma *= np.exp(c_sigma / ds * (np.linalg.norm(p_sigma) / chi_n - 1))

                # Update mean
                x0 = x_m

            return best_x, best_f, call_count

        def _update_cinv_sqrt(self, C, Cinv_sqrt):
            # Simple eigendecomposition-based update
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            Cinv_sqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
            return Cinv_sqrt

        # Nelder-Mead polish
        def nelder_mead(func, x0, max_iter, max_time, max_calls):
            nonlocal call_count
            n = x0.shape[0]
            # Initialize simplex
            simplex = [x0.copy()]
            for i in range(n):
                pt = x0.copy()
                pt[i] = x0[i] + 0.05 if x0[i] < 4.95 else x0[i] - 0.05
                simplex.append(pt)

            fvals = [func(pt) for pt in simplex]
            call_count += n + 1

            for it in range(max_iter):
                if time.time() > max_time:
                    break
                if call_count >= max_calls:
                    break

                # Sort
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                f_worst = fvals[-1]

                # Reflection
                alpha = 1.0
                x_r = centroid + alpha * (centroid - worst)
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = func(x_r)
                call_count += 1

                if f_r < fvals[0]:
                    # Expansion
                    gamma = 2.0
                    x_e = centroid + gamma * (centroid - worst)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = func(x_e)
                    call_count += 1
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    rho = 0.5
                    x_c = centroid + rho * (worst - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = func(x_c)
                    call_count += 1
                    if f_c < f_worst:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = func(simplex[i])
                            call_count += 1

            best_idx = np.argmin(fvals)
            return simplex[best_idx], fvals[best_idx], call_count

        # Main optimization loop
        best_x = None
        best_f = np.inf
        remaining_budget = budget

        # Phase 1: CMA-ES
        x0 = rng.uniform(-4, 4, d)
        f0 = wrapped_func(x0)
        call_count = 1
        if f0 < best_f:
            best_f = f0
            best_x = x0.copy()

        cma_x, cma_f, cma_calls = self.cma_es(wrapped_func, x0, 2.0, 300, deadline)
        call_count += cma_calls
        if cma_f < best_f:
            best_f = cma_f
            best_x = cma_x.copy()

        # Phase 2: Nelder-Mead polish
        remaining = budget - call_count
        if remaining > d + 10 and time.time() < deadline - 5:
            nm_x, nm_f, nm_calls = self.nelder_mead(wrapped_func, best_x, 200, deadline - 2, remaining)
            call_count += nm_calls
            if nm_f < best_f:
                best_f = nm_f
                best_x = nm_x.copy()

        # Return best found
        return best_x, best_f
# EVOLVE-BLOCK-END
