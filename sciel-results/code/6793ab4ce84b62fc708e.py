"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: CMA-ES exploration + Nelder-Mead exploitation for 5D BBOB functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def safe_func(x):
            nonlocal calls
            if calls >= budget:
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            if len(x) != dim:
                x = np.resize(x, dim)
            f = float(func(x))
            calls += 1
            return f

        def nelder_mead(x0, max_iter=50):
            """Nelder-Mead local refinement."""
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return x0, best_f

            # Initialize simplex
            simplex = [x0.copy()]
            f0 = safe_func(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            for i in range(dim):
                xi = x0.copy()
                xi[i] += 0.5
                simplex.append(xi)

            # Evaluate initial simplex
            fvals = [f0]
            for i in range(1, dim + 1):
                fvals.append(safe_func(simplex[i]))
                if fvals[-1] < best_f:
                    best_f = fvals[-1]
                    best_x = simplex[i].copy()

            alpha, gamma, rho = 1.0, 2.0, -0.5

            for it in range(max_iter):
                if calls >= budget or time.time() > deadline:
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = safe_func(xr)
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = safe_func(xe)
                    if fe < best_f:
                        best_f = fe
                        best_x = xe.copy()
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
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid + rho * (simplex[-1] - centroid)
                    fc = safe_func(xc)
                    if fc < best_f:
                        best_f = fc
                        best_x = xc.copy()
                    if fc < fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = safe_func(simplex[i])
                            if fvals[i] < best_f:
                                best_f = fvals[i]
                                best_x = simplex[i].copy()

                if calls >= budget or time.time() > deadline:
                    break

            return best_x, best_f

        # Phase 1: CMA-ES with small population for exploration
        lambda_ = 12  # Small population for speed
        mu = 6  # Top half

        # Recombination weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Step size adaptation
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        # Covariance matrix adaptation
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initialize CMA-ES
        x_mean = np.zeros(dim)
        sigma = 2.5
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)

        # Run CMA-ES for first ~200 calls
        cma_calls = 0
        max_cma_calls = 200

        while cma_calls < max_cma_calls:
            if calls >= budget or time.time() > deadline:
                break

            # Generate population
            pop = np.zeros((lambda_, dim))
            fvals = np.zeros(lambda_)

            for i in range(lambda_):
                if calls >= budget or time.time() > deadline:
                    break
                z = np.random.standard_normal(dim)
                y = B @ (D * z)
                pop[i] = np.clip(x_mean + sigma * y, -5, 5)
                fvals[i] = safe_func(pop[i])
                cma_calls += 1

            if cma_calls >= max_cma_calls or calls >= budget or time.time() > deadline:
                break

            # Sort by fitness
            indices = np.argsort(fvals)
            pop_sorted = pop[indices]
            fvals_sorted = fvals[indices]

            # Weighted recombination
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * pop_sorted[i]

            # Update evolution paths
            y_diff = (x_new - x_mean) / sigma
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_diff
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_diff

            # Update step size
            cs_norm = np.linalg.norm(pc) / chi_n
            hsig = (cs_norm < (1.4 + 1 / dim) * chi_n)
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

            # Update covariance matrix
            if hsig:
                delta_hs = (1 - c_c) / c_c
            else:
                delta_hs = 1
            rank1 = np.outer(pc, pc)
            rankmu = np.zeros((dim, dim))
            for i in range(mu):
                zi = (pop_sorted[i] - x_mean) / sigma
                rankmu += weights[i] * np.outer(zi, zi)
            C = (1 + c_c * (delta_hs - 1) - c_1 - c_mu) * C + c_1 * rank1 + c_mu * rankmu

            # Eigen decomposition
            D2, B2 = np.linalg.eigh(C)
            D2 = np.sqrt(np.maximum(D2, 1e-20))
            B = B2
            D = D2

            x_mean = x_new

        # Phase 2: Nelder-Mead on best point and a few good candidates
        if calls < budget and time.time() < deadline:
            # Refine the best point found
            nelder_mead(best_x.copy(), max_iter=30)

            # Try a few additional random points for multimodal functions
            if calls < budget and time.time() < deadline:
                for _ in range(3):
                    if calls >= budget or time.time() > deadline:
                        break
                    x_rand = np.random.uniform(-4, 4, dim)
                    f_rand = safe_func(x_rand)
                    if f_rand < best_f * 1.5 or f_rand < 0:  # Promising candidate
                        nelder_mead(x_rand.copy(), max_iter=20)

        return best_x, best_f
# EVOLVE-BLOCK-END
