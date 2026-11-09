"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for BBOB f18 (Schaffers F7 ill-conditioned).
    Strategy:
    1. Random sampling to find a good initial basin.
    2. CMA-ES with a larger population size for faster covariance adaptation.
    3. No restarts to preserve budget for convergence.
    4. Final Nelder-Mead polish for high-precision refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        calls_used = 0

        # ---- Phase 1: Random Sampling ----
        # Use a small number of random samples to find the best starting point.
        # 30 samples is a good balance for 5D.
        n_samples = 30
        best_x = None
        best_fx = float('inf')

        for _ in range(n_samples):
            if calls_used >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_used += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        if calls_used >= budget:
            return best_fx

        # ---- Phase 2: CMA-ES ----
        # Use a larger population size to improve the quality of the covariance matrix estimation.
        # For dim=5, a population of 10-12 is very effective.
        pop_size = 12
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)

        # Standard CMA-ES parameters
        c_sigma = (pop_size + 2) / (dim + 4)
        c1 = 2 / ((dim + 1.3) ** 2 + pop_size)
        cmu = min(0.5, c1 * (pop_size // 2 - 1) / 2)
        d_sigma = 1 + 2 * max(0, (dim / pop_size - 1) / 3) + cmu
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        m = best_x.copy()
        # Start with a moderate sigma. Since we sampled [-5,5], a sigma of 1.0 is reasonable.
        # If the function is ill-conditioned, CMA-ES will shrink it quickly.
        sigma = 1.0
        C = np.eye(dim)
        C_sqrt = np.eye(dim)

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        while calls_used < budget - 50: # Reserve 50 calls for polish
            if time.time() > deadline - 5.0:
                break

            # Generate population
            offspring = []
            for _ in range(pop_size):
                z = rng.standard_normal(dim)
                y = m + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                offspring.append(y)

            # Evaluate offspring
            evaluated = []
            for y in offspring:
                if calls_used >= budget - 50:
                    break
                fy = func(y)
                calls_used += 1
                evaluated.append((fy, y))

            if len(evaluated) < pop_size:
                break

            evaluated.sort(key=lambda t: t[0])

            # Update best
            if evaluated[0][0] < best_fx:
                best_fx = evaluated[0][0]
                best_x = evaluated[0][1].copy()

            # Update mean
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * (evaluated[i][1] - m) / sigma
            m = m + sigma * y_w
            m = np.clip(m, -5, 5)

            # Update step size
            y_sigma = np.zeros(dim)
            for i in range(mu):
                y_sigma += weights[i] * (evaluated[i][1] - m) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            p_sigma_norm = np.linalg.norm(p_sigma)
            
            # h_sigma calculation for covariance update
            # The condition for h_sigma is typically based on the norm of p_sigma
            h_sigma = (p_sigma_norm / chi_n < (1.4 + 2 / (dim + 1)) * np.sqrt(1 + (1 - c_sigma) ** (2 * calls_used / (pop_size * (dim + 1)))))
            
            sigma *= np.exp(c_sigma / d_sigma * (p_sigma_norm / chi_n - 1))

            # Update covariance
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * (evaluated[i][1] - m) / sigma
            p_c = (1 - c1) * p_c + h_sigma * np.sqrt(c1 * (2 - c1) * mu_eff) * y_w

            C = (1 + c1 * (h_sigma * (2 - c1) - 1) + cmu) * C
            for i in range(mu):
                y_i = (evaluated[i][1] - m) / sigma
                C -= cmu * weights[i] * (np.outer(y_i, y_i) - C)

            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            try:
                C_sqrt = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                C_sqrt = np.eye(dim)

        # ---- Phase 3: Nelder-Mead Polish ----
        # Use the remaining budget for a local simplex search to refine the solution.
        if calls_used < budget and time.time() < deadline - 2.0:
            x = best_x.copy()
            fx = best_fx
            # Initialize simplex
            simplex = [x.copy()]
            for i in range(dim):
                x_i = x.copy()
                # Use a small step, e.g., 1% of the domain or based on current sigma
                step = max(0.1, sigma * 0.1)
                x_i[i] += step
                simplex.append(np.clip(x_i, -5, 5))
            
            # Evaluate initial simplex
            simplex_f = []
            for sx in simplex:
                if calls_used >= budget:
                    break
                sf = func(sx)
                calls_used += 1
                simplex_f.append(sf)
                if sf < fx:
                    fx = sf
                    x = sx.copy()
                    best_fx = fx
                    best_x = x.copy()

            # Nelder-Mead iterations
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma_nm = 0.5 # shrinkage

            while calls_used < budget and time.time() < deadline - 2.0:
                # Sort simplex
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls_used += 1
                
                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    if calls_used >= budget:
                        break
                    f_e = func(x_e)
                    calls_used += 1
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    if calls_used >= budget:
                        break
                    f_c = func(x_c)
                    calls_used += 1
                    if f_c < simplex_f[-1]:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrinkage
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if calls_used >= budget:
                                break
                            f_s = func(simplex[i])
                            calls_used += 1
                            simplex_f[i] = f_s
                
                # Update global best
                min_f = min(simplex_f)
                if min_f < best_fx:
                    best_fx = min_f
                    best_x = simplex[np.argmin(simplex_f)].copy()

        return best_fx
# EVOLVE-BLOCK-END
