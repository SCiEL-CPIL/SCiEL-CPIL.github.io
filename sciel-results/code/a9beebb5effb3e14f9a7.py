"""Combined optimizer: CMA-ES with covariance adaptation and Nelder-Mead polishing."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with proper covariance adaptation, followed by Nelder-Mead polishing."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline - 3.0

        rng = np.random.default_rng(12345)

        # CMA-ES parameters
        pop_size = 4 + int(3 * np.log(dim))  # ~11 for dim=5
        pop_size = max(pop_size, 2)
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (1 + 2 / max(dim - 1, 1) + pop_size / (2 * dim)) / (dim + 4)
        c_c = (4 + pop_size / dim) / (dim + 4 + 2 * pop_size / dim)
        c_1 = (2 / ((dim + 1.3) ** 2 + mu_eff)) / (dim + 4)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        best_x = None
        best_fx = np.inf
        call_count = 0

        def safe_call(x):
            nonlocal call_count, best_x, best_fx
            if call_count >= budget or not time_ok():
                return None
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            call_count += 1
            if val < best_fx:
                best_fx = val
                best_x = x_clipped.copy()
            return val

        # Phase 1: CMA-ES
        x_mean = rng.uniform(-4, 4, dim)
        sigma = 2.0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Initial evaluation
        val = safe_call(x_mean)
        if val is None:
            return best_x, best_fx

        while call_count < budget and time_ok():
            # Generate population
            offspring = []
            for i in range(pop_size):
                if not time_ok() or call_count >= budget:
                    break
                z = rng.standard_normal(dim)
                # Compute C_inv_sqrt via Cholesky
                try:
                    L = np.linalg.cholesky(C)
                    L_inv = np.linalg.solve(L, np.eye(dim))
                    C_inv_sqrt = L_inv.T
                except np.linalg.LinAlgError:
                    C_inv_sqrt = np.eye(dim)
                
                y = x_mean + sigma * (C_inv_sqrt @ z)
                y = np.clip(y, -5, 5)
                fy = safe_call(y)
                if fy is None:
                    break
                offspring.append((fy, y))

            if len(offspring) < mu:
                break

            offspring.sort(key=lambda t: t[0])

            # Check convergence
            if offspring[0][0] - offspring[-1][0] < 1e-15:
                break

            # Save old mean
            old_mean = x_mean.copy()

            # Update mean
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * offspring[i][1]
            x_mean = new_mean

            # Normalize differences
            z_vec = np.zeros((dim, mu))
            for i in range(mu):
                z_vec[:, i] = (offspring[i][1] - old_mean) / sigma

            # Update p_sigma
            c_sqrt_sigma = np.sqrt(c_sigma * (2 - c_sigma) * mu_eff)
            h_sigma_threshold = (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / max(dim - 1, 1))
            # Approximate h_sigma
            norm_z = np.linalg.norm(z_vec[:, 0])
            h_sigma = 1.0 if norm_z < h_sigma_threshold else 0.0
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * c_sqrt_sigma * z_vec[:, 0]

            # Update p_c
            c_sqrt_c = np.sqrt(c_c * (2 - c_c) * mu_eff)
            h_c = h_sigma  # Approximate
            p_c = (1 - c_c) * p_c + h_c * c_sqrt_c * z_vec[:, 0]

            # Update covariance
            H = np.zeros((dim, dim))
            for i in range(mu):
                H += weights[i] * np.outer(z_vec[:, i], z_vec[:, i])
            
            C_new = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_c) * c_c * (2 - c_c) * C) + c_mu * H
            C = 0.5 * (C_new + C_new.T)

            # Update sigma
            sigma = sigma * np.exp(c_sigma / dim * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-10, 100.0)

            # Keep mean in bounds
            x_mean = np.clip(x_mean, -5, 5)

        # Phase 2: Nelder-Mead polishing
        nm_budget = min(50, budget - call_count)
        if nm_budget > 10 and best_x is not None and time_ok():
            # Initialize simplex
            simplex = [best_x.copy()]
            for i in range(dim):
                point = best_x.copy()
                step = 0.1 if abs(best_x[i]) > 1e-10 else 0.1
                point[i] += step
                point = np.clip(point, -5, 5)
                simplex.append(point)
            
            # Evaluate initial simplex
            fvals = []
            for pt in simplex:
                val = safe_call(pt)
                if val is None:
                    break
                fvals.append(val)
            
            if len(fvals) == dim + 1:
                # Nelder-Mead iterations
                alpha, gamma, rho = 1.0, 2.0, 0.5
                max_nm_iter = nm_budget // (dim + 1)
                
                for _ in range(max_nm_iter):
                    if not time_ok() or call_count >= budget:
                        break
                    
                    # Sort by function value
                    order = np.argsort(fvals)
                    simplex = [simplex[i] for i in order]
                    fvals = [fvals[i] for i in order]
                    
                    # Centroid of all but worst
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflection
                    x_r = centroid + alpha * (centroid - simplex[-1])
                    x_r = np.clip(x_r, -5, 5)
                    f_r = safe_call(x_r)
                    if f_r is None:
                        break
                    
                    if fvals[0] <= f_r < fvals[-2]:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                    elif f_r < fvals[0]:
                        # Expansion
                        x_e = centroid + gamma * (x_r - centroid)
                        x_e = np.clip(x_e, -5, 5)
                        f_e = safe_call(x_e)
                        if f_e is None:
                            break
                        if f_e < f_r:
                            simplex[-1] = x_e
                            fvals[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            fvals[-1] = f_r
                    else:
                        # Contraction
                        if f_r < fvals[-1]:
                            # Outside contraction
                            x_c = centroid + rho * (x_r - centroid)
                        else:
                            # Inside contraction
                            x_c = centroid - rho * (x_r - centroid)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = safe_call(x_c)
                        if f_c is None:
                            break
                        if f_c < max(f_r, fvals[-1]):
                            simplex[-1] = x_c
                            fvals[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                val = safe_call(simplex[i])
                                if val is None:
                                    break
                                fvals[i] = val
                            if not time_ok() or call_count >= budget:
                                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
