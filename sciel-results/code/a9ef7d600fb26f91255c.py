import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 300s limit with 10s safety margin

        call_count = [0]
        best_x = None
        best_f = float('inf')

        def wrapped_func(x):
            nonlocal best_x, best_f
            if call_count[0] >= self.budget:
                return float('inf')
            x = np.asarray(x, dtype=float)
            if x.shape != (self.dim,):
                x = x.reshape(self.dim)
            fx = float(func(x))
            call_count[0] += 1
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def time_left():
            return deadline - time.time()

        rng = np.random.default_rng(42)

        # Phase 0: Quick random search to find a good starting region
        rs_budget = min(20, self.budget // 5)
        rs_points = rng.uniform(-4, 4, (rs_budget, self.dim))
        rs_fits = np.empty(rs_budget)
        for i in range(rs_budget):
            if time_left() < 5:
                break
            rs_fits[i] = wrapped_func(rs_points[i])
        
        if best_x is None:
            return np.zeros(self.dim), 0.0

        # Use the best random point as CMA-ES initialization center
        finite_mask = np.isfinite(rs_fits)
        if finite_mask.sum() > 0:
            best_rs_idx = np.where(finite_mask)[0][np.argmin(rs_fits[finite_mask])]
            cma_center = rs_points[best_rs_idx].copy()
        else:
            cma_center = np.zeros(self.dim)

        # Phase 1: CMA-ES for global exploration
        N = 12
        pop = np.zeros((N, self.dim))
        for i in range(N):
            pop[i] = cma_center + rng.standard_normal(self.dim) * 1.5
            pop[i] = np.clip(pop[i], -5, 5)
        
        fit = np.empty(N)
        for i in range(N):
            if time_left() < 5:
                break
            fit[i] = wrapped_func(pop[i])
        
        finite_mask = np.isfinite(fit)
        if finite_mask.sum() == 0:
            return best_x, best_f
        pop = pop[finite_mask]
        fit = fit[finite_mask]
        N = len(fit)

        best_idx = np.argmin(fit)
        best_x = pop[best_idx].copy()
        best_f = fit[best_idx]

        # Track top-k points for multi-start polish
        top_k_points = [pop[best_idx].copy()]
        top_k_fits = [fit[best_idx]]

        # CMA-ES parameters
        sigma = 1.0
        mu = N // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
        c_mu = (mu_eff - 2 + 2 / (self.dim + mu_eff)) / ((self.dim + 2)**2 + mu_eff)
        c_mu = min(c_mu, 2 * c_1)

        # State variables
        mean = pop[best_idx].copy()
        C = np.eye(self.dim)
        step_size = 1.0
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        chi_n = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))

        de_budget = int(self.budget * 0.6)
        max_iter_de = max(1, de_budget // N)

        for gen in range(max_iter_de):
            if time_left() < 15:
                break
            if call_count[0] >= self.budget - 30:
                break

            # Precompute Cholesky for this generation
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(self.dim)

            # Sample population
            pop = np.zeros((N, self.dim))
            for i in range(N):
                z = rng.standard_normal(self.dim)
                pop[i] = mean + step_size * (L @ z)
                pop[i] = np.clip(pop[i], -5, 5)
            
            # Evaluate
            fit = np.empty(N)
            for i in range(N):
                if time_left() < 5:
                    break
                fit[i] = wrapped_func(pop[i])
            
            # Sort by fitness
            idx = np.argsort(fit)
            pop = pop[idx]
            fit = fit[idx]
            
            # Update best and track top-k
            if fit[0] < best_f:
                best_f = fit[0]
                best_x = pop[0].copy()
            
            # Keep top 3 points for later polish
            for i in range(min(3, N)):
                if np.isfinite(fit[i]):
                    top_k_points.append(pop[i].copy())
                    top_k_fits.append(fit[i])

            # Update mean
            new_mean = np.sum(weights[:, None] * pop[:mu], axis=0)
            
            # Update step size
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L.T @ (new_mean - mean) / step_size)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2*(gen+1)))) < (1.4 + 2/(self.dim+1)) * chi_n
            step_size *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            step_size = min(step_size, 10.0)
            
            # Update covariance matrix
            if hsig:
                pc = (1 - c_c) * pc + c_c * (new_mean - mean) / step_size
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            C = (1 - c_1 - c_mu + delta_hsig) * C + c_1 * (np.outer(pc, pc) + (1 - hsig) * c_c * (2 - c_c) * C)
            for i in range(mu):
                y_i = (pop[i] - mean) / step_size
                C += c_mu * weights[i] * np.outer(y_i, y_i)
            
            mean = new_mean
            mean = np.clip(mean, -5, 5)

        # Phase 2: Multi-start local refinement
        polish_starts = []
        
        # Add top-k unique points
        seen = set()
        for pt, f in zip(top_k_points, top_k_fits):
            key = tuple(np.round(pt, 4))
            if key not in seen and np.isfinite(f):
                polish_starts.append((pt.copy(), f))
                seen.add(key)
        
        # Add periodic shifts from the best point (Büche-Rastrigin has period ~2π)
        period = 2 * math.pi
        best_pt = best_x.copy()
        for d in range(self.dim):
            for shift in [-period, period]:
                candidate = best_pt.copy()
                candidate[d] += shift
                candidate = np.clip(candidate, -5, 5)
                key = tuple(np.round(candidate, 4))
                if key not in seen:
                    polish_starts.append((candidate, None))
                    seen.add(key)
        
        # Limit to top 5 polish starts, sort by known fitness
        polish_starts = polish_starts[:5]
        polish_starts.sort(key=lambda x: x[1] if x[1] is not None else 1e10)
        
        remaining_budget = self.budget - call_count[0]
        
        for start_pt, start_f in polish_starts:
            if time_left() < 10 or call_count[0] >= self.budget - 5:
                break
            
            x = start_pt.copy()
            fx = start_f if start_f is not None and np.isfinite(start_f) else wrapped_func(x)
            
            if not np.isfinite(fx):
                continue
            
            # Initialize simplex
            simplex = np.zeros((self.dim + 1, self.dim))
            simplex[0] = x
            step = 0.5
            for i in range(self.dim):
                simplex[i + 1] = x.copy()
                simplex[i + 1][i] += step if x[i] < 5 - step else -step
                simplex[i + 1] = np.clip(simplex[i + 1], -5, 5)

            simplex_f = np.empty(self.dim + 1)
            for i in range(self.dim + 1):
                if time_left() < 5 or call_count[0] >= self.budget:
                    break
                simplex_f[i] = wrapped_func(simplex[i])
            
            if np.any(~np.isfinite(simplex_f)):
                continue

            # Nelder-Mead iterations
            max_iter_nm = min(30, remaining_budget // (self.dim + 2))
            for _ in range(max_iter_nm):
                if time_left() < 5:
                    break
                if call_count[0] >= self.budget:
                    break

                # Sort simplex
                idx = np.argsort(simplex_f)
                simplex = simplex[idx]
                simplex_f = simplex_f[idx]

                # Check convergence
                if abs(simplex_f[0] - simplex_f[-1]) < 1e-10:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = wrapped_func(x_r)

                if not np.isfinite(f_r):
                    f_r = 1e10

                if simplex_f[0] < f_r < simplex_f[-2]:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                elif f_r < simplex_f[0]:
                    # Expansion
                    gamma = 2.0
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5, 5)
                    f_e = wrapped_func(x_e)
                    if np.isfinite(f_e) and f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                else:
                    # Contraction
                    beta = 0.5
                    if f_r < simplex_f[-1]:
                        # Outside contraction
                        x_c = centroid + beta * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - beta * (x_r - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = wrapped_func(x_c)

                    if np.isfinite(f_c) and f_c < min(f_r, simplex_f[-1]):
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, self.dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex_f[i] = wrapped_func(simplex[i])
                            if time_left() < 5 or call_count[0] >= self.budget:
                                break

        return best_x, best_f
