import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0

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

        # Phase 1: CMA-ES with Restarts
        N = 6
        mu = max(1, N // 2)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
        c_mu = (mu_eff - 2 + 2 / (self.dim + mu_eff)) / ((self.dim + 2)**2 + mu_eff)
        c_mu = min(c_mu, 2 * c_1)
        chi_n = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))

        # Reserve 25% of budget for local refinement
        cma_budget = int(self.budget * 0.75)
        
        # Initialize CMA-ES state
        def reset_cma_state():
            nonlocal mean, C, step_size, ps, pc
            mean = rng.uniform(-4.0, 4.0, self.dim)
            C = np.eye(self.dim)
            step_size = 2.0 # Start wide
            ps = np.zeros(self.dim)
            pc = np.zeros(self.dim)

        reset_cma_state()
        stagnation_count = 0
        prev_best_f = best_f
        gen = 0
        max_stagnation = 12 # Restart after 12 generations without improvement

        while call_count[0] < cma_budget and time_left() > 20:
            # Sample population
            pop = np.zeros((N, self.dim))
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(self.dim)
            
            for i in range(N):
                if time_left() < 5:
                    break
                z = rng.standard_normal(self.dim)
                pop[i] = mean + step_size * (L @ z)
                pop[i] = np.clip(pop[i], -5, 5)
                
            # Evaluate
            fit = np.empty(N)
            for i in range(N):
                if time_left() < 5:
                    break
                fit[i] = wrapped_func(pop[i])
                
            # Sort
            idx = np.argsort(fit)
            pop = pop[idx]
            fit = fit[idx]
            
            # Update Best
            if fit[0] < best_f:
                best_f = fit[0]
                best_x = pop[0].copy()
                stagnation_count = 0
            else:
                stagnation_count += 1
                
            # Check Restart Condition
            if stagnation_count >= max_stagnation:
                # Restart with a new random location
                # Keep the best solution found so far
                reset_cma_state()
                stagnation_count = 0
                gen = 0 # Reset internal gen counter for CMA-ES math if needed, though not strictly required for simple restart
                continue # Skip CMA-ES update for this generation as state is reset

            # Standard CMA-ES Update
            new_mean = np.sum(weights[:, None] * pop[:mu], axis=0)
            
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L.T @ (new_mean - mean) / step_size)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2*(gen+1)))) < (1.4 + 2/(self.dim+1)) * chi_n
            step_size *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            if hsig:
                pc = (1 - c_c) * pc + c_c * (new_mean - mean) / step_size
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            C = (1 - c_1 - c_mu + delta_hsig) * C + c_1 * (np.outer(pc, pc) + (1 - hsig) * c_c * (2 - c_c) * C)
            for i in range(mu):
                y_i = (pop[i] - mean) / step_size
                C += c_mu * weights[i] * np.outer(y_i, y_i)
            
            mean = new_mean
            mean = np.clip(mean, -5, 5)
            gen += 1

        # Phase 2: Nelder-Mead
        remaining_budget = self.budget - call_count[0]
        if remaining_budget > 10 and time_left() > 10 and best_x is not None:
            x = best_x.copy()
            
            # Initialize simplex
            simplex = np.zeros((self.dim + 1, self.dim))
            simplex[0] = x
            step = 0.5 
            for i in range(self.dim):
                simplex[i + 1] = x
                simplex[i + 1][i] += step if x[i] < 5 - step else -step
                simplex[i + 1] = np.clip(simplex[i + 1], -5, 5)

            simplex_f = np.empty(self.dim + 1)
            for i in range(self.dim + 1):
                if time_left() < 5 or call_count[0] >= self.budget:
                    break
                simplex_f[i] = wrapped_func(simplex[i])
            
            if not np.all(np.isfinite(simplex_f)):
                # If simplex init failed, try a smaller step
                step = 0.1
                for i in range(self.dim):
                    simplex[i + 1] = x
                    simplex[i + 1][i] += step if x[i] < 5 - step else -step
                    simplex[i + 1] = np.clip(simplex[i + 1], -5, 5)
                simplex_f = np.empty(self.dim + 1)
                for i in range(self.dim + 1):
                    if time_left() < 5 or call_count[0] >= self.budget:
                        break
                    simplex_f[i] = wrapped_func(simplex[i])
                if not np.all(np.isfinite(simplex_f)):
                    return best_x, best_f

            max_iter_nm = remaining_budget // (self.dim + 2)
            for _ in range(max_iter_nm):
                if time_left() < 5:
                    break
                if call_count[0] >= self.budget:
                    break

                idx = np.argsort(simplex_f)
                simplex = simplex[idx]
                simplex_f = simplex_f[idx]

                if abs(simplex_f[0] - simplex_f[-1]) < 1e-12:
                    break

                centroid = np.mean(simplex[:-1], axis=0)

                alpha = 1.0
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = wrapped_func(x_r)

                if simplex_f[0] < f_r < simplex_f[-2]:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                elif f_r < simplex_f[0]:
                    gamma = 2.0
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5, 5)
                    f_e = wrapped_func(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                else:
                    beta = 0.5
                    if f_r < simplex_f[-1]:
                        x_c = centroid + beta * (x_r - centroid)
                    else:
                        x_c = centroid - beta * (x_r - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = wrapped_func(x_c)

                    if f_c < min(f_r, simplex_f[-1]):
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        for i in range(1, self.dim + 1):
                            if time_left() < 5 or call_count[0] >= self.budget:
                                break
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex_f[i] = wrapped_func(simplex[i])

        return best_x, best_f
