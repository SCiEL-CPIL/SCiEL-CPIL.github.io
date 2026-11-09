import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin
        dim = self.dim
        total_budget = self.budget
        used = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def check_deadline():
            return time.time() > deadline

        def evaluate(x):
            nonlocal used, best_f, best_x
            x_clip = np.clip(x, -5, 5)
            f_val = func(x_clip)
            used += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x_clip.copy()
            return f_val

        # CMA-ES Implementation with fixed clipping and standard population size
        class CMAES:
            def __init__(self, x0, sigma0, budget, func, dim, global_best_x, global_best_f, used_counter, deadline_check):
                self.x = x0.copy()
                self.sigma = sigma0
                self.C = np.eye(dim)
                self.p_sigma = np.zeros(dim)
                self.p_c = np.zeros(dim)
                self.chi_N = dim ** 0.5 * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
                
                # Standard CMA-ES population size
                self.lambda_ = 4 + int(3 * np.log(dim))
                self.mu = self.lambda_ // 2
                self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
                self.weights /= np.sum(self.weights)
                self.mueff = 1 / np.sum(self.weights ** 2)
                
                self.c_sigma = (self.mueff + 2) / (dim + self.mueff + 5)
                self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (dim + 1)) - 1) + self.c_sigma
                self.c_c = (4 + self.mueff / dim) / (dim + 4 + 2 * self.mueff / dim)
                self.c_1 = 2 / ((dim + 1.3) ** 2 + self.mueff)
                self.c_mu = min(1 - self.c_1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((dim + 2) ** 2 + self.mueff))
                
                self.budget = budget
                self.func = func
                self.dim = dim
                self.global_best_x = global_best_x
                self.global_best_f = global_best_f
                self.used_counter = used_counter
                self.deadline_check = deadline_check
                self.rng = np.random.default_rng(int(np.sum(x0 * 12345.6789) + 42 + self.used_counter[0]))

            def step(self):
                if self.used_counter[0] >= self.budget or self.deadline_check():
                    return False

                # Sampling: lambda_ points
                B = self.rng.standard_normal((self.lambda_, self.dim))
                # Whitening transform
                try:
                    L = np.linalg.cholesky(self.C)
                except np.linalg.LinAlgError:
                    self.C = (self.C + self.C.T) / 2
                    L = np.linalg.cholesky(self.C + 1e-6 * np.eye(self.dim))
                
                B = B @ L.T

                # Evaluate all lambda_ points
                ys = []
                for i in range(self.lambda_):
                    x_i = self.x + self.sigma * B[i]
                    # FIX: Use clipped point for both evaluation and storage for update
                    x_clipped = np.clip(x_i, -5, 5)
                    f_i = self.func(x_clipped)
                    self.used_counter[0] += 1
                    ys.append((f_i, x_clipped))
                    if f_i < self.global_best_f[0]:
                        self.global_best_f[0] = f_i
                        self.global_best_x[0] = x_clipped.copy()

                # Sort by fitness
                ys.sort(key=lambda t: t[0])

                # Update mean
                y_w = np.zeros(self.dim)
                for i in range(self.mu):
                    y_w += self.weights[i] * (ys[i][1] - self.x) / self.sigma

                self.x = self.x + self.sigma * y_w

                # Update sigma
                norm_y_w = np.linalg.norm(self.C ** 0.5 @ y_w + 1e-16)
                self.p_sigma = (1 - self.c_sigma) * self.p_sigma + \
                               np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mueff) * y_w / norm_y_w
                self.sigma *= np.exp(self.c_sigma / self.d_sigma * (np.linalg.norm(self.p_sigma) / self.chi_N - 1))

                # Update covariance matrix
                self.p_c = (1 - self.c_c) * self.p_c + np.sqrt(self.c_c * (2 - self.c_c) * self.mueff) * y_w
                
                # hsig check
                gen_count = self.used_counter[0] 
                hsig = (np.linalg.norm(self.p_sigma) / np.sqrt(1 - (1 - self.c_sigma) ** (2 * gen_count)) <
                        (1.4 + 2 / (self.dim + 1)) * self.chi_N)
                
                # Correct Covariance Update
                y_w_sum = np.zeros((self.dim, self.dim))
                for i in range(self.mu):
                    y_i = (ys[i][1] - self.x) / self.sigma
                    y_w_sum += self.weights[i] * np.outer(y_i, y_i)
                
                if hsig:
                    hsig_term = 0.0
                else:
                    hsig_term = np.sqrt((self.mueff - 1) * (1 + 2 * (self.dim + 1) / (self.mueff - 1)))
                
                self.C = (1 + self.c_1 + self.c_mu) * self.C - \
                          self.c_1 * (self.C - self.p_c @ self.p_c.T - hsig_term * self.C) - \
                          self.c_mu * y_w_sum

                # Symmetrize C
                self.C = (self.C + self.C.T) / 2

                # Check convergence
                if self.sigma < 1e-10:
                    return False

                return True

        # Initial best is center
        evaluate(np.zeros(dim))

        # Multi-start with adaptive budgets
        remaining = total_budget - used
        if remaining <= 0:
            return best_x, best_f
            
        b1 = int(remaining * 0.4)
        b2 = int(remaining * 0.3)
        b3 = remaining - b1 - b2
        
        restart_budgets = [b1, b2, b3]
        
        for i, rb in enumerate(restart_budgets):
            if used >= total_budget or check_deadline():
                break

            if i == 0:
                x0 = np.zeros(dim)
                sigma0 = 2.0
            else:
                # Perturb best
                rng_restart = np.random.default_rng(i * 12345 + used)
                x0 = best_x + rng_restart.normal(0, 1.0, dim)
                x0 = np.clip(x0, -5, 5)
                sigma0 = 1.0

            cmaes = CMAES(x0, sigma0, rb, func, dim, [best_x], [best_f], [used], check_deadline)

            while cmaes.step():
                if used >= total_budget or check_deadline():
                    break

        return best_x, best_f
