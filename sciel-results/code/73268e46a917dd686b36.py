"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """CMA-ES with local Nelder-Mead refinement for Sharp Ridge (f13)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        best_x_global = None
        best_f_global = np.inf
        calls = 0

        def time_ok():
            return time.time() < deadline

        def check_and_update(x, f):
            nonlocal best_x_global, best_f_global
            if f < best_f_global:
                best_f_global = f
                best_x_global = x.copy()

        def nelder_mead(x0, f0, max_calls=50):
            nonlocal calls, best_x_global, best_f_global
            
            # Initialize simplex
            simplex = [x0.copy()]
            # Use a moderate initial step size
            step_size = 0.5
            for i in range(self.dim):
                x_new = x0.copy()
                x_new[i] += step_size
                simplex.append(x_new)
            
            # Evaluate simplex
            f_values = [f0]
            for i in range(1, self.dim + 1):
                if not time_ok() or calls >= self.budget - 5:
                    break
                fx = func(simplex[i])
                calls += 1
                f_values.append(fx)
                check_and_update(simplex[i], fx)
            
            if len(f_values) < self.dim + 1:
                return best_x_global, best_f_global

            alpha = 1.0    # reflection
            gamma = 2.0    # expansion
            rho = 0.5      # contraction
            sigma = 0.5    # shrink

            max_iter = 100
            for _ in range(max_iter):
                if not time_ok() or calls >= self.budget - 5:
                    break

                # Sort simplex by function value
                order = np.argsort(f_values)
                simplex = [simplex[i] for i in order]
                f_values = [f_values[i] for i in order]

                # Check convergence
                if f_values[-1] - f_values[0] < 1e-12:
                    break

                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls += 1
                check_and_update(x_r, f_r)

                if f_values[0] <= f_r < f_values[-2]:
                    # Replace worst with reflected
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                elif f_r < f_values[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls += 1
                    check_and_update(x_e, f_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_values[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (x_r - centroid)
                    
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls += 1
                    check_and_update(x_c, f_c)

                    if f_c < min(f_r, f_values[-1]):
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_new = func(simplex[i])
                            calls += 1
                            f_values[i] = f_new
                            check_and_update(simplex[i], f_new)
                            if calls >= self.budget - 5 or not time_ok():
                                break

            return best_x_global, best_f_global

        # CMA-ES Implementation
        # Parameters
        n = self.dim
        # Population size lambda
        lambda_ = 4 + int(3 * math.log(n))
        # Initial sigma
        sigma = 2.0
        # Initial mean: random point in [-4, 4]^5
        x_mean = rng.uniform(-4, 4, n)
        
        # Evaluate initial mean
        f_mean = func(x_mean)
        calls += 1
        check_and_update(x_mean, f_mean)

        # Covariance matrix C
        C = np.eye(n)
        # Inverse covariance
        C_inv = np.eye(n)
        # Step size adaptation
        cs = 1 / (n + 4)
        ds = 1 + 2 * max(0, (n + 1) / 3 - 1) + cs
        c1 = (2 + 2 / (n + 1.3)) / ((n + 2) ** 2 + 2)
        cmu = min(1 - c1, 2 * (sqrt(lambda_) - 2 + 1 / sqrt(lambda_)) / ((n + 2) ** 2 + 2))
        
        # Evolution paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Initial population
        population = []
        f_values = []
        
        # Generate initial population
        for i in range(lambda_):
            z = rng.standard_normal(n)
            x = x_mean + sigma * z
            x = np.clip(x, -5, 5)
            population.append(x)
            f = func(x)
            calls += 1
            f_values.append(f)
            check_and_update(x, f)
            
        if calls >= self.budget - 5 or not time_ok():
            return best_x_global, best_f_global

        # Sort population by fitness
        indices = np.argsort(f_values)
        population = [population[i] for i in indices]
        f_values = [f_values[i] for i in indices]
        
        # Selection
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)
        
        # Main CMA-ES loop
        while calls < self.budget - 5 and time_ok():
            # Update mean
            x_new_mean = np.sum(np.array(population[:mu]) * weights[:, np.newaxis], axis=0)
            
            # Update sigma
            z_old = (x_mean - x_new_mean) / (sigma * np.sqrt(np.diag(C)))
            p_sigma = (1 - cs) * p_sigma + cs * np.sign(np.dot(z_old, x_new_mean - x_mean))
            h_sigma = (np.linalg.norm(p_sigma) / math.sqrt(1 - (1 - cs) ** (2 * (calls + 1))) < (1.4 + 2 / (n + 1)) * np.sqrt((mu_eff + 1) / (n + 1)))
            
            # Update covariance
            p_c = (1 - c1) * p_c + h_sigma * math.sqrt(c1 * (2 - c1) * mu_eff) * (x_new_mean - x_mean) / sigma
            
            y = np.array(population[:mu]) - x_mean
            delta_h = (1 - h_sigma) * c1 * (2 - c1) * np.trace(C)
            
            C = (1 + c1 * delta_h - c1 - cmu) * C + c1 * np.outer(p_c, p_c) + cmu / sigma ** 2 * np.sum(weights[:, np.newaxis, np.newaxis] * y[:, np.newaxis, :] * y[:, np.newaxis, :], axis=0)
            
            # Ensure positive definiteness
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C_inv = eigvecs @ np.diag(1 / eigvals) @ eigvecs.T
            
            # Update sigma
            sigma *= math.exp(cs / ds * (np.linalg.norm(p_sigma) / np.sqrt(n) - 1))
            sigma = np.clip(sigma, 1e-12, 10)
            
            x_mean = x_new_mean
            x_mean = np.clip(x_mean, -5, 5)
            f_mean = func(x_mean)
            calls += 1
            check_and_update(x_mean, f_mean)
            
            # Generate new population
            population = []
            f_values = []
            for i in range(lambda_):
                z = rng.standard_normal(n)
                # Cholesky decomposition for sampling
                try:
                    L = np.linalg.cholesky(C)
                    x = x_mean + sigma * (L @ z)
                except np.linalg.LinAlgError:
                    # Fallback if Cholesky fails
                    x = x_mean + sigma * z
                x = np.clip(x, -5, 5)
                population.append(x)
                f = func(x)
                calls += 1
                f_values.append(f)
                check_and_update(x, f)
                
            if calls >= self.budget - 5 or not time_ok():
                break
                
            # Sort and select
            indices = np.argsort(f_values)
            population = [population[i] for i in indices]
            f_values = [f_values[i] for i in indices]
            
            # Check for stagnation
            if len(f_values) > 0 and f_values[0] == f_values[1] and np.linalg.norm(population[0] - population[1]) < 1e-8:
                break

        # Local refinement with Nelder-Mead on the best point found
        if best_x_global is not None and calls < self.budget - 5 and time_ok():
            nelder_mead(best_x_global, best_f_global, max_calls=self.budget - calls - 5)

        return best_x_global, best_f_global

# EVOLVE-BLOCK-END
