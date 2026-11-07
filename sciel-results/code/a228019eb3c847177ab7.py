"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer for BBOB f7 (Step Ellipsoidal)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # Safety margin
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        n_calls = 0
        best_x = np.zeros(d)
        best_f = np.inf

        def check_time():
            return time.time() < deadline and n_calls < budget

        def evaluate(x):
            nonlocal n_calls, best_x, best_f
            if not check_time():
                return None
            x = np.asarray(x, dtype=float)
            if x.shape != (d,) or not np.all(np.isfinite(x)):
                return None
            x = np.clip(x, -5, 5)
            fx = func(x)
            n_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global Exploration
        n_global = 15
        for _ in range(n_global):
            if not check_time():
                break
            x = rng.uniform(-5, 5, d)
            evaluate(x)

        # Phase 2: CMA-ES
        # Implement a simple CMA-ES from scratch
        # Population size
        lambda_pop = 4 + int(3 * np.log(d))  # ~15 for d=5
        # Step size (sigma)
        sigma = 2.0
        # Mean
        mean = best_x.copy()
        # Covariance matrix
        C = np.eye(d)
        # Evolution paths
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        # Learning rates
        c_sigma = (lambda_pop + 2) / (d + lambda_pop + 5)
        c_c = (4 + lambda_pop / d) / (d + 4 + 2 * lambda_pop / d)
        c_1 = 2 / ((d + 1.3)**2 + lambda_pop)
        c_mu = min(1 - c_1, 2 * ((lambda_pop / 2 - 1 + 1 / d) / (d + 2) - 1))
        # Damping for sigma
        d_sigma = 1 + 2 * max(0, np.sqrt((lambda_pop - 1) / (d + 1)) - 1) + c_sigma
        # Chi distribution
        chi_lambda = np.sqrt(lambda_pop) * (1 - 1/(4*lambda_pop) + 1/(21*lambda_pop**2))
        
        # CMA-ES main loop
        stagnation_counter = 0
        max_stagnation = 30
        
        while check_time():
            # Generate population
            pop = []
            for i in range(lambda_pop):
                if not check_time():
                    break
                # Sample from multivariate normal
                z = rng.standard_normal(d)
                # Transform using C and sigma
                # Use Cholesky decomposition
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(d)
                x_i = mean + sigma * (L @ z)
                x_i = np.clip(x_i, -5, 5)
                pop.append(x_i)
            
            if len(pop) < 2:
                break
            
            # Evaluate population
            f_vals = []
            for x_i in pop:
                fx = evaluate(x_i)
                if fx is None:
                    f_vals.append(np.inf)
                else:
                    f_vals.append(fx)
            
            # Sort by fitness
            indices = np.argsort(f_vals)
            mu = lambda_pop // 2  # Number of weights
            
            # Compute weights
            weights = np.zeros(mu)
            log_weights = np.log(np.arange(mu, 0, -1))
            weights = np.exp(log_weights - np.mean(log_weights))
            weights /= weights.sum()
            mu_eff = 1 / np.sum(weights**2)
            
            # Update mean
            new_mean = np.zeros(d)
            for i in range(mu):
                new_mean += weights[i] * pop[indices[i]]
            
            # Update sigma
            sigma_old = sigma
            if np.linalg.norm(new_mean - mean) > 1e-12:
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) / sigma * (new_mean - mean)
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_lambda - 1))
            else:
                p_sigma = (1 - c_sigma) * p_sigma
            
            # Update covariance matrix
            p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) / sigma * (new_mean - mean)
            
            delta_h = 1 if np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c)**2) < (1.4 + 2/(d+1)) * chi_lambda else 0
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - delta_h) * C)
            
            for i in range(mu):
                y_i = (pop[indices[i]] - mean) / sigma
                C += c_mu * weights[i] * np.outer(y_i, y_i)
            
            mean = new_mean
            mean = np.clip(mean, -5, 5)
            
            # Check for stagnation
            if f_vals[indices[0]] >= best_f - 1e-8:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            
            # If stagnated, restart with perturbation
            if stagnation_counter >= max_stagnation:
                # Perturb mean
                mean = best_x + rng.standard_normal(d) * sigma
                mean = np.clip(mean, -5, 5)
                sigma = max(sigma * 0.5, 0.1)
                C = np.eye(d)
                p_sigma = np.zeros(d)
                p_c = np.zeros(d)
                stagnation_counter = 0
        
        return best_x, best_f
# EVOLVE-BLOCK-END
