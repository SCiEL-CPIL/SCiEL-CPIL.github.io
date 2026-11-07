"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with adaptive restarts for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s limit

        calls_made = 0
        best_x = None
        best_f = float('inf')

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # CMA-ES parameters
        pop_size = 12  # lambda
        min_sigma = 1e-6
        max_sigma = 5.0
        convergence_threshold = 1e-10
        stagnation_window = 30  # No improvement in this many evaluations triggers restart
        max_evals_per_run = 80  # Max evaluations per CMA-ES run before forced restart

        # Initial population
        x0 = np.random.uniform(-4.0, 4.0, dim)
        sigma = 1.5

        while calls_made < budget and time.time() < deadline:
            # CMA-ES state
            mean = x0.copy()
            sigma = max(sigma * 0.5, min_sigma)  # Shrink sigma for restart
            C = np.eye(dim)
            C_sqrt = np.eye(dim)
            step_size = 1.0
            
            # Evolution paths
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Adaptation parameters
            mu = pop_size // 2  # number of parents
            weights = np.ones(mu)
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)
            
            # Learning rates
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (weights[0] - 0.5) / ((dim + 2) ** 2 + mu_eff))
            
            # Damping
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            
            stagnation_count = 0
            local_best_f = float('inf')
            evals_this_run = 0
            
            while calls_made < budget and time.time() < deadline:
                if evals_this_run >= max_evals_per_run:
                    break
                    
                # Generate population
                population = []
                for i in range(pop_size):
                    z = np.random.randn(dim)
                    x_candidate = mean + sigma * (C_sqrt @ z)
                    population.append(x_candidate)
                
                # Evaluate population
                f_values = []
                for x_candidate in population:
                    x_candidate = np.clip(x_candidate, -5.0, 5.0)
                    f_val = evaluate(x_candidate)
                    f_values.append(f_val)
                    evals_this_run += 1
                    calls_made += 1
                    
                    if calls_made >= budget or time.time() > deadline:
                        break
                
                if calls_made >= budget or time.time() > deadline:
                    break
                
                # Sort by fitness
                order = np.argsort(f_values)
                population = [population[i] for i in order]
                f_values = [f_values[i] for i in order]
                
                # Update local best
                if f_values[0] < local_best_f:
                    local_best_f = f_values[0]
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                
                # Check for stagnation
                if stagnation_count >= stagnation_window:
                    break
                
                # Check convergence
                if f_values[-1] - f_values[0] < convergence_threshold * (1 + abs(f_values[0])):
                    break
                
                # Update mean
                mean_new = sum(weights[i] * population[i] for i in range(mu))
                
                # Update sigma (CSA - Cumulative Step Adaptation)
                y_mean = (mean_new - mean) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * y_mean
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (evals_this_run + 1))) / chi_n * (1 + 0.3 * min(1, (dim + 2) / dim - 1))
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = max(min_sigma, min(max_sigma, sigma))
                
                # Update correlation matrix
                if h_sigma < 1.4 + 2 / (dim + 1):
                    p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * y_mean
                
                C = (1 - c_1 - c_mu) * C + c_1 * (2 - h_sigma) * np.outer(p_c, p_c) + c_mu * sum(weights[i] * np.outer(population[i] - mean, population[i] - mean) for i in range(mu))
                
                # Update C_sqrt via eigenvalue decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-20, None)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
                
                mean = mean_new
                
                # Check if we've made significant progress
                if f_values[0] < best_f - 1e-8:
                    stagnation_count = 0
            
            # Restart: either perturb best or new random point
            if calls_made < budget and time.time() < deadline:
                if np.random.random() < 0.25:
                    x0 = np.random.uniform(-4.0, 4.0, dim)
                    sigma = 1.5
                else:
                    x0 = best_x + np.random.normal(0, 0.3, dim)
                    x0 = np.clip(x0, -4.0, 4.0)
                    sigma = 0.5

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
