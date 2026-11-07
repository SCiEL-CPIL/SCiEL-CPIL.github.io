"""Rewritten optimizer for BBOB f4 (Büche-Rastrigin) using CMA-ES with multi-start."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer with multi-start for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        rng = np.random.default_rng(42)
        
        calls_counter = [0]
        n = self.dim
        
        def make_cmaes(x0, sigma0, budget_local):
            """Run CMA-ES from x0 with initial step sigma0, using up to budget_local calls."""
            # CMA-ES parameters
            pop_size = max(4, int(4 + 3 * np.log(n)))  # Standard population size
            pop_size = min(pop_size, 20)  # Cap for efficiency
            max_iter = budget_local // pop_size + 1
            
            # CMA-ES state
            x = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            C_inv = np.eye(n)
            
            # Update rates
            mu = pop_size // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / np.sum(w)
            mueff = 1.0 / np.sum(w**2)
            c_sigma = (mueff + 2) / (n + mueff + 5)
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
            
            # Evolution path
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            
            # Initial population
            best_x = x.copy()
            best_f = func(x)
            calls_counter[0] += 1
            if calls_counter[0] > self.budget:
                return best_x, best_f, calls_counter[0]
            
            for gen in range(max_iter):
                if calls_counter[0] >= self.budget or time.time() > deadline:
                    break
                
                # Generate population
                population = np.zeros((pop_size, n))
                fitnesses = np.zeros(pop_size)
                
                for i in range(pop_size):
                    if calls_counter[0] >= self.budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(n)
                    # Cholesky of C
                    try:
                        C_chol = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        C_chol = np.eye(n)
                    population[i] = x + sigma * C_chol @ z
                    fitnesses[i] = func(population[i])
                    calls_counter[0] += 1
                
                if calls_counter[0] >= self.budget or time.time() > deadline:
                    break
                
                # Sort by fitness
                order = np.argsort(fitnesses)
                population_sorted = population[order]
                fitnesses_sorted = fitnesses[order]
                
                # Update best
                if fitnesses_sorted[0] < best_f:
                    best_f = fitnesses_sorted[0]
                    best_x = population_sorted[0].copy()
                
                # Compute weighted mean
                m_new = np.sum(w[:mu, np.newaxis] * population_sorted[:mu], axis=0)
                
                # Update step size (sigma)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (C_inv @ (m_new - x))
                sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mueff) - 1))
                sigma = max(sigma, 1e-12)
                sigma = min(sigma, 10.0)
                
                # Update covariance matrix
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * (m_new - x) / sigma
                delta_heav = (1 + (1 - c_c) * (1 - c_c) * (mueff - 1)) * c_c
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - c_c) * (1 - c_c) * (mueff - 1) * C)
                for i in range(mu):
                    y_i = (population_sorted[i] - m_new) / sigma
                    C += c_mu * w[i] * np.outer(y_i, y_i)
                
                # Update inverse covariance
                try:
                    C_inv = np.linalg.inv(C)
                except np.linalg.LinAlgError:
                    C_inv = np.eye(n)
                
                x = m_new
            
            return best_x, best_f, calls_counter[0]

        # Multi-start CMA-ES strategy
        # First, do a few random starts to find good basins
        best_x = None
        best_f = np.inf
        
        # Try several starting points
        n_starts = 3
        budget_per_start = self.budget // n_starts
        
        for s in range(n_starts):
            if calls_counter[0] >= self.budget or time.time() > deadline:
                break
            
            # Different starting points
            if s == 0:
                x0 = np.zeros(n)
                sigma0 = 2.0
            elif s == 1:
                x0 = rng.uniform(-3, 3, n)
                sigma0 = 1.5
            else:
                x0 = rng.uniform(-4, 4, n)
                sigma0 = 1.0
            
            local_budget = self.budget - calls_counter[0]
            if s < n_starts - 1:
                local_budget = min(budget_per_start, local_budget)
            
            bx, bf, _ = make_cmaes(x0, sigma0, local_budget)
            
            if bf < best_f:
                best_f = bf
                best_x = bx
        
        return best_x, best_f
# EVOLVE-BLOCK-END
