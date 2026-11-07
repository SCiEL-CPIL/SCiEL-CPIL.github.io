import numpy as np
import time

class Optimizer:
    """
    Two-phase optimizer for BBOB f13 (Sharp ridge):
    Phase 1: Coarse exploration to find a good starting point and estimate ridge direction.
    Phase 2: CMA-ES with adaptive step size for fine-tuning.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        def time_ok():
            return time.time() < deadline
        
        # Phase 1: Coarse exploration with random restarts
        # Use a small number of evaluations to find a promising region
        phase1_budget = min(50, budget // 10)
        best_x = None
        best_f = np.inf
        total_evals = 0
        
        # Try several random starting points
        for _ in range(min(5, phase1_budget // 5)):
            if total_evals >= phase1_budget or not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            total_evals += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            total_evals += 1
        
        # Phase 2: CMA-ES for fine-tuning
        # Initialize CMA-ES parameters
        x = best_x.copy()
        f = best_f
        sigma = 1.0  # Initial step size
        C = np.eye(dim)  # Covariance matrix
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # CMA-ES parameters
        lam = 12  # Population size
        mu = lam // 2  # Number of recombination weights
        # Compute recombination weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Track successes for step size adaptation
        successes = []
        window_size = 10
        
        remaining_budget = budget - total_evals
        no_improve = 0
        best_ever_f = best_f
        best_ever_x = best_x.copy()
        
        while total_evals < budget and time_ok():
            # Generate population
            population = []
            fitnesses = []
            
            for i in range(lam):
                # Generate candidate
                z = rng.standard_normal(dim)
                y = x + sigma * (C @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                total_evals += 1
                
                population.append(y)
                fitnesses.append(fy)
                
                # Update global best
                if fy < best_ever_f:
                    best_ever_f = fy
                    best_ever_x = y.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                
                if not time_ok():
                    break
            
            if not time_ok() or total_evals >= budget:
                break
            
            # Sort by fitness
            indices = np.argsort(fitnesses)
            population = [population[i] for i in indices]
            fitnesses = [fitnesses[i] for i in indices]
            
            # Recombination
            x_new = np.sum([weights[i] * population[i] for i in range(mu)], axis=0)
            
            # Update covariance matrix
            y_vec = (x_new - x) / sigma
            C += c_1 * (np.outer(y_vec, y_vec) + c_1 / c_c * C)
            for i in range(mu):
                C += c_mu * weights[i] * (np.outer(population[i] - x, population[i] - x) / sigma ** 2 - C)
            
            # Update step size
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_vec
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update evolution path
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_vec
            
            # Update mean
            x = x_new
            f = fitnesses[0]
            
            # Check for convergence
            if sigma < 1e-10 or no_improve > 50:
                # Restart from best point with larger sigma
                x = best_ever_x.copy()
                sigma = 2.0
                C = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                no_improve = 0
        
        return best_ever_x, best_ever_f
