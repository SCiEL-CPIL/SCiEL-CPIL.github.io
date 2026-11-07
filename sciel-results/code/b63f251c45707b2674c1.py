"""CMA-ES based optimizer for BBOB f17."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
    CMA-ES is robust for black-box continuous optimization, especially with
    rotated/shifted landscapes, as it adapts the search distribution to the
    problem's geometry.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit, use 290s to be safe
        deadline = start_time + 290
        
        n = self.dim
        calls = 0
        best_x = np.zeros(n)
        best_f = np.inf
        
        # CMA-ES Parameters
        # Initial mean: center of the box
        mean = np.zeros(n)
        
        # Initial standard deviation
        # BBOB functions are usually scaled such that sigma=1 or 2 is reasonable
        # Given range [-5, 5], a sigma of 2.0 covers the space well initially.
        sigma = 2.0
        
        # Population size
        # Standard choice: lambda = 4 + floor(3 * log(n))
        # For n=5, log(5)~1.6, 3*1.6=4.8, floor=4, lambda=8.
        # Let's use a slightly larger population for better exploration in early stages.
        # lambda = 10 is a good balance for n=5.
        lambda_ = max(4, int(4 + 3 * np.log(n)))
        
        # Covariance matrix
        C = np.eye(n)
        
        # Step sizes for mean update
        # Standard CMA-ES formulas
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # State variables
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        # Evaluation counter
        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            
            # Clip to box [-5, 5]
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx
        
        # Initial evaluation of the mean
        evaluate(mean)
        
        # Main loop
        while calls < self.budget and time.time() < deadline:
            # Generate population
            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset C
                C = np.eye(n)
                L = np.eye(n)
                
            # Sample lambda_ points
            # Use standard normal random numbers
            # We need a deterministic RNG for reproducibility within the run?
            # CMA-ES is stochastic. Using a fixed seed for the RNG inside __call__ 
            # makes the run reproducible given the same func behavior, but func 
            # behavior depends on hidden seeds. 
            # However, the task says "hidden seeded shift...". 
            # It's generally safer to use a fixed RNG seed for the optimizer itself 
            # to ensure consistent behavior across runs if the function is static, 
            # but here the function is different per instance. 
            # Using a fixed seed (e.g., 42) for the internal RNG is fine.
            rng = np.random.default_rng(42)
            
            pop = []
            fvals = []
            
            for i in range(lambda_):
                if calls >= self.budget or time.time() > deadline:
                    break
                z = rng.standard_normal(n)
                x = mean + sigma * (L @ z)
                pop.append(x)
                fx = evaluate(x)
                fvals.append(fx)
            
            if len(pop) < 2:
                break
                
            # Sort by fitness
            indices = np.argsort(fvals)
            pop = [pop[i] for i in indices]
            fvals = [fvals[i] for i in indices]
            
            # Update mean
            new_mean = np.zeros(n)
            for i in range(mu):
                new_mean += weights[i] * pop[i]
            
            # Update evolution path for covariance (pc)
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (new_mean - mean)
            
            # Update evolution path for sigma adaptation (ps)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (L.T @ (new_mean - mean) / sigma)
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update covariance matrix
            delta_diag = 1 + 2 * (np.linalg.norm(ps)**2 / chi_n**2 > 1.4 + 2/n)
            rank_one = np.outer(pc, pc)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                zi = L.T @ (pop[i] - mean) / sigma
                rank_mu += weights[i] * np.outer(zi, zi)
            
            C = (1 + c_1 * delta_diag - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Update mean
            mean = new_mean
            
            # Check for convergence/stagnation
            # If sigma is very small or C is degenerate, we might need to restart
            # But for 500 calls, we likely won't hit full convergence.
            
            # If the best fitness hasn't improved for a while, we could restart,
            # but CMA-ES handles this via sigma adaptation.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
