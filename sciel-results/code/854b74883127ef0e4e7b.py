"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer with adaptive population size."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0 # Safety margin before 300s limit
        
        best_x = None
        best_f = float('inf')
        call_count = 0
        rng = np.random.default_rng(42)

        # CMA-ES Parameters
        # Initial sigma (step size)
        sigma = 2.0 
        # Population size: lambda = 4 + floor(3 * log(dim)) is standard, but for dim=5, lambda=9.
        # We use a slightly larger lambda to ensure coverage, e.g., 10 or 15.
        # With budget 500, we can afford a decent number of generations.
        lambda_pop = 10 
        mu = lambda_pop // 2 # Selection pressure
        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size control parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff)))
        
        # State variables
        mean = np.zeros(dim)
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim) # Initially sqrt(C^-1)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Precompute constants
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Helper to evaluate and update best
        def evaluate(x):
            nonlocal best_x, best_f, call_count
            if call_count >= budget:
                return float('inf')
            x = np.clip(x, -5, 5)
            f = func(x)
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Initial evaluation of mean
        f_mean = evaluate(mean)
        
        # Main loop
        while call_count < budget and time.time() < deadline:
            # Generate population
            Y = []
            X = []
            F = []
            
            for i in range(lambda_pop):
                # Sample Z
                Z = rng.standard_normal(dim)
                # Transform Z using C_inv_sqrt
                # Note: We maintain C_inv_sqrt such that C_inv_sqrt @ C_inv_sqrt.T = C^-1
                # This is an approximation for speed; a full Cholesky update is better but slower.
                # For dim=5, we can afford a proper Cholesky decomposition of C each generation.
                
                # Proper Cholesky of C
                try:
                    L = np.linalg.cholesky(C)
                    # C = L L^T
                    # We want x = mean + sigma * L @ Z
                    y = L @ Z
                except np.linalg.LinAlgError:
                    # Fallback if C is not positive definite
                    y = Z
                
                x = mean + sigma * y
                x = np.clip(x, -5, 5)
                f = evaluate(x)
                
                Y.append(Z)
                X.append(x)
                F.append(f)
                
                if call_count >= budget or time.time() > deadline:
                    break
            
            if len(X) < mu:
                break
                
            # Sort by fitness
            sort_idx = np.argsort(F)
            X = [X[i] for i in sort_idx]
            Y = [Y[i] for i in sort_idx]
            F = [F[i] for i in sort_idx]
            
            # Update Mean
            old_mean = mean.copy()
            mean = np.sum([w * X[i] for i, w in enumerate(weights)], axis=0)
            
            # Update Covariance Matrix
            # First, update C using the rank-mu update
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - (1 - c_sigma)^2) * C) + c_mu * sum w_i (y_i @ y_i.T)
            
            # Normalize Y for covariance update (using sigma)
            Y_norm = [Y[i] / sigma for i in range(mu)]
            
            # Rank-mu update
            C_update = np.zeros((dim, dim))
            for i in range(mu):
                C_update += weights[i] * np.outer(Y_norm[i], Y_norm[i])
            
            # Rank-one update for pc
            # pc = (1 - c_sigma) * pc + c_sigma * sqrt(mu_eff) * (mean - old_mean) / sigma
            pc = (1 - c_sigma) * pc + c_sigma * np.sqrt(mu_eff) * (mean - old_mean) / sigma
            
            # Update C
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - (1 - c_sigma)**2) * C) + c_mu * C_update
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update ps
            # ps = (1 - c_sigma) * ps + c_sigma * sqrt(mu_eff) * (mean - old_mean) / sigma
            # Wait, standard CMA-ES uses the normalized difference for ps update in some variants, 
            # but usually: ps = (1-c_sigma)ps + c_sigma * sqrt(mu_eff) * (x_mean - x_mean_old) / sigma
            ps = (1 - c_sigma) * ps + c_sigma * np.sqrt(mu_eff) * (mean - old_mean) / sigma
            
            # Adapt sigma
            # sigma = sigma * exp(c_sigma / d_sigma * (||ps|| / chi_n - 1))
            ps_norm = np.linalg.norm(ps)
            sigma *= np.exp(c_sigma / d_sigma * (ps_norm / chi_n - 1))
            
            # Clamp sigma
            if sigma < 1e-12:
                break
            if sigma > 10:
                sigma = 10

        return best_x, best_f

# EVOLVE-BLOCK-END
