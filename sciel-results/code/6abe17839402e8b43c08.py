"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """CMA-ES implementation for BBOB f23 (Katsuura)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Standard population size: lambda = 4 + floor(3 * ln(n))
        # For n=5, lambda = 4 + floor(3 * 1.609) = 4 + 4 = 8
        pop_size = 4 + int(3 * np.log(dim)) if dim > 1 else 4
        # Ensure we have at least 2 points
        pop_size = max(pop_size, 2)
        
        # Check if we can even do one generation
        # Each generation costs pop_size evaluations
        # We need at least 1 eval for initialization? No, CMA-ES samples from distribution.
        # We will sample pop_size points.
        if budget < pop_size:
             # Fallback to random search if budget is too small for CMA-ES
             rng = np.random.default_rng(42)
             best_x = rng.uniform(-5, 5, dim)
             best_f = func(best_x)
             for _ in range(budget - 1):
                 y = rng.uniform(-5, 5, dim)
                 fy = func(y)
                 if fy < best_f:
                     best_f = fy
                     best_x = y
             return best_x, best_f

        # Initialization
        rng = np.random.default_rng(42)
        
        # Start from center or random?
        # Katsuura is symmetric-ish but shifted/rotated. Center is a safe bet.
        # However, random start with large sigma might explore better initially.
        # Let's start with a random point to be safe against shifts, 
        # but initialize CMA-ES around it.
        
        # Actually, standard CMA-ES starts with a given mean.
        # Let's pick a random start point to avoid biasing towards 0 if the shift is large.
        start_x = rng.uniform(-4, 4, dim)
        
        # Initial step size
        # The box is [-5, 5]. A step size of 2.0 covers the space reasonably.
        sigma = 2.0
        
        # CMA-ES state
        mean = start_x.copy()
        C = np.eye(dim) # Covariance matrix
        C_inv = np.eye(dim) # Inverse covariance
        # Pre-compute eigen decomposition for efficient sampling and updates
        # For small dim, we can just use Cholesky or eigendecomp directly
        
        # Learning rates
        # Standard values
        mu = pop_size // 2  # Number of selected points
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size adaptation (CSA)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        
        # Covariance matrix adaptation
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Path accumulators
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Eigen decomposition of C for sampling
        # We need to sample N(0, C).
        # x = mean + sigma * z, where z ~ N(0, C)
        # We can use Cholesky decomposition of C.
        
        calls_made = 0
        best_f = float('inf')
        best_x = None
        
        # Main loop
        while calls_made < budget:
            # Check if we can afford another generation
            if calls_made + pop_size > budget:
                # Use remaining budget for random search or (1+1)-ES around best
                # (1+1)-ES is more efficient for few remaining calls
                remaining = budget - calls_made
                if best_x is None:
                    # Should not happen, but safety
                    best_x = rng.uniform(-5, 5, dim)
                    best_f = func(best_x)
                    calls_made += 1
                    remaining -= 1
                sigma_local = 0.5
                for _ in range(remaining):
                    y = np.clip(best_x + sigma_local * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y
                        sigma_local *= 1.1
                    else:
                        sigma_local *= 0.9
                break

            # Update eigen decomposition of C
            # Since dim is small (5), we can do eigendecomp every step or use Cholesky
            # Cholesky is faster but requires C to be positive definite.
            # CMA-ES updates can make it slightly ill-conditioned.
            # Let's use eigendecomposition for robustness.
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            # Ensure positive eigenvalues
            eigenvalues = np.maximum(eigenvalues, 1e-12)
            
            # Sample population
            # z = eigenvectors * (sqrt(eigenvalues) * standard_normal)
            Z = rng.standard_normal((pop_size, dim))
            # Transform Z to have covariance C
            # X = Z @ diag(sqrt(eigenvalues)) @ eigenvectors.T
            # This is equivalent to sampling from N(0, C)
            # Let's do: Y = Z * sqrt(eigenvalues) [broadcasting]
            Y = Z * np.sqrt(eigenvalues)
            # Then rotate
            X = Y @ eigenvectors.T
            
            # Add mean and scale
            Pop = mean + sigma * X
            Pop = np.clip(Pop, -5, 5)
            
            # Evaluate population
            F = np.empty(pop_size)
            for i in range(pop_size):
                F[i] = func(Pop[i])
                calls_made += 1
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = Pop[i].copy()
            
            # Sort population by fitness
            sort_idx = np.argsort(F)
            Pop = Pop[sort_idx]
            F = F[sort_idx]
            
            # Select top mu individuals
            Y = Pop[:mu]
            
            # Weighted mean of selected individuals
            new_mean = np.sum(weights[:, None] * Y, axis=0)
            
            # Update step size
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (new_mean - mean) / sigma
            # We need C_inv_sqrt. From eigen decomp:
            # C_inv_sqrt = eigenvectors @ diag(1/sqrt(eigenvalues)) @ eigenvectors.T
            C_inv_sqrt = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (new_mean - mean) / sigma)
            
            norm_p_sigma = np.linalg.norm(p_sigma)
            # h_sigma condition
            # E[||N(0, I)||] approx
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            h_sigma = 1 if (norm_p_sigma / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2/(dim+1)) * chi_n) else 0
            
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            
            # Update covariance matrix
            # p_c = (1 - c_c) * p_c + h_sigma * sqrt(c_c * (2 - c_c) * mu_eff) * (new_mean - mean) / sigma
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * ((new_mean - mean) / sigma)
            
            # Rank-1 and Rank-mu update
            # C = (1 + c_c * (2 - c_c) - c_1) * C 
            #       + c_1 * (p_c * p_c.T - (1 - h_sigma) * (1 - c_c) * C)
            #       + c_mu * sum(weights[i] * (Y[i] - mean) * (Y[i] - mean).T)
            
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            
            C = (1 + c_c * (2 - c_c) - c_1 - delta_h) * C \
                + c_1 * (np.outer(p_c, p_c) + delta_h * C) \
                + c_mu * np.sum(weights[:, None, None] * np.outer(Y - mean, Y - mean), axis=0)
                
            # Update mean
            mean = new_mean
            
            # Ensure C is symmetric positive definite (numerical stability)
            C = (C + C.T) / 2
            # Add small regularization if needed
            # C += 1e-12 * np.eye(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
