"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    CMA-ES with restarts (pop_size=10) for black-box optimization on [-5,5]^5.
    Uses a fixed budget of 500 evaluations. Early progress is critical for AOCC,
    so we use a relatively large population for fast initial exploration,
    then let CMA-ES refine the solution.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290.0  # 10s safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        # CMA-ES parameters
        pop_size = 10  # lambda
        mu = pop_size // 2  # number of parents
        
        def cma_es(x0, sigma0, remaining_budget):
            nonlocal calls_made, best_x, best_f
            
            if calls_made >= budget or time.time() > deadline:
                return
            
            n = dim
            # CMA-ES parameters
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            d_sigma = 1 + 2 * max(0, sqrt(mu_eff / n - 1) - 1) + c_sigma
            
            # Step size control
            chi_n = n ** 0.5 * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            
            # Initialization
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)
            
            # Strategy parameters
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            d_sigma = 1 + 2 * max(0, np.sqrt(mu_eff / n - 1) - 1) + c_sigma
            
            while calls_made < budget and time.time() < deadline:
                # Generate population
                archive = []
                for i in range(pop_size):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    y = rng.standard_normal(n)
                    x_i = x_mean + sigma * (C @ y)
                    x_i = np.clip(x_i, -5, 5)
                    f_i = func(x_i)
                    calls_made += 1
                    if f_i < best_f:
                        best_f = f_i
                        best_x = x_i.copy()
                    archive.append((f_i, x_i))
                
                if len(archive) < pop_size:
                    break
                    
                # Sort by fitness
                archive.sort(key=lambda t: t[0])
                
                # Update mean
                x_new_mean = np.zeros(n)
                for i in range(mu):
                    x_new_mean += w[i] * archive[i][1]
                x_mean = x_new_mean
                
                # Update covariance matrix
                for i in range(mu):
                    z = archive[i][1] - x_new_mean
                    C += (2 / (c_c * (1 - c_c))) * w[i] * np.outer(z, z)
                    C -= (1 - (1 - c_sigma) ** 2 * d_sigma) * c_sigma * (1 - c_sigma) * sigma ** 2 * C
                    
                # Update step size
                ps = (1 - c_sigma) * ps + c_sigma / np.sqrt(c_c) * (C @ (x_new_mean - x_mean) / sigma)
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                
                # Update evolution path
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new_mean - x_mean) / sigma
                
                # Ensure C is positive definite
                # Use eigenvalue decomposition for stability
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                
                # Clamp sigma
                sigma = np.clip(sigma, 1e-6, 50.0)
                
                # Check convergence
                if sigma < 1e-6:
                    break

        # Phase 1: Diversified initialization to find a good starting point
        n_init = 5
        candidates = [np.zeros(dim)]
        for _ in range(n_init - 1):
            candidates.append(rng.uniform(-5, 5, dim))
            
        for x0 in candidates:
            if calls_made >= budget or time.time() > deadline:
                break
            fx = func(x0)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()

        # Phase 2: CMA-ES from the best initial point
        remaining = budget - calls_made
        if remaining > 0:
            cma_es(best_x.copy(), 1.0, remaining)

        return best_x, best_f
# EVOLVE-BLOCK-END
