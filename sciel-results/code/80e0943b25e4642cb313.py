import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0
        
        try:
            # CMA-ES parameters
            lam = 4 + int(3 * np.log(dim))  # Standard CMA-ES population size
            mu = lam // 2  # Number of best individuals to update weights
            
            # Weights for the best mu individuals
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            
            # Effective population size
            mueff = 1.0 / np.sum(weights ** 2)
            
            # Step size adaptation
            cs = 1 / (np.sqrt(dim + 2) - 1) + 2 / (dim + 2 - 4)
            ds = (cs + 2) * (mueff / (dim + mueff) + 2)
            
            # Covariance matrix adaptation
            c1 = (mueff + 2) / ((dim + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
            
            # Initial step size
            sigma = 2.0
            
            # Initial mean
            m = rng.uniform(-5, 5, dim)
            
            # Initial covariance matrix
            C = np.eye(dim)
            
            # Eigenvalue decomposition for efficient sampling
            eigvals, eigvecs = np.linalg.eigh(C)
            
            # State for path evolution
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Evaluate initial mean
            f_m = func(m)
            calls_made += 1
            best_x = m.copy()
            best_f = f_m
            
            while calls_made < budget and time.time() < deadline:
                # Generate population
                offspring = []
                offspring_f = []
                
                for i in range(lam):
                    if calls_made >= budget or time.time() >= deadline:
                        break
                    
                    # Sample from N(m, sigma^2 * C)
                    z = rng.standard_normal(dim)
                    # Transform using eigendecomposition
                    # C = eigvecs @ diag(eigvals) @ eigvecs.T
                    # sqrt(C) = eigvecs @ diag(sqrt(eigvals)) @ eigvecs.T
                    x_new = m + sigma * (eigvecs @ (np.sqrt(np.maximum(eigvals, 1e-10)) * (eigvecs.T @ z)))
                    x_new = np.clip(x_new, -5, 5)
                    
                    f_new = func(x_new)
                    calls_made += 1
                    offspring.append(x_new)
                    offspring_f.append(f_new)
                
                if len(offspring) == 0:
                    break
                
                # Sort offspring by fitness
                sorted_indices = np.argsort(offspring_f)
                sorted_offspring = [offspring[i] for i in sorted_indices]
                sorted_offspring_f = [offspring_f[i] for i in sorted_indices]
                
                # Update best
                if sorted_offspring_f[0] < best_f:
                    best_f = sorted_offspring_f[0]
                    best_x = sorted_offspring[0].copy()
                
                # Update mean
                m_old = m.copy()
                m = np.sum(weights[:, np.newaxis] * np.array(sorted_offspring[:mu]), axis=0)
                
                # Update path evolution strategies
                # Transform (m - m_old) / sigma
                delta = (m - m_old) / sigma
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * delta
                hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made / lam + 1)))) < \
                       (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1))
                p_c = (1 - cs) * p_c + hsig * np.sqrt(cs * (2 - cs) * mueff) * delta
                
                # Update covariance matrix
                rank_one = np.outer(p_c, p_c)
                if hsig:
                    rank_mu = np.sum(weights[:, np.newaxis, np.newaxis] * np.array([np.outer(x, x) for x in sorted_offspring[:mu]]), axis=0)
                    rank_mu = (1 - cmu) * rank_mu + c1 * (1 + (2 / (dim + 1) - 1) * mueff) * rank_one
                else:
                    rank_mu = (1 - cmu) * rank_mu + c1 * rank_one
                C = (1 - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu
                
                # Ensure C is symmetric
                C = (C + C.T) / 2
                
                # Update eigenvalue decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                
                # Adapt step size
                sigma = sigma * np.exp(cs / ds * (np.linalg.norm(p_sigma) / np.sqrt(dim) - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)
                
                # Check for convergence
                if np.linalg.norm(m - m_old) < 1e-10 * sigma or sigma < 1e-10:
                    break

        except Exception:
            pass
        
        return best_x, best_f
