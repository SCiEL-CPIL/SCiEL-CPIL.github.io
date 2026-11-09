import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Time guard: aim to finish well before 330s, though 500 calls is trivial.
        # We set a generous internal limit to avoid hanging on unexpected slowness.
        start_time = time.time()
        time_limit = 300.0 

        # CMA-ES Parameters
        # For dim=5 and budget=500, a population of 6-8 is reasonable.
        # lambda = 4 + floor(3 * log(dim)) is a standard heuristic, giving 4+3*1.6=8.
        # Let's use lambda=7 for slightly faster early convergence (fewer evals per gen).
        lam = 7
        
        # Initial step size. The domain is [-5, 5]. Min is in [-4, 4].
        # Starting with a step size of 2.0 covers a good portion of the space.
        sigma = 2.0
        
        # Initial mean. Center of the box is a safe bet.
        mean = np.zeros(dim)
        
        # Initial covariance
        C = np.eye(dim)
        # Inverse square root of C for sampling
        Cinv = np.linalg.inv(C)
        Cinv_sqrt = np.linalg.cholesky(Cinv).T  # Note: cholesky returns lower triangular, we want upper for standard use or just handle it
        
        # Actually, standard CMA-ES uses z ~ N(0, I), y = sigma * C_sqrt @ z.
        # We need C_sqrt. Initially C=I, so C_sqrt=I.
        C_sqrt = np.eye(dim)
        
        # Cumulation path
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Learning rates
        mu = lam // 2  # Weights for top mu individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, sqrt(mu_eff / (dim + 1) - 1)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        # Evaluation loop
        x_best = mean
        f_best = func(x_best)
        calls_made = 1
        
        # Pre-allocate population storage
        population = np.zeros((lam, dim))
        fitness = np.zeros(lam)
        
        while calls_made < budget:
            # Check time
            if time.time() - start_time > time_limit:
                break
                
            # Generate offspring
            for i in range(lam):
                # Sample z ~ N(0, I)
                z = np.random.randn(dim)
                # y = sigma * C_sqrt @ z
                y = sigma * (C_sqrt @ z)
                x = mean + y
                # Clip to box [-5, 5]
                x = np.clip(x, -5, 5)
                population[i] = x
                fitness[i] = func(x)
                calls_made += 1
                
                # Update best so far
                if fitness[i] < f_best:
                    f_best = fitness[i]
                    x_best = x.copy()
                    
            # Sort by fitness
            idx = np.argsort(fitness)
            fitness_sorted = fitness[idx]
            population_sorted = population[idx]
            
            # Update mean
            mean_old = mean
            mean = np.sum(weights[:, None] * population_sorted[:mu], axis=0)
            
            # Update step size
            # Calculate weighted mean of the new points relative to old mean
            # y_k = (x_k - mean_old) / sigma
            y_k = (population_sorted[:mu] - mean_old) / sigma
            # Weighted mean
            y_mean = np.sum(weights[:, None] * y_k, axis=0)
            
            # Update ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (Cinv_sqrt @ y_mean)
            
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update pc
            # h_sigma condition
            norm_ps = np.linalg.norm(ps)
            h_sigma = 1 if (norm_ps / np.sqrt(1 - (1 - c_sigma)**2 * (calls_made + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            
            pc = (1 - c_c) * pc + h_sigma * c_c * (mean - mean_old) / sigma
            
            # Update Covariance C
            # Delta H = c_1 * (pc * pc.T + (1 - h_sigma) * c_c * (2 - c_c) * pc * pc.T - C)
            # + c_mu * sum(weights_i * y_i * y_i.T)
            
            # We need to update C_sqrt as well, which is complex. 
            # For a simplified version, we can just recompute C_sqrt from C every generation 
            # or use a simpler update if dim is small. 
            # Given dim=5, computing eigenvalues is cheap.
            
            # Rank-one update
            diff = mean - mean_old
            # pc is already updated. 
            # The term (1 - h_sigma) * c_c * (2 - c_c) * pc * pc.T is part of the delta.
            # Let's assemble the covariance update properly.
            
            # Standard CMA-ES update for C:
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc * pc.T + (1 - h_sigma) * c_c * (2 - c_c) * pc * pc.T) 
            #     + c_mu * sum(w_i * y_i * y_i.T)
            
            # Note: y_i here are the normalized steps (x_i - mean_old)/sigma
            # We have y_k which are the top mu y's.
            
            sum_yy = np.zeros((dim, dim))
            for i in range(mu):
                sum_yy += weights[i] * np.outer(y_k[i], y_k[i])
                
            pc_outer = np.outer(pc, pc)
            correction = (1 - h_sigma) * c_c * (2 - c_c) * pc_outer
            
            C = (1 - c_1 - c_mu) * C + c_1 * (pc_outer + correction) + c_mu * sum_yy
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # Recompute C_sqrt and Cinv_sqrt
            # Use eigen decomposition for stability
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip small negative eigenvalues
            eigvals = np.clip(eigvals, 1e-12, None)
            C_sqrt = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
            Cinv = np.linalg.inv(C)
            # Cinv_sqrt is used for ps update. 
            # Cinv_sqrt @ y_mean. 
            # We can compute Cinv_sqrt via eig: (eigvecs / sqrt(eigvals)) @ eigvecs.T
            Cinv_sqrt = (eigvecs / np.sqrt(eigvals)) @ eigvecs.T
            
        return x_best, f_best
