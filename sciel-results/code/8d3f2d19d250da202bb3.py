"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for the BBOB f24 task."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        lambda_pop = max(4 + 3 * int(np.log2(dim)), 10)
        
        # Initial step size
        # The domain is [-5, 5]. A step size of 2.5 covers half the range.
        sigma = 2.5
        
        # Initial mean
        mean = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Inverse covariance matrix
        C_inv = np.eye(dim)
        
        # Learning rates
        cc = (4 + dim / 3) / ((dim + 4) ** 2 + 4)
        cs = (dim + 1.7) / (dim + 5.5 + cc)
        cmu1 = 2 / ((dim + 1.3) ** 2 + 5)
        cmu2 = min(1 - cmu1, 2 * (dim - 2 + 1. / dim) / ((dim + 2) ** 2 + 3))
        
        # Cumulation path
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Damping factors
        dsigma = 1 + 2 * max(0, np.sqrt((dim - 1) / (dim + 3)) - 1) + cs
        chi_n = np.sqrt(dim) * (1 - 1. / (4. * dim) + 1. / (21. * dim ** 2))
        
        calls_used = 0
        best_x = mean.copy()
        best_f = func(best_x)
        calls_used += 1
        
        rng = np.random.default_rng(42)

        while calls_used < budget:
            # Generate population
            # We need to sample from N(mean, sigma^2 * C)
            # Compute eigen decomposition of C for sampling
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            D_sqrt = np.sqrt(eigvals)
            
            samples = []
            for i in range(lambda_pop):
                z_i = rng.standard_normal(dim)
                y = eigvecs @ (D_sqrt * z_i)
                x_i = mean + sigma * y
                samples.append(x_i)
            
            # Evaluate population
            fitnesses = []
            for x_i in samples:
                if calls_used >= budget:
                    break
                f_i = func(x_i)
                calls_used += 1
                fitnesses.append(f_i)
                if f_i < best_f:
                    best_f = f_i
                    best_x = x_i.copy()
            
            if len(fitnesses) < lambda_pop:
                break
            
            # Sort by fitness
            sorted_indices = np.argsort(fitnesses)
            sorted_samples = [samples[i] for i in sorted_indices]
            sorted_fitnesses = [fitnesses[i] for i in sorted_indices]
            
            # Selection
            mu = lambda_pop // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = np.sum(weights) ** 2
            
            # Update mean
            new_mean = np.sum([weights[i] * sorted_samples[i] for i in range(mu)], axis=0)
            
            # Update covariance matrix
            # y = (new_mean - mean) / sigma
            y = (new_mean - mean) / sigma
            
            # Update ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_inv @ y)
            
            # Update pc
            hsig = 1.0
            norm_ps = np.linalg.norm(ps)
            if norm_ps / np.sqrt(1 - (1 - cs) ** (calls_used / cs)) < (1.4 + 2. / (dim + 1)) * chi_n:
                hsig = 1.0
            else:
                hsig = 0.0
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y
            
            # Update C
            # Rank-1 update
            delta_hsig = (2 - hsig) * cc * (1 - cc)
            # Note: sorted_samples[i] - mean uses the OLD mean
            rank2_term = sum(weights[i] * np.outer(sorted_samples[i] - mean, sorted_samples[i] - mean) for i in range(mu)) / (sigma ** 2)
            
            C = (1 + cmu1 * delta_hsig - cmu1 - cmu2) * C + \
                cmu1 * np.outer(y, y) + \
                cmu2 * rank2_term
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update C_inv
            try:
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                C_inv = np.linalg.pinv(C)
            
            # Update sigma
            sigma *= np.exp(cs / dsigma * (norm_ps / chi_n - 1))
            
            # Update mean
            mean = new_mean
            
            # Convergence check
            if sigma < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
