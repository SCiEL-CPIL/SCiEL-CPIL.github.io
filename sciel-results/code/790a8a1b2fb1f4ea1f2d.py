"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) implementation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Population size
        lambda_pop = max(4, int(4 + 3 * np.log(d)))
        
        # Step size
        sigma = 2.0
        
        # Initial point
        x = np.zeros(d)
        
        # Initial covariance
        C = np.eye(d)
        
        # Evolution paths
        ps = np.zeros(d)
        pc = np.zeros(d)
        
        # Learning rates
        cs = (lambda_pop + 2) / (d + lambda_pop + 5)
        cmu = 2 / ((d + 2.)**2 + lambda_pop)
        cc = (4 + lambda_pop / d) / (d + 4 + 2 * lambda_pop / d)
        c1 = 2 / ((d + 1.3)**2 + lambda_pop)
        cmu = min(cmu, 2 * (1 - c1) / ((lambda_pop - 2 + 1 / 3) * (d + 2)))
        
        # Damping
        ds = 1 + 2 * max(0, np.sqrt((lambda_pop - 1) / (d + 1)) - 1) + cs
        dc = 1 + 2 * max(0, np.sqrt((lambda_pop - 1) / (d + 1)) - 1) + cc
        
        # Eigen decomposition cache
        D = np.ones(d)
        B = np.eye(d)
        chiN = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))
        
        # Weights
        mu = lambda_pop // 2
        w = np.log(np.arange(1, mu + 1)) - np.log(mu)
        w /= np.sum(w)
        mueff = 1 / np.sum(w**2)
        
        # Cholesky of C
        try:
            C_sqrt = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            C_sqrt = np.eye(d)
        
        best_x = x.copy()
        best_f = np.inf
        calls = 0
        
        # Initial evaluation
        fx = func(x)
        calls += 1
        if fx < best_f:
            best_f = fx
            best_x = x.copy()
        
        while calls < budget:
            # Update eigen decomposition
            try:
                D_inv_sqrt = 1 / np.sqrt(D)
            except:
                D_inv_sqrt = np.ones(d)
            
            # Generate offspring
            offspring = []
            fitness = []
            
            for i in range(lambda_pop):
                z = np.random.randn(d)
                # Transform z using C^(1/2)
                # y = x + sigma * B * D * z
                y = x + sigma * (B @ (D * z))
                # Clip to bounds
                y = np.clip(y, -5, 5)
                offspring.append(y)
                fy = func(y)
                calls += 1
                fitness.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                
                if calls >= budget:
                    break
            
            if len(offspring) == 0:
                break
                
            fitness = np.array(fitness)
            offspring = np.array(offspring)
            
            # Sort by fitness
            idx = np.argsort(fitness)
            offspring = offspring[idx]
            fitness = fitness[idx]
            
            # Update mean
            x_old = x.copy()
            x = sum(w[i] * offspring[i] for i in range(mu))
            
            # Update evolution paths
            # Compute C^(-1/2) (x - x_old)
            diff = x - x_old
            try:
                # Approximate C^(-1/2) using B and D
                # C = B D^2 B^T, so C^(-1/2) = B D^(-1) B^T
                diff_trans = B.T @ (diff / D)
            except:
                diff_trans = diff
            
            ps = (1 - cs) * ps + cs * np.sqrt(ds) * diff_trans
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (calls // lambda_pop)))) < (1.4 + 2 / (d + 1)) * chiN
            
            pc = (1 - cc) * pc + hsig * np.sqrt(dc * cc * (2 - cc)) * diff_trans
            
            # Update covariance
            try:
                # C^(-1/2)
                C_inv_sqrt = B.T @ np.diag(1 / D)
                C_inv_sqrt = C_inv_sqrt @ B.T  # This is not quite right, let's use a simpler update
                
                # Standard CMA-ES covariance update
                # delta_hsig = (1 - hsig) * cc * (2 - cc)
                delta_hsig = (1 - hsig) * cc * (2 - cc)
                
                # Outer product of pc
                pc_outer = np.outer(pc, pc)
                
                # Weighted sum of outer products of selected offspring
                sum_outer = sum(
                    w[i] * np.outer(offspring[i] - x_old, offspring[i] - x_old)
                    for i in range(mu)
                )
                
                # Update C
                C = (1 + c1 * delta_hsig - c1 - cmu) * C + c1 * (pc_outer + delta_hsig * C) + cmu * sum_outer
                
                # Ensure symmetry
                C = (C + C.T) / 2
                
                # Eigen decomposition
                D2, B = np.linalg.eigh(C)
                D2 = np.clip(D2, 1e-20, None)
                D = np.sqrt(D2)
                
                # Check for numerical issues
                if not np.all(np.isfinite(D)):
                    break
            except:
                break
            
            # Adapt step size
            # Use CSA (Covariance Matrix Self-Adaptation) or simple rule
            # Simple rule based on success rate
            success_rate = np.sum(fitness[:mu] < fitness[mu:]) / mu if mu < lambda_pop else 0.5
            if success_rate > 0.5:
                sigma *= 1.05
            else:
                sigma *= 0.95
            
            sigma = np.clip(sigma, 1e-8, 10)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
