"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-phase optimizer for 5D Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s deadline with 50s safety margin
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        calls = [0]
        
        def call_func(x):
            calls[0] += 1
            return func(x)
        
        # Phase 1: Coarse random exploration
        best_x = None
        best_f = np.inf
        num_initial = 30
        for i in range(num_initial):
            if calls[0] >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = call_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: CMA-ES (main optimization)
        x = best_x.copy()
        fx = best_f
        
        # CMA-ES parameters
        pop_size = dim + int(3 * np.log(dim))  # ~8 for dim=5
        pop_size = max(4, min(pop_size, 12))
        
        sigma = 1.0
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        cs = (4 + dim / 3) / (dim + 4)
        cmu = 2 / ((dim + 1) ** 2 + 1)
        c1 = (2 + 2 / (dim + 1)) / ((dim + 2) ** 2 + 2)
        cc = (4 + dim / dim) / (dim + 4)
        
        mu = pop_size // 2 + 1  # number of recombination points
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1 / np.sum(w ** 2)
        
        # Track progress for restarts
        no_improvement = 0
        best_f_so_far = fx
        restart_count = 0
        max_restarts = 3
        stagnation_threshold = 50  # iterations without improvement before considering restart
        
        for gen in range(budget - calls[0]):
            if calls[0] >= budget or time.time() > deadline:
                break
            
            # Generate population
            offspring = []
            for i in range(pop_size):
                if calls[0] >= budget or time.time() > deadline:
                    break
                
                # Sample from multivariate normal
                z = rng.standard_normal(dim)
                y = x + sigma * C_sqrt @ z
                y = np.clip(y, -5, 5)
                fy = call_func(y)
                offspring.append((fy, y))
            
            if len(offspring) == 0:
                break
            
            # Sort offspring by fitness
            offspring.sort(key=lambda t: t[0])
            
            # Update best
            if offspring[0][0] < fx:
                fx = offspring[0][0]
                x = offspring[0][1].copy()
                no_improvement = 0
            else:
                no_improvement += 1
            
            if fx < best_f_so_far:
                best_f_so_far = fx
            
            # Check for stagnation and restart
            if no_improvement > stagnation_threshold and restart_count < max_restarts:
                restart_count += 1
                # Restart with larger sigma
                sigma = max(sigma * 2, 1.0)
                no_improvement = 0
                # Optionally, perturb the center
                x = x + rng.standard_normal(dim) * sigma * 0.5
                x = np.clip(x, -5, 5)
                fx = call_func(x)
                if fx < best_f_so_far:
                    best_f_so_far = fx
                continue
            
            # Update CMA-ES state
            # Weighted mean of selected offspring
            m_new = np.zeros(dim)
            for i in range(min(mu, len(offspring))):
                m_new += w[i] * offspring[i][1]
            
            # Update evolution path for covariance
            if np.linalg.norm(m_new - x) > 1e-12:
                pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mu_eff) * (m_new - x) / sigma
            
            # Update evolution path for step-size control
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * C_sqrt.T @ ((m_new - x) / sigma)
            
            # Update covariance matrix
            delta_hs = (1 + cs / (2 * cc)) * (np.linalg.norm(ps) ** 2 / dim - 1)
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - (1 - cs) ** 2) * np.eye(dim))
            for i in range(min(mu, len(offspring))):
                y_i = (offspring[i][1] - x) / sigma
                C += cmu * w[i] * np.outer(y_i, y_i)
            
            # Ensure C is positive definite
            C = (C + C.T) / 2
            try:
                C_sqrt, _ = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
            except np.linalg.LinAlgError:
                # Fallback: recompute from eigen decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            
            # Update sigma
            target_sigma = sigma
            sigma *= np.exp((cs / (dim + 1)) * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Update mean
            x = m_new
        
        # Phase 3: Fine-tuning near the best point (remaining budget)
        if calls[0] < budget:
            sigma = 0.1
            for i in range(budget - calls[0]):
                if calls[0] >= budget or time.time() > deadline:
                    break
                step = sigma * rng.standard_normal(dim)
                y = np.clip(x + step, -5, 5)
                fy = call_func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                else:
                    sigma *= 0.9
                if sigma < 0.001:
                    break
        
        return x, fx
# EVOLVE-BLOCK-END
