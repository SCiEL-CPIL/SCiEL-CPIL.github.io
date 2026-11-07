"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for BBOB f4 (Büche-Rastrigin) using CMA-ES.
    CMA-ES adapts its covariance matrix to the rotated landscape,
    which is crucial for BBOB functions with hidden rotations.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 330s limit, allow 290s for safety
        deadline = start_time + 290.0
        rng = np.random.default_rng(42)
        
        dim = self.dim
        budget = self.budget
        calls = 0
        
        best_x = None
        best_f = np.inf

        # --- Stage 1: Global Exploration ---
        # Use a structured quasi-random sequence to cover the space.
        n_explore = min(50, budget)
        
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=dim, scramble=True, seed=123)
            next_pow2 = 1
            while next_pow2 < n_explore:
                next_pow2 *= 2
            points = sampler.random(next_pow2)[:n_explore]
        except Exception:
            points = rng.random((n_explore, dim))
            
        points = points * 10.0 - 5.0
        
        for i in range(n_explore):
            if time.time() > deadline or calls >= budget:
                break
            x = points[i]
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            x0 = np.zeros(dim)
            best_f = func(x0)
            best_x = x0.copy()
            calls = 1

        # --- Stage 2: CMA-ES ---
        # Implement a lightweight CMA-ES from scratch to avoid scipy dependency issues
        # and to have full control over the budget and time.
        
        # Parameters
        pop_size = 4 + int(3 * np.log(dim))  # ~10 for dim=5
        pop_size = min(pop_size, 15) # Cap for speed
        
        # Initialize CMA-ES state
        x_mean = best_x.copy()
        sigma = 1.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        cs = np.zeros(dim)
        
        # Learning rates
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (2 + mu_eff / (2 * dim)) / (dim + 2 + mu_eff / (2 * dim))
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (weights[0] - 1 + weights[1]) / ((dim + 2)**2 + mu_eff))
        
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        
        # Precompute for efficiency
        sqrt_eigvals = np.ones(dim)
        sqrt_eigvecs = np.eye(dim)
        
        # Helper to update sqrt decomposition
        def update_sqrt_eig(C, threshold=1e-16):
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, threshold, None)
            sqrt_eigvals = np.sqrt(eigvals)
            return sqrt_eigvals, eigvecs

        sqrt_eigvals, sqrt_eigvecs = update_sqrt_eig(C)
        
        while calls < budget and time.time() < deadline:
            # Generate population
            ys = []
            fs = []
            
            for k in range(pop_size):
                if time.time() > deadline or calls >= budget:
                    break
                # Sample
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (sqrt_eigvecs @ (sqrt_eigvals * z))
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls += 1
                ys.append(y)
                fs.append(fy)
                
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            
            if len(ys) < 2:
                break
            
            # Sort by fitness
            idx = np.argsort(fs)
            ys = np.array(ys)[idx]
            fs = np.array(fs)[idx]
            
            # Update mean
            x_mean_new = np.sum(weights[:, np.newaxis] * ys[:mu], axis=0)
            
            # Update step size (1/5 success rule approximation or standard CMA-ES)
            # Standard CMA-ES uses cs and cc
            y_sigma = (x_mean_new - x_mean) / sigma
            cs = (1 - cs) * cs + cs * np.linalg.norm(y_sigma) / damps
            pc = (1 - cc) * pc + cc * (1 + (1 - cs) * cs * np.linalg.norm(y_sigma) / damps) * y_sigma
            hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - cc)**(2 * (calls - pop_size)))) < (1.4 + 2 / (dim + 1)) * (1 - cc)
            
            sigma *= np.exp(cs / damps * (np.linalg.norm(y_sigma) / np.sqrt(2 * dim) - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # Update covariance
            delta_pc = (1 + hsig * (1 - cc) * cc) * pc - pc
            rank_one = np.outer(delta_pc, delta_pc)
            
            # Rank-mu update
            z_diffs = (ys[:mu] - x_mean) / sigma
            rank_mu = np.sum(weights[:, np.newaxis, np.newaxis] * np.einsum('i,ij,ik->jk', weights, z_diffs, z_diffs), axis=0)
            # Wait, rank_mu formula is sum_k w_k * (y_k - x_mean)/sigma * ((y_k - x_mean)/sigma)^T
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                z = (ys[i] - x_mean) / sigma
                rank_mu += weights[i] * np.outer(z, z)
            
            C = (1 - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Add small regularization if needed
            C += 1e-12 * np.eye(dim)
            
            # Update square root decomposition
            sqrt_eigvals, sqrt_eigvecs = update_sqrt_eig(C)
            
            x_mean = x_mean_new
            
            # Stagnation check: if sigma is very small or no improvement for a long time
            # We can restart from best_x with a larger sigma if stuck
            if sigma < 1e-6:
                # Restart
                x_mean = best_x + rng.standard_normal(dim) * 1.0
                x_mean = np.clip(x_mean, -5, 5)
                sigma = 1.0
                C = np.eye(dim)
                pc = np.zeros(dim)
                cs = np.zeros(dim)
                sqrt_eigvals, sqrt_eigvecs = update_sqrt_eig(C)

        return best_x, best_f
# EVOLVE-BLOCK-END
