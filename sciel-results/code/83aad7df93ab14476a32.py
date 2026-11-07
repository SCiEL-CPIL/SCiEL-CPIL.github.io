"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB f13 (Sharp Ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Deadline guard: 330s limit, leave 10s margin
        start_time = time.time()
        deadline = start_time + 320.0

        rng = np.random.default_rng(42)

        # CMA-ES Parameters
        # Standard population size for dim=5
        # lambda = 4 + floor(3 * log(mu)) is common, but let's use fixed standard values
        # For dim 5, lambda=9 is a good standard choice
        lam = 4 + int(3 * np.log(dim + 1)) 
        mu = lam // 2
        
        # Weights for the selected individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        
        # Initial step size
        # f13 is on [-5, 5]^5. The minimum is shifted. 
        # A reasonable initial sigma is 2.0 or 1.0. 
        # Since we don't know the scale of the function values directly relative to distance, 
        # but the box is 10 units wide, sigma=1.0 is a safe start.
        sigma = 1.0
        
        # Initial mean: center of the box? Or random?
        # The problem says minimizer lies in [-4, 4]^5. Center is 0.
        # Let's start at 0.
        mean = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Cholesky factor of C
        Csqrt = np.eye(dim)
        Csqrtinv = np.eye(dim)
        
        # Track best
        best_x = mean.copy()
        best_f = func(best_x)
        calls_used = 1
        
        # Helper to evaluate
        def eval_point(x):
            nonlocal best_x, best_f
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Main loop
        while calls_used < budget:
            if time.time() > deadline:
                break
            
            # 1. Sample population
            # Update Cholesky if needed? In standard CMA-ES, C is updated every generation.
            # For efficiency, we can update Csqrt less frequently or just use it.
            # Since dim is small (5), computing Cholesky every generation is cheap.
            try:
                Csqrt, _ = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, regularize
                C += 1e-6 * np.eye(dim)
                Csqrt, _ = np.linalg.cholesky(C)
            
            # Generate candidates
            ys = []
            for i in range(lam):
                if calls_used >= budget:
                    break
                z = rng.standard_normal(dim)
                y = mean + sigma * (Csqrt @ z)
                # Clip to box
                y = np.clip(y, -5, 5)
                fy = eval_point(y)
                calls_used += 1
                ys.append((fy, y))
            
            if len(ys) < mu:
                break

            # 2. Sort by fitness
            ys.sort(key=lambda t: t[0])
            
            # 3. Update mean
            old_mean = mean.copy()
            mean = np.sum([weights[i] * ys[i][1] for i in range(mu)], axis=0)
            
            # 4. Update covariance matrix
            # Calculate difference vectors
            diff_mean = mean - old_mean
            
            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (Csqrtinv @ diff_mean)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_used // lam)))) < (1.4 + 2 / (dim + 1)) * cc
            
            pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mueff) * diff_mean / sigma
            
            # Update C
            delta = (1 - hsig) * (1 - cc) * cc
            rank_one = np.outer(pc, pc)
            
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                y_diff = (ys[i][1] - old_mean) / sigma
                rank_mu += weights[i] * np.outer(y_diff, y_diff)
            
            C = (1 + c1 * delta - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu
            
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            # Simple regularization if needed
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-12:
                C += 1e-12 * np.eye(dim)
            
            # 5. Update sigma
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            sigma = sigma * np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Clamp sigma
            sigma = np.clip(sigma, 1e-12, 10.0)

        return best_x, best_f

# EVOLVE-BLOCK-END
