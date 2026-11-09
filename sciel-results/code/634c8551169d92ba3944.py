"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB f17."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin for 300s limit
        
        # CMA-ES Parameters
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        
        # Population size
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Strategy parameters
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        
        # Initial state
        # Start near center, but random enough to escape local minima if needed
        # f17 is multimodal. Starting at 0 is often risky if the shift is far.
        # Uniform start in [-5, 5] is safer for global exploration initially.
        x = rng.uniform(-5, 5, dim)
        sigma = 2.0
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        best_x = x
        best_f = func(x)
        calls = 1
        
        # Pre-compute Cholesky decomposition update logic
        # We will use a simplified CMA-ES update to avoid complex linear algebra overhead
        # which might be slow in pure Python/Numpy for small dims if not optimized.
        
        # However, standard CMA-ES is robust. Let's implement a robust version.
        
        while calls < budget:
            # Check time
            if time.time() > deadline - 5.0:
                break
                
            # Generate population
            ys = []
            for i in range(lam):
                # Use current C and sigma
                # Efficient generation: z ~ N(0, I), y = x + sigma * C_sqrt @ z
                z = rng.standard_normal(dim)
                y = x + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                ys.append(y)
            
            # Evaluate
            fs = []
            for y in ys:
                if calls >= budget:
                    break
                fy = func(y)
                fs.append(fy)
                calls += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            
            if len(fs) < lam:
                break
                
            # Sort
            idx = np.argsort(fs)
            ys = [ys[i] for i in idx]
            fs = [fs[i] for i in idx]
            
            # Update mean
            x_old = x.copy()
            x = sum(weights[i] * ys[i] for i in range(mu))
            
            # Update step-size control (CSA)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (C_sqrt.T @ (x - x_old))
            norm_ps = np.linalg.norm(ps)
            
            # Update covariance matrix
            # hsig = (norm_ps / chi_n) < (1.4 + 2 / (dim + 1)) * (1 - cc)
            hsig = (norm_ps / chi_n) < (1.4 + 2 / (dim + 1)) * (1 - cc)
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (x - x_old) / sigma
            
            # Rank-one and rank-mu updates
            # Note: C_sqrt is used for sampling, but updates are on C
            # We need to maintain C and C_sqrt. Updating C_sqrt is tricky.
            # For dim=5, we can recompute Cholesky every step. It's cheap.
            
            delta_sig = 1 + 0.3 * min(1, (norm_ps**2 / chi_n**2 - 1) / lam)
            # Wait, standard formula:
            # sigma *= exp(cs / damps * (norm_ps / chi_n - 1))
            sigma *= np.exp(cs / damps * (norm_ps / chi_n - 1))
            
            # C update
            C = (1 - c1 - cmu) * C
            C += c1 * (np.outer(pc, pc) + (hsig - 1 + cc) * cc * C)
            for i in range(mu):
                yi = (ys[i] - x_old) / sigma
                C += cmu * weights[i] * np.outer(yi, yi)
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # Recompute Cholesky
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If Cholesky fails, add small identity
                C += 1e-9 * np.eye(dim)
                C_sqrt = np.linalg.cholesky(C)
            
            # Stopping criteria
            if sigma < 1e-12:
                break
            if norm_ps > 1e12:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
