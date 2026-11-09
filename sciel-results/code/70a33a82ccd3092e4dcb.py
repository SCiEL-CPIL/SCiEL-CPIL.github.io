"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline or calls >= budget:
                return np.inf
            x_clipped = np.clip(x, -5, 5)
            calls += 1
            f_val = func(x_clipped)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # CMA-ES Parameters
        # Population size lambda
        # Standard recommendation: lambda = 4 + floor(3 * ln(dim))
        # For dim=5: 4 + floor(3*1.6) = 4 + 4 = 8. Let's use 10 for better exploration in 500 calls.
        lam = 10
        
        # Initial mean
        x_mean = np.zeros(dim)
        
        # Initial sigma (step size)
        # The box is [-5, 5]. The minimum is in [-4, 4].
        # A sigma of 1.0 covers a reasonable range.
        sigma = 1.0
        
        # Covariance matrix C
        C = np.eye(dim)
        
        # Inverse square root of C (for sampling)
        B = np.eye(dim)
        D = np.ones(dim)
        D_inv = np.ones(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Parameters
        c_sigma = (lam + 2) / (dim + 4)
        mu_sigma = 0.3 * (lam + 2)
        c_c = (4 + lam / dim) / (dim + 4 + 2 * lam / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + lam)
        c_mu = min(1 - c_1, 2 * ((mu_sigma + 2) / (mu_sigma + lam - 2)) / ((dim + 2) ** 2 + lam))
        
        # Weights for selection
        mu = lam // 2  # Number of parents
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1 / np.sum(w ** 2)
        
        # Initial population evaluation
        # We will run generations until budget is exhausted
        # Each generation consumes 'lam' evaluations
        
        while calls + lam <= budget:
            if time.time() > deadline:
                break
                
            # Generate offspring
            ys = []
            fs = []
            for i in range(lam):
                # Sample from N(x_mean, sigma^2 * C)
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (B @ (D * z))
                fy = evaluate(y)
                ys.append(y)
                fs.append(fy)
                
            # Sort by fitness
            order = np.argsort(fs)
            ys_sorted = [ys[i] for i in order]
            fs_sorted = [fs[i] for i in order]
            
            # Update mean
            new_x_mean = np.sum([w[i] * ys_sorted[i] for i in range(mu)], axis=0)
            
            # Update evolution paths
            # p_sigma update
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B @ (D_inv * ((new_x_mean - x_mean) / sigma)))
            
            # h_sigma check
            norm_p_sigma = np.linalg.norm(p_sigma)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            h_sigma = 1 if (norm_p_sigma / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls + 1) / lam))) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            
            # p_c update
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (new_x_mean - x_mean) / sigma
            
            # Covariance matrix update
            v = p_c / sigma
            C = (1 + c_1 * (h_sigma * (2 - c_c) - c_c) - c_1) * C + c_1 * np.outer(v, v)
            
            # Rank-mu update
            y_w = np.array([(ys_sorted[i] - x_mean) / sigma for i in range(mu)])
            C += c_mu * (np.outer(y_w.T, y_w * w[:, np.newaxis]).sum(axis=2) + 2 * c_1 * (h_sigma * (2 - c_c) - c_c) / c_mu * np.outer(p_c, p_c) / (sigma ** 2))
            
            # Symmetrize and ensure positive definite
            C = (C + C.T) / 2
            try:
                B, D = np.linalg.eigh(C)
                D = np.sqrt(D)
                D_inv = 1 / D
            except:
                # Reset if numerical issues
                C = np.eye(dim)
                B = np.eye(dim)
                D = np.ones(dim)
                D_inv = np.ones(dim)
                
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update mean
            x_mean = new_x_mean
            
            # Check for convergence or stagnation
            if np.linalg.norm(sigma * D) < 1e-10:
                break
                
            # Optional: Restart if stuck? 
            # For 500 calls in 5D, CMA-ES usually handles it well.
            # If we detect very slow progress, we might restart, but let's trust CMA-ES first.
            
        # Use remaining budget for local refinement if any
        while calls < budget:
            if time.time() > deadline:
                break
            # Simple local search around best_x
            y = best_x + 0.1 * rng.standard_normal(dim)
            fy = evaluate(y)
            if fy < best_f:
                x_mean = best_x.copy() # Update mean for potential further refinement if we were to continue, but we are just doing local search
                # Actually, let's just keep the best_x
                # The evaluate function updates best_x automatically
            else:
                # If no improvement, maybe step back?
                # For simplicity, just keep trying small random steps
                pass
                
        return best_x, best_f
# EVOLVE-BLOCK-END
