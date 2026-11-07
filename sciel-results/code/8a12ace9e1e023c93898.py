"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for BBOB f12 (Bent Cigar) on [-5, 5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280  # 300s limit, 20s safety margin
        rng = np.random.default_rng(42)
        
        # CMA-ES Parameters
        n = self.dim
        # Initial sigma: heuristic for box size
        sigma = 1.0 
        x_best = np.zeros(n)
        f_best = np.inf
        
        # Population size
        lambda_ = 4 + int(3 * np.log(n))
        mu = lambda_ // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step-size adaptation parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n**2))
        
        # Covariance matrix adaptation parameters
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # State
        x_mean = np.zeros(n)
        C = np.eye(n)
        C_sqrt = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Initial evaluation
        x_init = np.zeros(n)
        f_init = func(x_init)
        if f_init < f_best:
            f_best = f_init
            x_best = x_init.copy()
            
        calls_made = 1
        budget_left = self.budget - 1
        
        while budget_left > 0 and time.time() < deadline:
            # Generate population
            pop = []
            f_pop = []
            
            # Determine how many we can evaluate in this step (usually all, but check time)
            # For simplicity and robustness, we evaluate the whole population if time allows,
            # but we must be careful not to exceed the budget.
            # Since n=5, lambda is small (13). 500 budget / 13 ~ 38 generations.
            
            for i in range(lambda_):
                if budget_left <= 0 or time.time() > deadline:
                    break
                z = rng.standard_normal(n)
                # Cholesky factorization of C is expensive, but for n=5 it's fine.
                # However, updating C_sqrt every generation is heavy. 
                # Standard CMA-ES updates C, and uses eigendecomposition or Cholesky for sampling.
                # For n=5, Cholesky is fast.
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback if C becomes singular or negative definite (shouldn't happen often with regularization)
                    L = np.linalg.cholesky(C + 1e-12 * np.eye(n))
                
                y = x_mean + sigma * L @ z
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls_made += 1
                budget_left -= 1
                pop.append(y)
                f_pop.append(fy)
                
                if fy < f_best:
                    f_best = fy
                    x_best = y.copy()
            
            if not pop:
                break
            
            # Sort by fitness
            idx = np.argsort(f_pop)
            pop = [pop[i] for i in idx]
            f_pop = [f_pop[i] for i in idx]
            
            # Update mean
            y_w = np.zeros((mu, n))
            for i in range(mu):
                y_w[i] = pop[i]
            
            x_mean_old = x_mean.copy()
            x_mean = np.sum(weights[:, np.newaxis] * y_w, axis=0)
            
            # Update path covariance
            C_inv_sqrt = np.linalg.inv(C_sqrt) # Approximation, actually we need C^(-1/2)
            # Proper update:
            # pc = (1 - c_c) * pc + c_c * sqrt(mu_eff) * C_inv_sqrt @ (x_mean - x_mean_old) / sigma
            try:
                C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C)) # This is C^-1/2 if C = L L^T
                # Wait, cholesky gives L such that C = L L^T. So C^-1/2 = L^-1? No.
                # C = L L^T. C^-1/2 = L^-1? No. 
                # Let C = V D V^T. C^-1/2 = V D^-1/2 V^T.
                # Using Cholesky L: C = L L^T. C^-1/2 = L^-T L^-1? No.
                # Actually, if we have L, we can compute C^-1/2 via eigendecomp or just use C^-1 for pc update?
                # Standard CMA-ES uses C^-1/2.
                # For n=5, eigendecomp is cheap.
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            except:
                C_inv_sqrt = np.eye(n)
                
            pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff) * C_inv_sqrt @ (x_mean - x_mean_old) / sigma
            
            # Update step size
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (x_mean - x_mean_old) / sigma
            norm_ps = np.linalg.norm(ps)
            
            # Hessian correction term
            hsig = 1 if (norm_ps / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made / lambda_)))) < (1.4 + 2 / (n + 1)) * chi_n else 0
            
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 1e10)
            
            # Update covariance
            delta_sig = (1 - hsig) * c_sigma * (2 - c_sigma)
            rank_1 = (1 - c_1) + delta_sig * c_c * (1 - c_c)
            # Actually standard formula:
            # C = (1 - c_1 - c_mu + delta_sig) C + c_1 * pc pc^T + c_mu * sum w_i (y_i - x_mean)(y_i - x_mean)^T
            
            # Correction:
            # C = (1 - c_1 - c_mu + delta_sig) * C + c_1 * np.outer(pc, pc) + c_mu * np.sum(weights[:, np.newaxis] * np.einsum('ij,ik->ijk', y_w - x_mean, y_w - x_mean), axis=0)
            
            # Ensure symmetry and positive semi-definite
            diff = y_w - x_mean
            rank_mu = np.sum(weights[:, np.newaxis, np.newaxis] * diff[:, np.newaxis, :] * diff[:, :, np.newaxis], axis=0)
            
            C = (1 - c_1 - c_mu + delta_sig) * C + c_1 * np.outer(pc, pc) + c_mu * rank_mu
            C = (C + C.T) / 2
            
            # Regularize C
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update C_sqrt (Cholesky) for next iteration
            try:
                C_sqrt = np.linalg.cholesky(C)
            except:
                C_sqrt = np.linalg.cholesky(C + 1e-12 * np.eye(n))

        return x_best, f_best
# EVOLVE-BLOCK-END
