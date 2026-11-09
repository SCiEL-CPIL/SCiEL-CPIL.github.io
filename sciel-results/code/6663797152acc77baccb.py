"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-restart CMA-ES for Rastrigin f15."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')

        # Multi-restart strategy
        # With 500 evals, we can afford 3 restarts of ~166 evals each.
        # Or 2 restarts of 250 evals.
        # CMA-ES needs some generations to adapt. 
        # Let's try 3 restarts.
        n_restarts = 3
        evals_per_restart = budget // n_restarts

        for r in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Diverse initialization
            # Restart 0: Origin (often good for Rastrigin if shift is small, but shift is hidden)
            # Better: Random points in the box [-4, 4]
            if r == 0:
                mean = np.zeros(dim)
            else:
                mean = rng.uniform(-4, 4, dim)
            
            sigma = 2.0 # Initial step size, large enough to cover the box
            
            # CMA-ES parameters
            lam = 10
            mu = lam // 2
            
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)
            
            # Rates
            cs = (mu_eff + 2) / (dim + mu_eff + 5)
            cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c1 = 2 / ((dim + 1.3)**2 + mu_eff)
            cmu = 2 / ((dim + 2)**2 + mu_eff) - c1
            
            chiN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            
            # State
            C = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros((dim, dim))
            
            evals_used = 0
            max_evals = evals_per_restart - (r * 0) # Adjust if we want to use remaining budget
            
            while evals_used < max_evals:
                if time.time() > deadline:
                    break
                
                # Generate population
                pop = np.zeros((lam, dim))
                fit = np.zeros(lam)
                
                # Cholesky decomposition
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    L = np.eye(dim)
                
                for i in range(lam):
                    if evals_used >= max_evals or time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    y = L @ z
                    xi = np.clip(mean + sigma * y, -5, 5)
                    pop[i] = xi
                    fit[i] = func(xi)
                    evals_used += 1
                    
                    if fit[i] < best_f:
                        best_f = fit[i]
                        best_x = xi.copy()
                
                if evals_used < lam:
                    break # Not enough budget to complete a generation
                
                # Sort
                order = np.argsort(fit)
                fit_sorted = fit[order]
                pop_sorted = pop[order]
                
                # Update mean
                mean_old = mean.copy()
                mean = np.sum(pop_sorted[:mu] * weights[:, None], axis=0)
                
                # Update evolution paths
                diff = mean - mean_old
                inv_sigma = 1.0 / sigma
                ps = (1 - cs) * ps + cs * inv_sigma * (L.T @ diff)
                
                pc = (1 - cc) * pc + cc * cs / (1 - cs) * (
                    (1 + cs) * inv_sigma * diff - ps
                )
                
                # Update covariance
                # Rank-one
                C += c1 * (np.outer(diff, diff) - C)
                
                # Rank-mu
                diff_mu = pop_sorted[:mu] - mean_old
                rank_mu = np.zeros((dim, dim))
                for k in range(mu):
                    rank_mu += weights[k] * np.outer(diff_mu[k], diff_mu[k])
                C += cmu * rank_mu
                
                # Correction for pc
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (evals_used // lam + 1)))) < (1.4 + 2/(dim+1)) * chiN
                if hsig:
                    C += (c1 * (hsig - 1 + cc) + cmu * (hsig - 1)) * (cc / (1 - cc) * C + pc)
                
                # Symmetrize
                C = (C + C.T) / 2
                
                # Update sigma
                sigma *= np.exp(cs / chiN * (np.linalg.norm(ps) / chiN - 1))
                
                # Bounds for sigma
                sigma = np.clip(sigma, 1e-5, 10.0)
                
                # Regularize C periodically
                if evals_used % (lam * 5) < lam:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-8)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T

        return best_x, best_f
# EVOLVE-BLOCK-END
