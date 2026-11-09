"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """CMA-ES optimizer on the box [-5, 5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        sigma = 0.5 
        m = np.zeros(n)
        C = np.eye(n)
        
        def get_sqrt_C(C):
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

        sqrt_C = get_sqrt_C(C)
        
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, (mueff - 1) / (n + 1)) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        best_x = m.copy()
        best_f = func(m)
        count = 1
        
        while count < budget:
            remaining = budget - count
            current_lam = min(lam, remaining)
            
            if current_lam == 0:
                break
                
            z = np.random.randn(current_lam, n)
            pop = m[None, :] + sigma * (z @ sqrt_C.T)
            
            fvals = np.array([func(x) for x in pop])
            count += current_lam
            
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_f:
                best_f = fvals[min_idx]
                best_x = pop[min_idx].copy()
                
            if count >= budget:
                break
                
            # Update CMA-ES state only if we have a full population
            if current_lam == lam:
                idx = np.argsort(fvals)[:mu]
                y_sorted = [pop[i] for i in idx]
                
                new_m = np.sum(weights[:, None] * np.array(y_sorted), axis=0)
                
                y_w = (new_m - m) / (sigma * chi_n)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_w
                
                # Approximate h_sig
                decay = 1 - (1 - c_sigma)**(2 * (budget - count))
                norm_p_sigma = np.linalg.norm(p_sigma)
                h_sig = 1 if (norm_p_sigma / np.sqrt(decay) < (1.4 + 2 / (n + 1)) * chi_n) else 0
                
                p_c = (1 - c_c) * p_c + h_sig * np.sqrt(c_c * (2 - c_c) * mueff) * y_w
                m = new_m
                
                C = (1 - c_1 - c_mu) * C
                C += c_1 * (np.outer(p_c, p_c) + (1 - h_sig) * c_c * (2 - c_c) * C)
                
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    rank_mu += weights[i] * np.outer(y_sorted[i] - m, y_sorted[i] - m)
                C += c_mu * rank_mu / (sigma * sigma)
                
                sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
                sqrt_C = get_sqrt_C(C)
            # If partial population, we just skip the state update and finish after this batch
            
        return best_x, best_f
# EVOLVE-BLOCK-END
