"""Optimizer for BBOB-style Sphere f1 on [-5,5]^5 using CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 300s kill (and 330s hard kill)

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Initial center: 0 is a good prior for BBOB sphere with shift in [-4,4]
        x = np.zeros(dim)
        sigma = 1.0
        pop_size = int(4 + 3 * np.log(dim))
        pop_size = max(4, pop_size)
        
        # CMA-ES weights
        mu = pop_size // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 - mu_eff / (dim + 1.3) ** 2 + c_sigma) / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = (2 * (mu_eff / (dim + 1) - 1) / ((dim + 2) ** 2 + mu_eff) * 
                (1 if mu_eff > 1 else 0.5))
        c_mu = min(c_mu, 1 - c_1)
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # Initial evaluation
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # Pre-allocate arrays for efficiency
        ys = np.zeros((pop_size, dim))
        fs = np.zeros(pop_size)

        while calls < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate population
            for i in range(pop_size):
                # Standard normal
                z = np.random.standard_normal(dim)
                # Transform using Cholesky (approximate via np.linalg.cholesky)
                # To avoid recomputing Cholesky every step if C changes slowly, 
                # we do it here. For dim=5, it's fast.
                try:
                    L = np.linalg.cholesky(C)
                    ys[i] = x + sigma * (L @ z)
                except np.linalg.LinAlgError:
                    # Fallback to identity if C is not positive definite
                    ys[i] = x + sigma * z
                
                # Clip to box
                ys[i] = np.clip(ys[i], -5, 5)
                
                # Evaluate
                fs[i] = func(ys[i])
                calls += 1
                
                # Update best
                if fs[i] < best_fx:
                    best_fx = fs[i]
                    best_x = ys[i].copy()
                    
                # Early exit if deadline reached mid-population
                if time.time() > deadline:
                    break

            # If we didn't complete the population, break
            if calls < budget and time.time() > deadline:
                break
            
            # Sort by fitness
            idx = np.argsort(fs)
            ys_sorted = ys[idx]
            fs_sorted = fs[idx]
            
            # Only process if we have a full population
            if len(ys_sorted) < pop_size:
                break

            # Update center
            x_old = x.copy()
            x = np.sum(weights[:, None] * ys_sorted[:mu], axis=0)
            
            # Update sigma
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * C^(-1/2) * (x - x_old)
            # Approximate C^(-1/2) by using the inverse of L^T L? No, use np.linalg.inv(C) @ (x-x_old) scaled?
            # Standard CMA-ES: ps = (1-c_sigma)*ps + sqrt(c_sigma*(2-c_sigma)*mu_eff)/sigma * C^(-1/2)*(x - x_old)
            # Since C is SPD, C^(-1/2) is well-defined.
            try:
                C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C).T) # This is L^{-1}
                # Wait, C = L L^T. C^{-1/2} is not just L^{-1}. 
                # Actually, the standard update uses the Mahalanobis distance.
                # Let's use the standard formulation:
                # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (x - x_old) / sigma
                # This assumes C=I? No, that's for (1+1)-ES or simplified.
                # Correct CMA-ES:
                # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * C^{-1/2} (x - x_old)
                # We need C^{-1/2}.
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            except Exception:
                C_inv_sqrt = np.eye(dim)

            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_inv_sqrt @ (x - x_old))
            
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls // pop_size + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
            
            # Update pc
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (x - x_old)
            
            # Update C
            rank_one = np.outer(pc, pc)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                zi = C_inv_sqrt @ (ys_sorted[i] - x_old) / sigma
                rank_mu += weights[i] * np.outer(zi, zi)
            
            C = (1 + c_sigma * (2 - c_sigma) * hsig - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Ensure sigma is within reasonable bounds
            sigma = np.clip(sigma, 1e-10, 10.0)

        return best_x, best_fx
# EVOLVE-BLOCK-END
