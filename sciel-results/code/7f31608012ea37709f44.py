"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for 5-dimensional black-box functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1 / np.sum(weights ** 2)
        # Step size control
        cs = (mueff + 2) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        # Covariance matrix adaptation
        cmacov = 1 / (dim + 2) * (2 / (1 + cs) + min(1, (mueff - 1 + np.sum(weights[:mu] ** 2)) / (dim + 1)))
        # Initial parameters
        x0 = np.zeros(dim)  # Start at origin, or random? Origin is often good for BBOB. Let's use random uniform to be safe against shifts.
        # Actually, BBOB functions have shifted optima. A random start is better.
        rng = np.random.default_rng(0)
        x0 = rng.uniform(-5, 5, dim)
        
        # We need to evaluate x0 first
        fx0 = func(x0)
        best_x = x0.copy()
        best_f = fx0
        num_calls = 1

        # CMA-ES state
        sigma = 2.0  # Initial step size
        C = np.eye(dim)
        Csqrt = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        def update_CMA_Covariance(pcs, C, Csqrt):
            # This is a simplified update, usually involves eigendecomposition
            # For performance in this context, we might use a simplified rank-mu update
            pass

        # To keep it simple and fast, we use a standard CMA-ES loop
        # We will implement the core loop manually
        
        # Precompute inverse square root of C for step size adaptation
        # Initially C=I, so C^-1/2 = I
        
        while num_calls < budget:
            if time.time() > deadline:
                break
                
            # 1. Sample points
            # Generate random numbers
            # z ~ N(0, I)
            z = rng.standard_normal((lam, dim))
            
            # Transform to x
            # x = x0 + sigma * Csqrt * z
            # Since C starts as I, Csqrt is I. As C changes, we need to update Csqrt.
            # For efficiency, we can use the fact that C is updated via rank updates.
            # However, a full CMA-ES implementation is complex. 
            # Given dim=5, we can afford to do SVD or Cholesky occasionally.
            
            # Let's use a simpler adaptive method if CMA-ES is too heavy, 
            # but CMA-ES is the right tool.
            # I will implement a standard CMA-ES.
            
            # Update Csqrt? 
            # Standard CMA-ES updates C and then computes C^-1/2.
            # For dim=5, this is cheap.
            
            # Sample
            # We need C^-1/2. 
            # Let's maintain C and compute its inverse square root every generation.
            # eigvals, eigvecs = np.linalg.eigh(C)
            # C_inv_sqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
            
            # For the first few iterations, C=I.
            # Let's just run the loop.
            
            # To save code length and complexity, I will use a simplified 
            # "Adaptive Nelder-Mead" or "Simplex" if CMA-ES is too long?
            # No, CMA-ES is better. I will write a compact CMA-ES.
            
            # Actually, let's look at the budget. 500 calls.
            # lam=10. 50 generations.
            # SVD of 5x5 matrix is very fast.
            
            # 1. Generate population
            # We need C_inv_sqrt. 
            # Let's compute it.
            eigvals, eigvecs = np.linalg.eigh(C)
            # Handle numerical issues
            eigvals = np.clip(eigvals, 1e-12, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            # Sample z
            z = rng.standard_normal((lam, dim))
            # x = x0 + sigma * C_inv_sqrt @ z.T
            pop = x0 + sigma * (C_inv_sqrt @ z.T).T
            
            # Evaluate population
            fvals = np.zeros(lam)
            for i in range(lam):
                if num_calls >= budget:
                    break
                # Clip to box
                xi = np.clip(pop[i], -5, 5)
                fvals[i] = func(xi)
                num_calls += 1
                
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = xi.copy()
            
            if num_calls >= budget:
                break
                
            # Sort by fitness
            indices = np.argsort(fvals)
            y = pop[indices]
            fsorted = fvals[indices]
            
            # Mean of best mu
            x_new = np.sum(weights[:, None] * y[:mu], axis=0)
            
            # Step size adaptation
            # ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * C_inv_sqrt @ (x_new - x0) / sigma
            # Note: C_inv_sqrt is for the current C. The standard formula uses C^-1/2.
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_inv_sqrt @ (x_new - x0)) / sigma
            
            # Covariance matrix adaptation
            # pc = (1-cs)*pc + sqrt(cs*(2-cs)*mueff) * (x_new - x0) / sigma
            pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mueff) * (x_new - x0) / sigma
            
            # Update C
            # Rank-mu update
            diff = (y[:mu] - x0) / sigma
            # C = (1 + cmacov*(cs*(2-cs) - (1-cs))) * C + cmacov * sum(weights_i * diff_i * diff_i.T)
            # Standard formula:
            # C = (1 - cmacov) * C + cmacov * ( (1/cs) * pc * pc.T + sum(weights_i * diff_i * diff_i.T) )
            # Wait, the standard update is:
            # C = (1 - cmacov) * C + cmacov * ( (1/cs) * pc * pc.T + sum(weights_i * diff_i * diff_i.T) )
            # Actually, the term (1/cs)*pc*pc.T is part of the rank-1 update?
            # No, the rank-1 update is usually done on ps or pc?
            # Standard CMA-ES:
            # 1. Update ps
            # 2. Update pc
            # 3. Update C: C = (1 - cmacov) * C + cmacov * ( (1/cs) * pc * pc.T + sum_{i=1}^mu weights_i * diff_i * diff_i.T )
            # This is correct.
            
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(diff[i], diff[i])
                
            C = (1 - cmacov) * C + cmacov * ( (1.0 / cs) * np.outer(pc, pc) + rank_mu )
            
            # Symmetrize
            C = (C + C.T) / 2.0
            
            # Update x0
            x0 = x_new
            
            # Update sigma
            # chi_n = sqrt(n) * (1 - 1/(4n) + 1/(21n^2))
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim * dim))
            sigma = sigma * np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
            
            # Stagnation check
            if np.linalg.norm(ps) > (1.4 + 2.0 / (dim + 1)) * chi_n:
                # Restart?
                # For simplicity, just continue. The step size adaptation should handle it.
                pass
                
            # Safety: if sigma becomes too small or too large
            sigma = np.clip(sigma, 1e-8, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
