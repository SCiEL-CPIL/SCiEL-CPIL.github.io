"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """Pure CMA-ES for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        rng = np.random.default_rng(123)
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size lambda
        lam = max(4 + int(3 * math.log(dim)), 10)
        # For dim=5, lam = 4 + 3*1.609 = 8.8 -> 8? 
        # Standard formula: lam = 4 + int(3*log(N)). For N=5, log(5)~1.6, 3*1.6=4.8, int=4, 4+4=8.
        # However, for multimodal problems like f21, a larger population is often better to explore.
        # Let's stick to a slightly larger population for better coverage, e.g., 20 or 30.
        # The directive suggests population size 30.
        lam = 30
        
        # Initial mean
        m = np.zeros(dim)
        # Initial sigma
        # The search space is [-5, 5]^5. A reasonable initial step size is 1.0 or 2.0.
        # Given f21 has peaks, a larger initial sigma might help escape local minima,
        # but we need to converge fast. 1.0 is standard for [-5,5] if centered at 0.
        sigma = 1.0
        
        # CMA-ES weights
        # mu = lam / 2
        mu = lam // 2
        # Weights w_i
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Learning rates
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # State variables
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        dcor = 1.0 + 2 * max(0, math.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        
        # Precompute square root of C (initially identity)
        # We will update C and its square root using eigen decomposition
        C_eigvals = np.ones(dim)
        C_eigvecs = np.eye(dim)
        
        # Helper to update C_eigvals and C_eigvecs
        def update_eig():
            nonlocal C, C_eigvals, C_eigvecs
            # Eigendecomposition
            C_eigvals, C_eigvecs = np.linalg.eigh(C)
            # Ensure positive eigenvalues (numerical stability)
            C_eigvals = np.maximum(C_eigvals, 1e-12)

        # Initial evaluation of mean
        calls_made = 0
        best_x = m.copy()
        best_f = func(m)
        calls_made += 1
        
        # List to store population and fitness
        # We need to generate, evaluate, and then select.
        
        while calls_made < budget and time.time() < deadline:
            # 1. Sample population
            # Compute square root of C
            # C = C_eigvecs @ diag(sqrt(C_eigvals)) @ C_eigvecs.T
            # C_sqrt = C_eigvecs @ diag(sqrt(C_eigvals))
            sqrt_C_eigvals = np.sqrt(C_eigvals)
            C_sqrt = C_eigvecs @ np.diag(sqrt_C_eigvals)
            
            # Generate directions
            # z_k ~ N(0, I)
            Z = rng.standard_normal((lam, dim))
            
            # y_k = m + sigma * C_sqrt @ z_k
            Y = m[None, :] + sigma * (C_sqrt @ Z.T).T
            # Clip to box [-5, 5]
            Y = np.clip(Y, -5.0, 5.0)
            
            # Evaluate population
            F = np.zeros(lam)
            for i in range(lam):
                if calls_made >= budget or time.time() > deadline:
                    break
                F[i] = func(Y[i])
                calls_made += 1
            
            if calls_made >= budget or time.time() > deadline:
                break
                
            # Sort by fitness
            idx = np.argsort(F)
            Y_sorted = Y[idx]
            F_sorted = F[idx]
            
            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = Y_sorted[0].copy()
                
            # 2. Update mean
            # m_new = sum(w_i * y_{i:lambda})
            m_old = m.copy()
            m = np.sum(w[:, None] * Y_sorted[:mu], axis=0)
            
            # 3. Update sigma, C, pc, ps
            # y = (m - m_old) / sigma
            y = (m - m_old) / sigma
            
            # Update ps
            hs = (math.sqrt((1 - math.exp(-2 * (cs + 1) * 1.0)) / (dim * (dim + 1))) * 
                  (dim + 2) / 2.0 + cs)
            # Note: The standard formula for hs involves the norm of C_sqrt.
            # hs = (sqrt(1 - exp(-2 * (cs + 1))) * (dim + 2) / 2 + cs) / ||C_sqrt||_F?
            # Actually, the standard CMA-ES formula is:
            # hs = (sqrt(1 - exp(-2 * (cs + 1))) * (dim + 2) / 2 + cs) * (norm(C_sqrt) / norm(I))?
            # Let's use the standard implementation from Hansen's code:
            # hs = (sqrt(1 - exp(-2 * (cs + 1))) * (dim + 2) / 2 + cs)
            # This assumes C is close to identity. A more robust version uses the norm of the transformation.
            # For simplicity and stability, let's use the standard heuristic:
            hs = (math.sqrt(1 - math.exp(-2 * (cs + 1))) * (dim + 2) / 2 + cs)
            
            if np.linalg.norm(y) < (1 + 5.0 / (dim + 1)) * hs * math.sqrt(dim) * sigma:
                ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mu_eff) * y
            else:
                # If the step is too large, we might reset ps or just not update?
                # Standard: ps = (1-cs)*ps (no addition)
                pass
            
            # Update pc
            if np.linalg.norm(ps) / math.sqrt(1 - (1 - cs)**(2 * (calls_made // lam + 1))) < (1 + 5.0 / (dim + 1)) * hs:
                pc = (1 - cc) * pc + math.sqrt(cc * (2 - cc) * mu_eff) * ps / sigma
            # else: pc = (1-cc)*pc
            
            # Update C
            # Delta_h = (1 + 2 * max(0, sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs) * (||ps||^2 / dim - 1)
            norm_ps_sq = np.dot(ps, ps)
            delta_h = (1 + 2 * max(0, math.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs) * (norm_ps_sq / dim - 1)
            
            # Rank-one update
            rank1 = np.outer(y, y)
            # Rank-mu update
            # Sum of top mu directions
            # y_i = (m - m_old) / sigma is the mean step.
            # The rank-mu update uses the steps of the selected individuals.
            # y_i = (Y_sorted[i] - m_old) / sigma
            Y_selected = Y_sorted[:mu]
            Y_steps = (Y_selected - m_old[None, :]) / sigma
            # Weighted sum of outer products
            rank_mu = np.sum(w[:, None, None] * np.einsum('ij,ik->ijk', Y_steps, Y_steps), axis=0)
            
            C = (1 + c1 * delta_h - c1 - cmu) * C + c1 * rank1 + cmu * rank_mu
            
            # Update sigma
            sigma = sigma * math.exp(cs / dcor * (np.linalg.norm(ps) / math.sqrt(dim) - 1))
            
            # Numerical stabilization: ensure C is symmetric and positive definite
            C = (C + C.T) / 2.0
            update_eig()
            
            # Check for convergence/stagnation
            # If sigma is too small or steps are too small
            if sigma < 1e-10:
                break
            
            # Optional: Restart if population is too clustered
            # For f21, we might want to restart if no progress for a while.
            # But CMA-ES is generally robust.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
