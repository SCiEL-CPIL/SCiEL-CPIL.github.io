"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hybrid Grid Search + CMA-ES for BBOB-style Schwefel f20."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # Safety margin for 300s limit
        
        # CMA-ES Parameters
        # Standard lambda for dim=5 is 4 + floor(3 * ln(5)) = 4 + 4 = 8?
        # Formula: lambda = 4 + floor(3 * ln(N)) -> 4 + 4 = 8.
        # Some use 4 * (4 + floor(3 * ln(N))). Let's use standard small lambda for speed.
        # Actually, for dim=5, lambda=8 is quite small. Let's try lambda=10 or 12 for better diversity?
        # With 500 calls, even lambda=8 allows ~60 generations.
        lambda_ = 12
        
        # Phase 1: Coarse Global Search (Grid/Random)
        # Allocate ~15% of budget to find a good starting point.
        # We want to find the global minimum of the multimodal Schwefel.
        # The minimum is in [-4, 4]^5.
        # We will sample random points to identify the best basin.
        
        n_grid = 40  # 40 calls for initial search
        if n_grid >= budget:
             n_grid = budget // 2
             
        rng = np.random.default_rng(42)
        
        # Sample points in [-4, 4] to ensure we are in the "interesting" region
        # Uniform in [-5, 5] is also fine, but [-4, 4] is safer per contract.
        candidates = rng.uniform(-4, 4, (n_grid, dim))
        
        best_x = None
        best_f = float('inf')
        
        calls_made = 0
        
        for i in range(n_grid):
            if time.time() > deadline:
                break
            x = candidates[i]
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        if best_x is None:
            # Fallback if time ran out immediately or budget was tiny
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1
            return best_x, best_f

        # Phase 2: Local Optimization with CMA-ES
        # Start CMA-ES from the best point found.
        # Initial sigma: The distance between grid points is roughly (8 / n_grid^(1/5)) ~ 8/2 = 4?
        # Actually, with 40 points in 5D, volume per point is 10^5/40 ~ 25000, side ~ 7.5.
        # But we are in [-4,4], volume 32768. Side ~ 5.
        # A sigma of 1.0 or 1.5 is reasonable to start.
        
        x = best_x.copy()
        sigma = 1.0
        C = np.eye(dim)
        # Inverse square root of C (identity at start)
        C_inv_sqrt = np.eye(dim)
        
        # CMA-ES Weights
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / dim) / ((dim + 2)**2 + mu_eff))
        
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # State
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        h_sigma = 0.0
        
        # Budget for CMA-ES
        remaining_budget = budget - calls_made
        
        # We will run CMA-ES until budget or time runs out.
        # Each generation costs lambda_ calls.
        
        while calls_made < budget:
            if time.time() > deadline:
                break
                
            # Generate population
            # Transform: x + sigma * B * z
            # B = C_inv_sqrt
            
            Z = rng.standard_normal((lambda_, dim))
            # Rotate and scale
            # Y = C_inv_sqrt @ Z.T -> (dim, lambda_)
            Y = (C_inv_sqrt @ Z.T).T
            X = np.array([x + sigma * y for y in Y])
            
            # Clip to domain
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.array([func(xi) for xi in X])
            calls_made += lambda_
            
            # Check for NaN/Inf just in case, though func is robust
            # If any fail, they should be large.
            
            # Sort
            order = np.argsort(F)
            X = X[order]
            F = F[order]
            
            # Update best
            if F[0] < best_f:
                best_f = F[0]
                best_x = X[0].copy()
                
            # Weighted mean
            m_new = np.sum(weights[:, np.newaxis] * X[:mu], axis=0)
            
            # Update sigma
            # p_sigma update
            h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - 2**-age)/ (1 + 2**-age)) < (1.4 + 2/(dim+1)) * chi_n
            # age is roughly 1/c_sigma, but we need to track it or approximate.
            # Simplified h_sigma check:
            # The condition is ||p_sigma|| / sqrt(1 - (1-2^-age)/(1+2^-age)) < (1.4 + 2/(dim+1)) * chi_n
            # We can approximate the denominator as 1 if age is large, or just use the norm.
            # For robustness in this short run, let's use the standard update but be careful.
            
            # Actually, let's simplify the CMA-ES update to avoid bugs in the complex h_sigma logic 
            # which is hard to get right without a full state tracker for 'age'.
            # We can use a simpler adaptive step size if CMA-ES becomes unstable, 
            # but let's try to stick to CMA-ES.
            
            # Standard CMA-ES update:
            # 1. p_sigma = (1-c_sigma)*p_sigma + h_sigma * sqrt(mu_eff)*c_sigma/sigma * (B^T (m_new - x))
            #    Note: B^T is C_inv_sqrt.T
            #    (m_new - x) is in x-space. We need to map it to y-space?
            #    Actually, the update formula is:
            #    p_sigma <- (1-c_sigma)p_sigma + h_sigma * sqrt(mu_eff) * c_sigma * (B^T (m_new - x))
            #    where B = C_inv_sqrt.
            
            # Let's implement the standard update carefully.
            
            # Calculate B^T (m_new - x)
            # B = C_inv_sqrt
            # diff = m_new - x
            # diff_y = B.T @ diff
            
            diff = m_new - x
            diff_y = C_inv_sqrt.T @ diff
            
            # h_sigma condition
            # We need the 'age' of the p_sigma vector. 
            # In a short run, we can approximate. 
            # Let's use the simplified condition: if ||p_sigma|| < 1.4*chi_n
            # This is a common simplification for short runs.
            if np.linalg.norm(p_sigma) < 1.4 * chi_n:
                h_sigma = 1.0
            else:
                h_sigma = 0.0
                
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(mu_eff) * c_sigma * diff_y
            
            # Update sigma
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            # Wait, the formula is sigma *= exp( c_sigma/c_sigma * ( ||p_sigma||/chi_n - 1 ) )
            # Usually c_sigma is the step size for p_sigma.
            # The update is: sigma <- sigma * exp( (c_sigma/c_sigma) * (||p_sigma||/chi_n - 1) )
            # This simplifies to exp(||p_sigma||/chi_n - 1).
            # Let's use the standard:
            sigma *= np.exp((c_sigma / c_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1.0))
            
            # 2. p_c update
            h_c = np.linalg.norm(p_c) / np.sqrt(1 - (1 - 2**-age_c)/(1 + 2**-age_c)) < (1.4 + 2/(dim+1)) * chi_n
            # Again, simplifying h_c
            if np.linalg.norm(p_c) < 1.4 * chi_n:
                h_c = 1.0
            else:
                h_c = 0.0
                
            p_c = (1 - c_c) * p_c + h_c * np.sqrt(mu_eff) * c_c * diff_y
            
            # 3. Covariance update
            # C <- (1 - c_1 - c_mu + mu_delta) C + c_1 (p_c p_c^T + (1-h_c) c_c (1-c_c) C) + c_mu sum w_i (y_i y_i^T)
            # where y_i = B^T (x_i - m_old)
            
            # Calculate Y matrix (lambda_ x dim) of transformed differences
            # Y = (m_new - x) is not right. We need (x_i - m_old) transformed.
            # The formula uses (x_i - m_old) where m_old is the mean BEFORE update.
            # So we use the X from this generation.
            
            # Y_i = B^T (x_i - x_old)
            # We have X (lambda_ x dim) and x (old mean).
            # Diffs = X - x
            # Y = (C_inv_sqrt.T @ Diffs.T).T
            
            Diffs = X - x
            Y = (C_inv_sqrt.T @ Diffs.T).T
            
            # Rank-1 update
            # Term 1: (1 - c_1 - c_mu + mu_delta) * C
            # mu_delta = c_mu * (1 - mu_eff) / (mu_eff - 1) if mu_eff > 1 else 0
            # Actually mu_delta is usually defined to ensure C remains positive definite.
            # mu_delta = c_mu * (1 - mu_eff) / (mu_eff - 1) is not standard.
            # Standard: mu_delta = c_mu * (1 - mu_eff) / (mu_eff - 1) ? No.
            # Let's use mu_delta = 0 for simplicity in short runs, or standard formula.
            # Standard: mu_delta = c_mu * (1 - mu_eff) / (mu_eff - 1) is wrong.
            # It is: mu_delta = c_mu * (1 - mu_eff) / (mu_eff - 1) ... wait.
            # Hansen's paper: mu_delta = c_mu * (1 - mu_eff) / (mu_eff - 1) is not it.
            # mu_delta = c_mu * (1 - mu_eff) / (mu_eff - 1) ... 
            # Actually, mu_delta = c_mu * (1 - mu_eff) / (mu_eff - 1) is dimensionally weird.
            # Let's just use mu_delta = 0. It's safer for stability in short runs.
            mu_delta = 0.0
            
            C = (1 - c_1 - c_mu + mu_delta) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_c) * c_c * (1 - c_c) * C)
            
            # Rank-mu update
            # Sum of w_i * (y_i y_i^T)
            # Y is lambda_ x dim. We only use top mu.
            Y_mu = Y[:mu]
            # Weighted sum
            # C_mu = sum(weights[i] * outer(Y_mu[i], Y_mu[i]))
            # This can be done as: Y_mu.T @ (weights * Y_mu)
            # Y_mu.T (dim x mu) @ (weights (mu) * Y_mu (mu x dim)) -> (dim x dim)
            C_mu = Y_mu.T @ (weights * Y_mu)
            
            C += c_mu * C_mu
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update C_inv_sqrt
            # Eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip small negative eigenvalues
            eigvals = np.clip(eigvals, 1e-12, None)
            C_inv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
            
            # Update mean
            x = m_new
            
        return best_x, best_f

# EVOLVE-BLOCK-END
