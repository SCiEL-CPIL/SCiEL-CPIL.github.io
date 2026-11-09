"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize


class Optimizer:
    """Improved CMA-ES optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        
        # 1. Initial Exploration
        # Spend a small fraction of the budget to find a good starting point
        init_calls = min(10, budget // 10)
        best_x = np.zeros(n)
        best_f = func(best_x)
        calls_made = 1
        
        for _ in range(init_calls - 1):
            x_try = rng.uniform(-5, 5, n)
            f_try = func(x_try)
            if f_try < best_f:
                best_f = f_try
                best_x = x_try.copy()
        
        calls_made += init_calls - 1
        remaining = budget - calls_made

        # 2. CMA-ES Parameters
        # lambda = 4 + floor(3 * ln(n)) is a common heuristic, for n=5, ln(5)~1.6, 3*1.6=4.8, floor=4, lambda=8?
        # Let's use a slightly larger population for better exploration in dim 5.
        lambda_ = 12 
        mu = lambda_ // 2
        
        # Initial step size: large enough to explore the box [-5,5]
        sigma = 2.0
        
        # Mean
        m = best_x.copy()
        
        # Weights
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 1) - np.log(i + 1)
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Adaptation parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # State
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        C = np.eye(n)
        hs = (1 + (1 + 3 / n) * c_sigma) / (1 - c_sigma)
        
        gen = 0
        no_improve = 0
        
        while remaining > 0 and gen < 2000:
            # Generate population
            Y = np.zeros((lambda_, n))
            F = np.zeros(lambda_)
            
            for i in range(lambda_):
                if remaining <= 0:
                    break
                # Sample
                y = m + sigma * rng.multivariate_normal(np.zeros(n), C, 1)[0]
                # Clip to box
                y = np.clip(y, -5, 5)
                Y[i] = y
                F[i] = func(y)
                remaining -= 1
                calls_made += 1
            
            if remaining <= 0:
                break
                
            # Ensure we evaluated all lambda points in this generation
            # If remaining ran out mid-generation, we just use what we have?
            # Better to break if not enough budget for a full generation to maintain CMA-ES integrity?
            # Or just proceed with partial? Standard CMA-ES requires full pop.
            # If remaining < lambda_, we might do a final local search or just stop.
            # For simplicity, if remaining < lambda_, we do a few random steps or break.
            if remaining < lambda_:
                # Use remaining budget for random sampling around best
                for _ in range(remaining):
                    x_try = m + sigma * 0.5 * rng.standard_normal(n)
                    x_try = np.clip(x_try, -5, 5)
                    f_try = func(x_try)
                    if f_try < best_f:
                        best_f = f_try
                        best_x = x_try.copy()
                break

            # Sort
            idx = np.argsort(F)
            Y = Y[idx]
            F = F[idx]
            
            # Update best
            if F[0] < best_f:
                best_f = F[0]
                best_x = Y[0].copy()
                no_improve = 0
            else:
                no_improve += 1
            
            # CMA-ES Update
            # Recalculate mean
            m_new = np.sum(weights[:, None] * Y[:mu], axis=0)
            
            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * (m_new - m) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (m_new - m) / sigma
            
            # Update covariance matrix
            C = (1 + c_1 * (2 - c_1) - c_mu) * C
            C += c_1 * (np.outer(p_c, p_c) - np.eye(n))
            C += c_mu * np.sum(weights[:, None, None] * np.outer(Y[:mu] - m, Y[:mu] - m), axis=0)
            
            # Symmetrize
            C = (C + C.T) / 2
            
            m = m_new
            
            # Update sigma
            # Check if p_sigma is within expected norm
            norm_ps = np.linalg.norm(p_sigma)
            expected_norm = np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1) / (n + mu_eff)))
            # Actually, the standard check uses the expected value of the norm
            # E[||N(0, I)||] = sqrt(2) * Gamma((n+1)/2) / Gamma(n/2)
            # But the standard CMA-ES implementation uses:
            # sigma *= exp(c_sigma / hs * (||p_sigma|| / E[||N(0, I)||] - 1))
            # Where hs is the history threshold.
            
            # Let's use the standard formula from Hansen's paper
            # ||p_sigma|| / sqrt(1 - (1 - c_sigma)^(2 * (gen+1) / (n + mu_eff)))
            # This denominator approximates the expected norm of the path.
            denom = np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1) / (n + mu_eff)))
            if denom > 1e-8:
                sigma = sigma * np.exp(c_sigma / hs * (norm_ps / denom - np.sqrt(1 - np.exp(-2 * mu_eff / n))))
            
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            gen += 1
            
            # Restart logic
            if no_improve > 30 and remaining > 20:
                # Try a random restart
                x_try = rng.uniform(-5, 5, n)
                f_try = func(x_try)
                remaining -= 1
                if f_try < best_f:
                    m = x_try.copy()
                    best_f = f_try
                    best_x = x_try.copy()
                    sigma = 2.0
                    C = np.eye(n)
                    p_sigma = np.zeros(n)
                    p_c = np.zeros(n)
                    no_improve = 0
                else:
                    # If random restart fails, shrink sigma and continue?
                    # Or just reset local state around current best?
                    sigma = max(sigma * 0.5, 0.1)
                    C = np.eye(n)
                    p_sigma = np.zeros(n)
                    p_c = np.zeros(n)
                    no_improve = 0
                    gen = 0 # Reset generation counter for sigma update denominator?
        
        return best_x, best_f
# EVOLVE-BLOCK-END
