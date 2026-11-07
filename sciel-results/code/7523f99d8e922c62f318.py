"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        d = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        # Phase 1: Initial diverse sampling to find a good starting center
        # CMA-ES performs best when started near the optimum.
        # We use a small number of random samples to initialize the mean.
        n_init = 10
        points = [rng.uniform(-5, 5, d) for _ in range(n_init)]
        
        best_x = points[0].copy()
        best_f = func(best_x)
        calls = 1

        fits = [best_f]
        for x in points[1:]:
            if not time_ok() or calls >= budget:
                break
            fx = func(x)
            calls += 1
            fits.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # If we didn't use all init calls, we have some budget left
        remaining = budget - calls
        if remaining <= 0:
            return best_x, best_f

        # Phase 2: CMA-ES Optimization
        # Standard CMA-ES parameters
        # Population size: lambda = 4 + floor(3 * log(10 + dim))
        lam = 4 + int(3 * np.log(10 + d))
        # Ensure we don't exceed remaining budget in first generation
        if lam > remaining:
            lam = max(2, remaining)
            
        # Initial step size
        sigma = 1.0
        
        # Mean of the distribution
        m = best_x.copy()
        
        # Covariance matrix (identity initially)
        C = np.eye(d)
        
        # Inverse of C
        C_inv = np.eye(d)
        
        # Eigen decomposition cache for update
        # For small d (5), we can just do full updates or use simple heuristics
        # Standard CMA-ES updates:
        # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu) * sqrt(C) * (w_best_x - m) / sigma
        # p_c = (1 - c_c) * p_c + h_sigma * sqrt(c_c * (2 - c_c) * mu) * sqrt(C) * (w_best_x - m) / sigma
        # C = (1 - mu_cov - 2 * mu1 * c_c) * C + mu_cov * sum(wi * (xi - m)(xi - m)^T) + mu1 * p_c p_c^T
        # where mu is number of recombination weights
        
        # Simplified CMA-ES for robustness and speed in this context:
        # We will use a simplified version that adapts sigma and direction
        # but might not fully adapt C as efficiently as a perfect implementation,
        # or we can implement a basic diagonal CMA-ES (SEPA) which is often very effective
        # for BBOB functions and much faster to compute for small d.
        
        # Let's implement a basic CMA-ES with full covariance update since d=5 is small.
        
        # Parameters
        c_sigma = (mu + 2) / ((d + 2) + mu)
        c_c = (4 + mu / c_sigma) / ((d + 1.3) + mu / c_sigma)
        c_1 = 2 / ((d + 1.3)**2 + mu)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu) / ((d + 2)**2 + mu_eff))
        
        # Recombination weights
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1 / np.sum(w**2)
        
        # Recalculate c parameters with correct mu_eff if needed, but standard approx is fine
        # For d=5, let's stick to standard values
        
        # State variables
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        # We need to be careful with the budget.
        # Each generation costs lam function evaluations.
        
        while calls < budget and time_ok():
            # Generate population
            # We need to generate lam points from N(m, sigma^2 * C)
            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset to identity
                C = np.eye(d)
                L = np.eye(d)
                
            # Generate random normal vectors
            z = rng.standard_normal((lam, d))
            # Transform to N(0, C)
            # x = m + sigma * L * z^T
            pop = m[None, :] + sigma * (z @ L.T)
            
            # Evaluate population
            f_pop = np.array([func(x) for x in pop])
            calls += lam
            
            # Sort by fitness
            order = np.argsort(f_pop)
            sorted_pop = pop[order]
            sorted_f = f_pop[order]
            
            # Update best
            if sorted_f[0] < best_f:
                best_f = sorted_f[0]
                best_x = sorted_pop[0].copy()
                
            # Weighted mean of top mu individuals
            y = sorted_pop[:mu]
            m_new = np.sum(w[:, None] * y, axis=0)
            
            # Update sigma
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * sqrt(C) * (m_new - m) / sigma
            # sqrt(C) is L
            diff = (m_new - m) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L @ diff)
            
            # Update p_c
            # h_sigma = (||p_sigma|| / sqrt(1 - (1-c_sigma)^(2*g))) < (1 + 2/(d+1)) * sqrt(c_sigma * (2 - c_sigma))
            # g is generation counter, we can approximate or skip h_sigma for simplicity if it's complex
            # For now, let's assume h_sigma = 1 for simplicity, or calculate it properly.
            # Proper h_sigma:
            # This requires tracking the norm of p_sigma over time.
            # Simplified: h_sigma = 1.0
            h_sigma = 1.0
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (L @ diff)
            
            # Update C
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c p_c^T + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * sum(wi * (yi - m)(yi - m)^T)
            # Note: The term (1 - h_sigma) is usually small.
            
            # Calculate the sum of outer products
            y_diff = y - m
            sum_outer = np.zeros((d, d))
            for i in range(mu):
                sum_outer += w[i] * np.outer(y_diff[i], y_diff[i])
                
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * sum_outer
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update mean
            m = m_new
            
            # Adapt sigma
            # sigma = sigma * exp(c_sigma * (||p_sigma|| / sqrt(mu_eff) - chi_d) / d_sigma)
            # chi_d = sqrt(d) * (1 - 1/(4*d) + 1/(21*d^2))
            chi_d = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
            d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (d + 1) - 1) + c_sigma
            
            norm_p_sigma = np.linalg.norm(p_sigma)
            sigma *= np.exp(c_sigma * (norm_p_sigma / np.sqrt(mu_eff) - chi_d) / d_sigma)
            
            # Prevent sigma from becoming too small or too large
            if sigma < 1e-10:
                sigma = 1e-10
            if sigma > 10:
                sigma = 10

        return best_x, best_f
# EVOLVE-BLOCK-END
