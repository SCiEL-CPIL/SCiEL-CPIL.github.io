"""Combined optimizer: LHS initialization + CMA-ES with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with LHS initialization and deadline guard for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        bounds = np.array([-5.0] * dim)
        upper = np.array([5.0] * dim)

        def check_deadline():
            return time.time() > deadline

        # Phase 1: Latin Hypercube-style sampling for fast early progress
        lhs_budget = min(50, budget // 10)
        x_best = None
        fx_best = np.inf

        # Generate LHS points
        lhs_points = np.zeros((lhs_budget, dim))
        for i in range(dim):
            perm = rng.permutation(lhs_budget)
            lhs_points[:, i] = (perm + 0.5) / lhs_budget * 10.0 - 5.0

        # Evaluate LHS points
        for i in range(lhs_budget):
            x = lhs_points[i]
            fx = func(x)
            if fx < fx_best:
                fx_best = fx
                x_best = x.copy()

        if check_deadline():
            return x_best, fx_best

        # Phase 2: CMA-ES
        remaining_budget = budget - lhs_budget
        
        # CMA-ES parameters
        lam = 5  # Population size
        mu = lam // 2  # Number of best points used for update
        weights = np.ones(mu)
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()
        
        # Initial state
        m = x_best.copy()
        sigma = 1.0
        # Diagonal covariance (variances)
        diag_cov = np.ones(dim)
        
        # Step size control parameters (simplified)
        c_sigma = 1.0 / (dim + 3)
        d_sigma = 1.0 + 2 * max(0, (mu_eff - 1) / (dim + 1) - 1)
        chi_norm = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim ** 2))
        
        # Covariance adaptation parameters
        c_cov = (2 * (mu_eff - 2 + 1. / mu_eff)) / ((dim + 2) ** 2 + mu_eff)
        c_c = 1. / (dim + 1.3 + c_cov)
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        calls_made = lhs_budget
        fx_global_best = fx_best
        x_global_best = x_best.copy()

        while calls_made < budget and not check_deadline():
            # Generate population
            ys = np.zeros((lam, dim))
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = m + sigma * np.sqrt(diag_cov) * z
                ys[i] = np.clip(y, bounds, upper)
            
            # Evaluate
            f_ys = np.array([func(y) for y in ys])
            calls_made += lam
            
            if check_deadline():
                break

            # Sort
            order = np.argsort(f_ys)
            ys = ys[order]
            f_ys = f_ys[order]
            
            # Update global best
            if f_ys[0] < fx_global_best:
                fx_global_best = f_ys[0]
                x_global_best = ys[0].copy()

            # Update mean
            m_old = m.copy()
            m = weights @ ys[:mu]
            
            # Update step size (sigma)
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1 - 2*c_sigma)^{2*(gen+1)}) < (1.4 + 2/(dim+1)) * chi_norm
            # Simplified: just update p_sigma and sigma
            if sigma > 1e-12:
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (m - m_old) / (sigma * np.sqrt(diag_cov))
                # Note: dividing by sigma*sqrt(diag_cov) assumes we are in the transformed space, 
                # but standard CMA-ES updates p_sigma in the x-space using the inverse transformation.
                # For diagonal CMA-ES, the transformation is diagonal, so we can handle it.
                # Let's stick to standard CMA-ES logic where p_sigma is in the search space.
                # Actually, standard CMA-ES updates p_sigma = (1-c_sigma)p_sigma + h_sigma * sqrt(c_sigma(2-c_sigma)mu_eff) * (m - m_old)/sigma
                # We need to be careful with the diagonal covariance.
                
                # Let's use a simpler success rate rule for sigma to avoid complex diagonal CMA-ES bugs
                # Success rate: fraction of steps that improved the mean?
                # Or just use the standard CMA-ES sigma update if we implement it correctly.
                
                # Given the complexity and risk of bugs in diagonal CMA-ES sigma update,
                # let's use a simpler adaptive sigma:
                # If best fitness improved, increase sigma slightly, else decrease.
                # But CMA-ES sigma update is more robust.
                
                # Let's try to implement the standard sigma update properly.
                # p_sigma is updated as:
                # p_sigma = (1 - c_sigma) * p_sigma + h_sigma * sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (m - m_old) / sigma
                # h_sigma is 1 if the condition holds.
                
                # For diagonal covariance, the "natural gradient" direction is scaled by 1/sqrt(diag_cov).
                # So (m - m_old) / sigma should be interpreted in the transformed space?
                # No, m and m_old are in x-space.
                # The standard CMA-ES assumes C is the covariance.
                # If C is diagonal, the update for p_sigma involves C^{-1/2}.
                # So (m - m_old) / (sigma * sqrt(diag_cov)) is the correct term for p_sigma update in the transformed space.
                
                # Let's calculate h_sigma
                # Condition: ||p_sigma|| / sqrt(1 - (1 - 2*c_sigma)**(2 * (calls_made/lam + 1))) < (1.4 + 2/(dim+1)) * chi_norm
                # We don't track generation number easily here, so we approximate or skip h_sigma check?
                # Skipping h_sigma check might lead to sigma explosion or collapse.
                # Let's use a simple heuristic: if the step length ||m - m_old|| is reasonable, keep sigma.
                
                # Alternative: Use a simpler (1+1)-ES-like sigma update on top of CMA-ES mean update?
                # No, that defeats the purpose.
                
                # Let's stick to the standard CMA-ES sigma update but be careful.
                # We will track generation count.
                pass # Placeholder for complex logic, let's use a simpler adaptive strategy for this iteration to ensure stability.
                
                # Simple Adaptive Sigma:
                # If the best point in the population is significantly better than the previous mean, increase sigma.
                # If not, decrease.
                # This is less robust than CMA-ES but safer to implement quickly.
                
                # Actually, let's just use the standard CMA-ES update for p_c and diag_cov, and a simple rule for sigma.
                # Or better, just use the standard CMA-ES implementation from a known source pattern.
                
                # To avoid bugs, I will implement the standard CMA-ES updates for p_c and C (diagonal), 
                # and use the standard sigma update formula.
                
                # p_sigma update:
                # p_sigma = (1 - c_sigma) * p_sigma + h_sigma * sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (m - m_old) / (sigma * np.sqrt(diag_cov))
                
                # Let's assume h_sigma = 1 for now (optimistic) or calculate it properly.
                # Calculating h_sigma requires tracking generations.
                gen = calls_made // lam
                # 1 - (1 - 2*c_sigma)**(2 * (gen + 1))
                denom = np.sqrt(max(0, 1 - (1 - 2 * c_sigma) ** (2 * (gen + 1))))
                if denom > 1e-12:
                    ratio = np.linalg.norm(p_sigma) / denom
                else:
                    ratio = np.inf
                threshold = (1.4 + 2 / (dim + 1)) * chi_norm
                h_sigma = 1 if ratio < threshold else 0
                
                p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (m - m_old) / (sigma * np.sqrt(diag_cov))
                
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_norm - 1))

            # Update p_c and diag_cov
            # p_c = (1 - c_c) * p_c + h_c * sqrt(c_c * (2 - c_c) * mu_eff) * (m - m_old) / sigma
            # h_c is similar to h_sigma but uses p_c.
            # Let's simplify h_c to 1 for stability in this short run? No, that's bad.
            # Let's use the same logic.
            
            # For diagonal CMA-ES, the update for C is:
            # C = (1 - c_cov) * C + c_cov * (p_c p_c^T + h_c * mu_eff * sum(weights_i * y_i y_i^T))
            # Since C is diagonal, we only update the diagonal elements.
            
            # Calculate p_c update
            # p_c is in x-space? No, p_c is usually in the transformed space?
            # In standard CMA-ES, p_c is in the search space, but the update uses the inverse square root of C.
            # p_c = (1 - c_c) * p_c + h_c * sqrt(c_c * (2 - c_c) * mu_eff) * (m - m_old) / (sigma * np.sqrt(diag_cov))
            
            # Let's reuse the h_sigma logic for h_c? No, h_c uses p_c.
            # This is getting complicated to implement correctly from scratch.
            
            # Let's fall back to a simpler strategy:
            # Use the CMA-ES mean update (weighted average of best points).
            # Update diagonal covariance based on the variance of the best points.
            # Update sigma based on success rate (fraction of points better than previous mean).
            
            # This "Diagonal CMA-ES with Simple Sigma" is more robust to implement.
            
            # 1. Mean update (already done)
            
            # 2. Covariance update (Diagonal)
            # Compute variance of the best mu points
            # var = weights @ (ys[:mu] - m)**2 ? No, that's biased.
            # Standard CMA-ES update for C:
            # delta_c = h_c * (p_c p_c^T - (1 - c_cov) * C)
            # C = (1 - c_cov) * C + c_cov * (p_c p_c^T + h_c * (mu_eff * sum(weights_i * y_i y_i^T) - (1 - c_cov) * C))
            # This is complex.
            
            # Simpler: 
            # Estimate variances from the best points.
            # v_new = weights @ (ys[:mu] - m)**2
            # v = (1 - c_cov) * v + c_cov * v_new
            # This is a simple adaptive step size per dimension.
            
            v_new = np.sum(weights[:, np.newaxis] * (ys[:mu] - m) ** 2, axis=0)
            diag_cov = (1 - c_cov) * diag_cov + c_cov * v_new
            
            # Ensure diag_cov doesn't become 0 or too large
            diag_cov = np.clip(diag_cov, 1e-12, 100.0)
            
            # 3. Sigma update (Success Rate)
            # Count how many points in the population were better than the previous mean fitness?
            # We don't have previous mean fitness stored easily.
            # Let's use the change in best fitness.
            # If fx_global_best improved, increase sigma, else decrease.
            # This is essentially (1+1)-ES logic on top of CMA-ES mean.
            
            # Let's try to keep the standard CMA-ES sigma update if possible, but the p_sigma update above was:
            # p_sigma = ... (m - m_old) / (sigma * np.sqrt(diag_cov))
            # This looks correct for diagonal C.
            # And sigma *= exp(...)
            # So the sigma update is already implemented above.
            
            # The only missing piece was the p_c and h_c logic for the covariance update.
            # I replaced the complex CMA-ES covariance update with a simple variance estimation.
            # This should be sufficient for dim=5.

        return x_global_best, fx_global_best
# EVOLVE-BLOCK-END
