"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES implementation for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: The evaluator kills at 330s. We use 320s to be safe.
        deadline = start_time + 320.0
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size lambda
        # Standard: 4 + floor(3 * ln(n))
        # For n=5, 4 + 3*1.609 = 8.8 -> 8
        # We use 8 to balance exploration and exploitation
        pop_size = 4 + int(3 * np.log(dim))
        
        # Initial step size
        # Domain is [-5, 5]^5, minimizer in [-4, 4]^5.
        # We start wide to ensure we find the basin quickly.
        sigma = 2.0
        
        # Initial mean: Center of the likely region
        m = np.zeros(dim)
        
        # CMA-ES State
        C = np.eye(dim)  # Covariance matrix
        C_inv = np.eye(dim) # Inverse covariance matrix
        D = 1.0  # Eigenvalue product (sqrt(det(C)))
        
        # Learning rates (standard values)
        # mu = pop_size // 2
        mu = pop_size // 2
        # Weights: log weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        # mu_eff
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size adaptation parameters
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        ds = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + cs
        
        # Mean adaptation parameters
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initial p_sigma and p_c
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial eigen decomposition
        D = 1.0
        # We need to update C_inv and D periodically or use approximations
        # For simplicity and speed in this constrained budget, we will update 
        # C and C_inv every iteration using SVD, which is O(n^3) but n=5 is tiny.
        
        def update_cma_state():
            nonlocal C, C_inv, D
            # Eigen decomposition of C
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip eigenvalues to positive
            eigvals = np.clip(eigvals, 1e-12, None)
            D = np.prod(np.sqrt(eigvals))
            C_inv = np.linalg.inv(C)
            
        # Initial state
        update_cma_state()
        
        best_x = m.copy()
        best_f = float('inf')
        calls_made = 0
        rng = np.random.default_rng(42)
        
        def check_time():
            return time.time() > deadline
            
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # Initial evaluation of the mean
        f_m = func(m)
        calls_made += 1
        update_best(m, f_m)
        
        # Main Loop
        while calls_made < budget and not check_time():
            # 1. Sampling
            # We need to sample pop_size points
            # To stay within budget, we check if we have enough calls left
            # for the whole generation. If not, we stop or do partial?
            # Stopping is safer to avoid bias.
            if calls_made + pop_size > budget:
                break
                
            Y = rng.standard_normal((pop_size, dim))
            # Transform Y using C and D
            # x_i = m + D * C @ Y_i
            # Since D = sqrt(det(C)), and we want to use the Cholesky or Eigen form
            # Actually, the standard CMA-ES uses the square root of C.
            # Let's use the eigen decomposition for sampling:
            # x_i = m + D * eigvecs @ diag(sqrt(eigvals)) @ Y_i
            # Wait, D is sqrt(det(C)). 
            # The transformation is usually: z ~ N(0, I), x = m + sigma * B * d * z
            # where C = B diag(d^2) B^T.
            # So sigma is the step size, and B, d come from C.
            
            # Let's stick to the standard formulation:
            # C is the covariance of the search distribution.
            # We sample Z ~ N(0, I).
            # X = m + sigma * (C^{1/2} @ Z)
            
            # We maintain C. We need C^{1/2}.
            # From eigen decomposition: C = V diag(lambda) V^T
            # C^{1/2} = V diag(sqrt(lambda)) V^T
            
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            sqrt_eigvals = np.sqrt(eigvals)
            C_sqrt = (eigvecs * sqrt_eigvals) @ eigvecs.T
            
            X = m[None, :] + sigma * (Y @ C_sqrt.T)
            
            # Clip to box [-5, 5]
            X = np.clip(X, -5, 5)
            
            # 2. Evaluation
            F = np.array([func(x) for x in X])
            calls_made += pop_size
            
            # Update best
            for i in range(pop_size):
                update_best(X[i], F[i])
                
            if check_time():
                break
                
            # 3. Ranking
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]
            
            # 4. Weighted recombination (new mean)
            # m_new = sum(weights[i] * X_sorted[i])
            m_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
            
            # 5. Step-size control (CSA)
            # h_sigma = sqrt((1 - exp(-2/(cs*ds))) * mu_eff / dim)
            h_sigma = np.sqrt((1 - np.exp(-2 / (cs * ds))) * mu_eff / dim)
            
            # Delta C from previous iteration? 
            # We need C_prev for p_sigma update.
            # p_sigma = (1-cs)*p_sigma + h_sigma*sqrt(mu_eff)*C_inv_sqrt@(m_new-m)/sigma
            # We need C_inv_sqrt.
            C_inv_sqrt = (eigvecs * (1 / sqrt_eigvals)) @ eigvecs.T
            
            # Update p_sigma
            p_sigma = (1 - cs) * p_sigma + h_sigma * np.sqrt(mu_eff) * (C_inv_sqrt @ (m_new - m) / sigma)
            
            # Update sigma
            sigma = sigma * np.exp(cs / ds * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # 6. Covariance matrix adaptation
            # Update p_c
            h_c = (1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + cs) / (1 - cs)
            # This h_c formula is slightly different in some implementations, let's use standard:
            # h_c = (1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + cs) / (1 - cs) is for the threshold?
            # Actually, the standard check is:
            # if ||p_c|| / sqrt(1 - (1-cs)^{2i_gen}) < (1.4 + 2/(dim+1)) * h_sigma
            # For simplicity in a short run, we can skip the exact threshold or use a simplified one.
            # Let's use the standard update:
            p_c = (1 - cs) * p_c + cs * np.sqrt(mu_eff) * C_inv_sqrt @ (m_new - m) / sigma
            
            # Rank-one update
            C = (1 + c1 * mu_eff + c1 - c1) * C # Wait, formula is:
            # C = (1 + c1*mu_eff + c1 - c1) * C is wrong.
            # C = (1 + c1 * (2 - c1 - cmu*mu_eff)) * C
            #     + c1 * (m_new - m) (m_new - m)^T
            #     + cmu * sum(weights[i] * (y_i) (y_i)^T)
            # where y_i = (x_i - m) / sigma
            
            c1_coeff = 1 + c1 * (2 - c1 - cmu * mu_eff)
            C = c1_coeff * C + c1 * np.outer(m_new - m, m_new - m)
            
            # Rank-mu update
            # Y = (X_sorted[:mu] - m[None, :]) / sigma
            Y_mu = (X_sorted[:mu] - m[None, :]) / sigma
            # Sum of weighted outer products
            # C += cmu * sum(weights[i] * Y_mu[i] Y_mu[i]^T)
            # This is a sum of rank-1 updates.
            # For efficiency, we can do:
            # C += cmu * (Y_mu.T @ (weights[:, None] * Y_mu))
            
            C += cmu * (Y_mu.T @ (weights[:, None] * Y_mu))
            
            # Update m
            m = m_new
            
            # Update C state (inverse, D)
            update_cma_state()
            
            # Check for convergence
            if np.linalg.norm(m_new - m) < 1e-8 * sigma:
                 # Note: m was just updated to m_new, so this check is on the delta?
                 # No, we updated m = m_new AFTER the check logic in standard algo?
                 # Actually, we should check if the step was small.
                 pass
                 
            # Optional: Restart if sigma becomes too small or large
            if sigma < 1e-10 or sigma > 1e10:
                break
                
        return best_x, best_f

# EVOLVE-BLOCK-END
