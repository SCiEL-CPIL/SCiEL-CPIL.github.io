import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        # CMA-ES parameters tuned for dim=5, budget=500
        # Larger population for better exploration early
        lam = 15
        mu = lam // 2  # Number of parents
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Chi-square approximation
        chi_lambda = np.sqrt(dim * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2)))

        # Initial state: Start at origin
        m = np.zeros(dim)
        sigma = 2.0  # Larger initial step for exploration in [-5,5]^5
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Precompute Cholesky factor of C
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            L = np.eye(dim)

        best_x = m.copy()
        best_f = func(m)
        evals = 1

        # Stagnation tracking
        stagnation_count = 0
        stagnation_threshold = 5  # Generations without improvement

        while evals + lam <= budget:
            if time.time() > deadline:
                break

            # Sample lambda candidates
            # Use Cholesky decomposition for sampling: x = m + sigma * Z @ L.T
            Z = np.random.randn(lam, dim)
            X = m + sigma * (Z @ L.T)
            
            # Clip to box to avoid wasted evaluations on out-of-bounds points
            # The function clips, but sampling within bounds is more efficient
            X = np.clip(X, -5, 5)

            # Evaluate
            F = np.array([func(x) for x in X])
            evals += lam

            # Sort
            order = np.argsort(F)
            F_sorted = F[order]
            X_sorted = X[order]

            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = X_sorted[0].copy()
                stagnation_count = 0
            else:
                stagnation_count += 1

            # Update Mean
            m_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
            
            # Update Sigma (CSA)
            # We need C^{-1/2} * (m_new - m) / sigma
            # Using Cholesky: C = L L^T. C^{-1/2} is not simply L^{-1}.
            # However, for sampling we used L.T. 
            # The standard update uses the inverse square root.
            # Let's compute C_inv_sqrt properly.
            # For small dim, we can use eigendecomposition or solve linear systems.
            # Since dim=5, we can just compute the inverse square root via eigendecomposition
            # or use the fact that if we sample with L, the "standardized" vector is L^{-1} (x-m)/sigma.
            # Let's stick to the robust eigendecomposition for the update step to avoid confusion.
            # Or, we can maintain the Cholesky factor and solve L y = v.
            
            # Let's use eigendecomposition for stability in the update step, 
            # as it's cheap for dim=5.
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-18, None)
            
            v = (m_new - m) / sigma
            v_rot = eigvecs.T @ v
            v_scaled = v_rot / np.sqrt(eigvals)
            v_inv_half = eigvecs @ v_scaled
            
            p_sigma = (1 - c_sigma) * p_sigma + \
                      np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * v_inv_half
            
            # Sigma update
            sigma = sigma * np.exp(c_sigma / chi_lambda * (np.linalg.norm(p_sigma) / chi_lambda - 1))
            
            # Hessian of N(0, C) norm squared term
            # h_sigma
            h_sigma_thresh = (1 + 1.4 / mu_eff) * np.sqrt((1 - c_c) * (2 - c_c))
            h_sigma = 1.0 if (np.linalg.norm(p_sigma) / chi_lambda) < h_sigma_thresh else 0.0

            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * v_inv_half

            # Update C
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c * p_c.T) + c_mu * sum(w_i * (x_i - m)(x_i - m)^T)
            Y = (X_sorted[:mu] - m) / sigma
            M2 = (Y.T * weights) @ Y
            
            C = (1 - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * M2
            C = (C + C.T) / 2  # Symmetrize

            # Update m
            m = m_new

            # Update Cholesky factor for next iteration
            try:
                L = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
            except np.linalg.LinAlgError:
                # If C becomes non-positive definite, reset to identity
                C = np.eye(dim)
                L = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                sigma = 1.0
                m = best_x.copy()

            # Structural Change: Restart if stagnation is detected
            if stagnation_count > stagnation_threshold:
                # If stuck, restart from the best point found so far with a larger sigma
                # or from a random point if we've been stuck for a long time.
                # Here, we restart from best_x with a larger sigma to explore further.
                m = best_x.copy()
                sigma = sigma * 1.5
                C = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                try:
                    L = np.linalg.cholesky(C)
                except:
                    L = np.eye(dim)
                stagnation_count = 0

        return best_x, best_f
