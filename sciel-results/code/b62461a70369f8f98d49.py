import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 320
        
        # CMA-ES Parameters
        lam = 8  # Population size
        mu = lam // 2  # Number of best points
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        # Initial state
        x_mean = np.zeros(dim)
        sigma = 1.0
        C = np.eye(dim)
        
        # Path lengths
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Learning rates
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        best_x = x_mean.copy()
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(42)
        
        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            try:
                fx = func(np.clip(x, -5, 5))
                calls_made += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx
            except Exception:
                calls_made += 1
                return 1e12

        # Initial evaluation
        f_mean = safe_call(x_mean)
        
        # Expected norm of standard normal
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(32*dim**2))
        
        while calls_made < budget and time.time() < deadline:
            # 1. Sample population
            # Compute C^-1/2 for sampling and updates
            # Eigen decomposition of C
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip eigenvalues to avoid division by zero
            eigvals = np.maximum(eigvals, 1e-12)
            C_inv_sqrt = (eigvecs * (1 / np.sqrt(eigvals))) @ eigvecs.T
            
            ys = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                # y = x_mean + sigma * C_inv_sqrt @ z ?
                # No, if we sample z ~ N(0, I), then C_inv_sqrt @ z ~ N(0, C_inv)?
                # We want y ~ N(x_mean, sigma^2 C).
                # Let y = x_mean + sigma * A @ z.
                # Var(y) = sigma^2 A A^T. We want A A^T = C.
                # So A = C^{1/2}.
                # C^{1/2} = eigvecs @ diag(sqrt(eigvals)) @ eigvecs.T
                C_sqrt = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
                y = x_mean + sigma * (C_sqrt @ z)
                ys.append(y)
            
            # 2. Evaluate
            fs = [safe_call(y) for y in ys]
            
            # 3. Sort
            order = np.argsort(fs)
            ys = [ys[i] for i in order]
            fs = [fs[i] for i in order]
            
            # 4. Update Mean
            x_mean_old = x_mean.copy()
            x_mean = np.zeros(dim)
            for i in range(mu):
                x_mean += weights[i] * ys[i]
            
            # 5. Update Step Size
            # p_sigma update
            # p_sigma = (1-c_sigma) p_sigma + sqrt(c_sigma(2-c_sigma)mueff) / sigma * C^-1/2 (x_mean_new - x_mean_old)
            diff = x_mean - x_mean_old
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (C_inv_sqrt @ diff)
            
            # sigma update
            norm_ps = np.linalg.norm(ps)
            sigma *= np.exp((c_sigma / d_sigma) * (norm_ps / chi_n - 1))
            
            # 6. Update Covariance
            # h_sigma
            norm_pc = np.linalg.norm(pc)
            h_sigma = (norm_ps / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made + 1))) < (1 + 2 / (dim + 1)) * chi_n)
            
            # pc update
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (C_inv_sqrt @ diff)
            
            # C update
            # C = (1 - c_1 - c_mu) C + c_1 (pc pc^T + (1 - h_sigma) c_c (2 - c_c) C) + c_mu sum w_i (y_i - x_mean_old)(y_i - x_mean_old)^T / sigma^2
            # Note: The sum term uses y_i relative to x_mean_old, scaled by sigma?
            # Standard formula:
            # C <- (1 - c_1 - c_mu)C + c_1 (pc pc^T + (1 - h_sigma) c_c (2 - c_c) C) + c_mu / sigma^2 sum w_i (y_i - m_old)(y_i - m_old)^T
            # Wait, y_i are the actual points. The covariance update usually operates on the standardized variables or adjusts for sigma.
            # Let's use the formulation where we update C directly.
            
            rank1 = np.outer(pc, pc)
            if not h_sigma:
                rank1 += (1 - h_sigma) * c_c * (2 - c_c) * C
            
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                diff_i = (ys[i] - x_mean_old) / sigma
                rank_mu += weights[i] * np.outer(diff_i, diff_i)
            
            C = (1 - c_1 - c_mu) * C + c_1 * rank1 + c_mu * rank_mu
            
            # Regularize C to ensure positive definiteness
            # Symmetrize
            C = (C + C.T) / 2
            # Eigen check
            eigvals, _ = np.linalg.eigh(C)
            if np.any(eigvals < 1e-12):
                C = C + (1e-12 - np.min(eigvals)) * np.eye(dim)

        return best_x, best_f
