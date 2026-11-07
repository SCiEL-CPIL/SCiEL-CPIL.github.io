import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        lam = 4 + int(3 * np.log(n))
        
        # Weights
        w = np.log((lam + 1) / 2) - np.log(np.arange(1, lam + 1))
        w /= np.sum(w)
        
        # Effective population size
        mu = lam // 2
        mu_eff = 1.0 / np.sum(w[:mu]**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 - mu_eff + c_sigma) / (n + 4 - mu_eff)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Initialize
        rng = np.random.default_rng(0)
        x_mean = np.zeros(n) # Start at center
        sigma = 2.0 # Initial step size, covers [-2, 2] with 1 sigma, good for [-5,5]
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        best_x = x_mean.copy()
        best_f = np.inf
        calls = 0
        
        # Pre-compute Cholesky factor
        try:
            C_root = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            C_root = np.eye(n)
            
        # Main Loop
        while calls < budget:
            # Determine how many we can evaluate in this generation
            # We must evaluate exactly 'lam' points per generation for CMA-ES logic
            # If remaining budget < lam, we break or do a partial evaluation?
            # Standard CMA-ES needs full population. 
            # If budget < lam, we can't do a full generation.
            # Given budget=500 and lam=7, we do 71 full generations (497 calls).
            # 3 calls left. We can just stop or do 3 single-point perturbations.
            # Stopping is safer and simpler.
            
            if budget - calls < lam:
                break
            
            # Sample
            # z ~ N(0, I)
            z = rng.standard_normal((lam, n))
            # y = x_mean + sigma * (C_root @ z.T).T
            y = x_mean + sigma * (C_root @ z.T).T
            
            # Clip to box
            y = np.clip(y, -5, 5)
            
            # Evaluate
            f_vals = np.array([func(y[i]) for i in range(lam)])
            calls += lam
            
            # Update best
            idx_best = np.argmin(f_vals)
            if f_vals[idx_best] < best_f:
                best_f = f_vals[idx_best]
                best_x = y[idx_best].copy()
                
            # Sort
            sort_idx = np.argsort(f_vals)
            y_sorted = y[sort_idx]
            f_sorted = f_vals[sort_idx]
            
            # Update mean
            x_new_mean = np.sum(w[:mu, None] * y_sorted[:mu], axis=0)
            
            # Calculate difference
            diff = x_new_mean - x_mean
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * diff
            
            # Hessian norm approximation for CSA (or just use Euclidean norm for simplicity/robustness in low dim)
            # Standard CSA uses sqrt(Tr(C) - 1) / sqrt(2) factor.
            # For simplicity and stability in 5D, let's use the simpler step size adaptation often found in basic implementations:
            # sigma *= exp(c_sigma / d_sigma * (||p_sigma|| / chi_n - 1))
            
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            
            norm_p_sigma = np.linalg.norm(p_sigma)
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            
            # Update p_c
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * diff
            
            # Update Covariance
            # Delta H = (1 + c_mu * (2 - c_mu)) * (p_c @ p_c.T - C)
            # Rank-mu update
            y_psi = (y_sorted[:mu] - x_mean) / sigma
            rank_mu = np.sum(w[:mu, None] * (y_psi[:, None] * y_psi[None, :]), axis=0)
            
            C = (1 + c_1 * (2 - c_1) + c_mu - c_1) * C \
                - c_1 * C @ C \
                + c_1 * (p_c @ p_c.T) \
                + c_mu * rank_mu
                
            # Ensure symmetry and positive semi-definite
            C = (C + C.T) / 2
            
            # Check for positive definiteness, if not, re-initialize C or add small identity
            try:
                C_root = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Add small regularization
                C = C + 1e-6 * np.eye(n)
                try:
                    C_root = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Reset to identity if it fails completely
                    C = np.eye(n)
                    C_root = np.eye(n)
                    p_sigma = np.zeros(n)
                    p_c = np.zeros(n)
                    sigma = max(sigma, 0.1) # Don't let sigma vanish too fast
            
            x_mean = x_new_mean
            
        return best_x, best_f
