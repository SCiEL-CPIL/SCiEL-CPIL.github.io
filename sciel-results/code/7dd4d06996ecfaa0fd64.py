import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = float('inf')

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')
        
        # Wrap function to count calls and enforce limits
        def wrapped_func(x):
            if time.time() > deadline:
                raise TimeoutError()
            if self.calls >= budget:
                raise Exception("Budget Exceeded")
            self.calls += 1
            x = np.asarray(x, dtype=float)
            # Ensure it's the right shape
            if x.shape != (dim,):
                x = x.reshape(dim)
            x = np.clip(x, -5.0, 5.0)
            try:
                f = func(x)
            except Exception:
                f = float('inf')
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f

        # Strategy: Multi-start CMA-ES
        # 1. Initial global exploration with a few random starts or a large sigma CMA-ES.
        # 2. Refine with CMA-ES from the best point.
        
        # Since we have 500 calls and dim=5, we can afford a robust CMA-ES.
        # Let's implement a clean, robust CMA-ES.
        
        def run_cma_es(x0, sigma0, max_calls):
            """
            Runs CMA-ES starting from x0 with sigma0, using at most max_calls function evaluations.
            Returns best_x, best_f found during this run.
            """
            # Local state for this run
            local_calls = 0
            local_best_x = x0.copy()
            local_best_f = float('inf')
            
            # Parameters
            lam = 9  # Population size
            mu = lam // 2  # Number of parents
            
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1 / np.sum(w**2)
            
            # Adaptation rates
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (dim + 2) - 1))
            
            # State
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Precompute C^{-1/2} initially
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_inv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
            
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            
            while local_calls < max_calls:
                if time.time() > deadline:
                    break
                
                # Determine number of offspring
                n_off = min(lam, max_calls - local_calls)
                if n_off <= 0:
                    break
                
                # Sample offspring
                # Use Cholesky if possible for speed, else eigen
                try:
                    L = np.linalg.cholesky(C)
                    samples = []
                    for _ in range(n_off):
                        z = np.random.randn(dim)
                        y = x_mean + sigma * (L @ z)
                        y = np.clip(y, -5.0, 5.0)
                        fy = wrapped_func(y)
                        local_calls += 1
                        samples.append((y, fy))
                        if fy < local_best_f:
                            local_best_f = fy
                            local_best_x = y.copy()
                except:
                    # Fallback to eigen
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    std_eig = np.sqrt(eigvals)
                    samples = []
                    for _ in range(n_off):
                        z = np.random.randn(dim)
                        y = x_mean + sigma * ((eigvecs * std_eig) @ z)
                        y = np.clip(y, -5.0, 5.0)
                        fy = wrapped_func(y)
                        local_calls += 1
                        samples.append((y, fy))
                        if fy < local_best_f:
                            local_best_f = fy
                            local_best_x = y.copy()
                
                if time.time() > deadline:
                    break
                
                # Sort by fitness
                samples.sort(key=lambda x: x[1])
                
                # Update mean
                y_w = np.zeros(dim)
                for i in range(mu):
                    y_w += w[i] * samples[i][0]
                x_new_mean = y_w
                
                diff = x_new_mean - x_mean
                norm_diff = np.linalg.norm(diff)
                
                if norm_diff < 1e-14:
                    # Stagnation or identical points
                    # Try to restart or break
                    break
                
                # Update evolution paths
                # Need C^{-1/2} for the current C
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    C_inv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
                except:
                    C_inv_sqrt = np.eye(dim)
                
                # p_sigma update
                p_sigma = (1 - c_sigma) * p_sigma + \
                          np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ diff) / sigma
                
                # h_sigma check (approximate)
                # Theoretical expected norm of p_sigma is chi_n * sqrt(1 - (1-c_sigma)^t)
                # We use a simplified check based on current norm
                norm_p_sigma = np.linalg.norm(p_sigma)
                # If norm is significantly larger than expected, we might have a spike
                # Standard heuristic:
                h_sigma = 1.0
                # More accurate check:
                # chi_n * sqrt(1 - (1-c_sigma)**(2*(gen+1)))
                # Since we don't track gen easily here, we use a simple threshold
                # If norm_p_sigma > 1.4 * chi_n, it might be a spike
                if norm_p_sigma > 1.4 * chi_n:
                    h_sigma = 0.0
                
                # p_c update
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ diff) / sigma
                
                # Sigma update
                sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
                
                # Covariance update
                # Rank-1 update
                y_cov = p_c + c_c * diff  # Note: diff is the step, scaled by 1/sigma in paths? 
                # Wait, the standard formula uses the difference in means.
                # p_c is already scaled. The rank-1 term is p_c * p_c^T.
                # The rank-mu term uses the weighted outer products of the selected points relative to the new mean.
                
                c1_term = c_1 * np.outer(p_c, p_c)
                
                cmu_term = np.zeros((dim, dim))
                for i in range(mu):
                    y_i = samples[i][0]
                    diff_i = y_i - x_new_mean
                    cmu_term += w[i] * np.outer(diff_i, diff_i)
                
                C = (1 - c_1 - c_mu + mu_eff * c_mu) * C + c1_term + c_mu * cmu_term
                
                # Ensure symmetry and positive definiteness
                C = (C + C.T) / 2
                # Add small regularization if needed
                # C += 1e-12 * np.eye(dim)
                
                x_mean = x_new_mean
                
                # Optional: Check for stagnation
                # If sigma becomes too small, break
                if sigma < 1e-12:
                    break
            
            return local_best_x, local_best_f, local_calls

        # Main Strategy:
        # 1. Start with a few random points to get a sense of the landscape.
        # 2. Run CMA-ES from the best point with a moderate sigma.
        # 3. If time/budget remains, try another start or refine further.
        
        # Initial random sampling
        n_initial = min(10, budget)
        rng = np.random.default_rng(42)
        
        for i in range(n_initial):
            x = rng.uniform(-5, 5, dim)
            f = wrapped_func(x)
        
        if self.best_x is None:
            # Should not happen
            self.best_x = np.zeros(dim)
            self.best_f = wrapped_func(self.best_x)
        
        # Run CMA-ES from the best point found
        # Use a sigma that is somewhat large to explore, but not too large
        # Dim=5, box is 10 wide. Sigma=1.0 is reasonable.
        # We can use a fraction of the remaining budget for the first run.
        # Let's reserve 10% for potential restarts or just use most of it.
        # CMA-ES is efficient, so let's give it a good chunk.
        
        # Let's try a multi-start approach:
        # Run 1: From best point, sigma=1.0, use 80% of remaining budget.
        # Run 2: If budget remains, from a different random point or best point with smaller sigma.
        
        remaining_budget = budget - self.calls
        if remaining_budget <= 0:
            return self.best_x, self.best_f
        
        # Run 1
        budget_run1 = int(remaining_budget * 0.8)
        if budget_run1 < 20:
            budget_run1 = remaining_budget
            
        x_best, f_best, calls_used = run_cma_es(self.best_x.copy(), 1.0, budget_run1)
        
        if f_best < self.best_f:
            self.best_f = f_best
            self.best_x = x_best
        
        remaining_budget = budget - self.calls
        if remaining_budget > 20 and time.time() < deadline - 5:
            # Run 2: Refine from best point with smaller sigma
            # Or start from a random point if the first run didn't improve much?
            # Let's refine.
            sigma_refine = 0.5
            budget_run2 = remaining_budget
            x_best2, f_best2, calls_used2 = run_cma_es(self.best_x.copy(), sigma_refine, budget_run2)
            
            if f_best2 < self.best_f:
                self.best_f = f_best2
                self.best_x = x_best2
        
        return self.best_x, self.best_f
