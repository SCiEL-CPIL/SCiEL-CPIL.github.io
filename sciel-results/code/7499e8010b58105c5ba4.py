import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        # Counting wrapper
        calls = [0]
        def count_func(x):
            calls[0] += 1
            return float(func(np.asarray(x, dtype=float)))

        def time_ok():
            return time.time() < deadline and calls[0] < budget

        # Phase 1: Coarse grid search for early progress
        # Grid points: 3^dim = 243 for dim=5
        # But we only have 500 calls. 243 is a lot.
        # Let's use a smaller grid or randomized search.
        # Randomized search is often better for black-box.
        # Let's do 100 random points first.
        
        best_x = None
        best_f = float('inf')
        
        # Initial random points
        n_init = 100
        X_init = rng.uniform(-5, 5, (n_init, dim))
        for i in range(n_init):
            if not time_ok():
                break
            x = X_init[i]
            f = count_func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        if not time_ok() or best_x is None:
            return best_x if best_x is not None else np.zeros(dim), best_f

        # Phase 2: CMA-ES
        # Use the best point as start
        x0 = best_x.copy()
        
        # CMA-ES parameters
        lambda_ = max(4 * dim, 10)  # 20 for dim=5
        mu = lambda_ // 2
        weights = np.zeros(lambda_)
        w = np.log(lambda_ / 2 + 1) - np.log(np.arange(1, lambda_ + 1))
        weights[:mu] = w[:mu]
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights[:mu] ** 2)
        
        n = dim
        c_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (lambda_ + 1)) - 1) / (n + mu_eff)
        c_c = 4 / (n + 4)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(c_1 / 2, mu_eff / 2 - c_1 / 2)
        d_sigma = n / 2 + 1
        
        x = x0.copy()
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        best_f_cma = best_f
        best_x_cma = best_x.copy()
        
        # Precompute Cholesky factor of C
        # C is identity initially
        L = np.eye(dim)
        
        while time_ok():
            # Check if we can afford a full population
            if calls[0] + lambda_ > budget:
                # If not, do a single step? Or stop?
                # CMA-ES needs a full population for stability.
                # Let's stop CMA-ES and maybe do a local polish?
                # Or just break.
                break
            
            # Generate offspring
            Y = np.zeros((lambda_, dim))
            F = np.zeros(lambda_)
            
            # Update Cholesky if C changed? 
            # In standard CMA-ES, C is updated after the step.
            # So we use the current C to generate.
            # We need to ensure C is PD.
            # Let's recompute Cholesky every step for safety, though expensive.
            # For dim=5, it's cheap.
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If not PD, fix C
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = (eigvecs * eigvals) @ eigvecs.T
                L = np.linalg.cholesky(C)
            
            for i in range(lambda_):
                if not time_ok():
                    break
                z = rng.standard_normal(dim)
                y = x + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                Y[i] = y
                F[i] = count_func(y)
                
                if F[i] < best_f_cma:
                    best_f_cma = F[i]
                    best_x_cma = y.copy()
                    best_f = F[i]
                    best_x = y.copy()
            
            if not time_ok():
                break
            
            # Sort
            idx = np.argsort(F)
            Y_sorted = Y[idx]
            F_sorted = F[idx]
            
            # Update mean
            x_new = np.sum(weights[:mu, np.newaxis] * Y_sorted[:mu], axis=0)
            
            # Update step size
            y_sigma = (x_new - x) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update covariance
            y_diff = (x_new - x) / sigma
            pc = (1 - c_c) * pc + c_c * y_diff
            
            # Rank one update
            C_rank1 = np.outer(y_diff, y_diff)
            
            # Rank mu update
            C_rankmu = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (Y_sorted[i] - x) / sigma
                C_rankmu += weights[i] * np.outer(y_i, y_i)
                
            C = (1 + c_1 * (mu_eff - 2 + 2 / c_c) - c_mu) * C \
                + c_1 * C_rank1 \
                + c_mu * C_rankmu
                
            # Symmetrize and fix
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = (eigvecs * eigvals) @ eigvecs.T
            
            x = x_new
            
        return best_x, best_f
