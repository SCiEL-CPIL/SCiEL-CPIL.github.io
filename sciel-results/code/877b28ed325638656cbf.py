import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        
        # 1. Global Scan
        n_scan = 30
        # Use Sobol for better coverage
        try:
            import scipy.stats.qmc
            sampler = scipy.stats.qmc.Sobol(d=self.dim, scramble=True, seed=42)
            points = sampler.random(n_scan)
        except ImportError:
            points = rng.random((n_scan, self.dim))
            
        points = points * 10 - 5
        
        best_x = None
        best_f = np.inf
        candidates = []
        
        for x in points:
            if time.time() > deadline - 5.0: break
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            candidates.append((fx, x.copy()))
            
        if best_x is None:
             x0 = rng.uniform(-5, 5, self.dim)
             return x0, func(x0)

        # 2. Local Refinement with CMA-ES
        remaining = self.budget - len(candidates)
        if remaining <= 0:
            return best_x, best_f

        candidates.sort(key=lambda t: t[0])
        
        # Allocate budget to top candidates
        # CMA-ES is efficient, so we can give most to the best
        budget_best = int(remaining * 0.9)
        budget_second = remaining - budget_best
        
        def run_cma_es(start_x, start_f, n_evals):
            x = start_x.copy()
            fx = start_f
            # Initial sigma: large enough to explore, small enough to converge
            sigma = 1.0 
            # Population size
            lambda_pop = max(4, int(4 + 3 * np.log(self.dim)))
            # CMA-ES parameters
            cc = (4 + lambda_pop / self.dim) / (self.dim + 4 + 2 * lambda_pop / self.dim)
            cs = (lambda_pop / 2 + 1) / (self.dim + lambda_pop / 2 + 3)
            c1 = 2 / ((self.dim + 1.3) ** 2 + lambda_pop)
            cmu = min(1 - c1, 2 * ((lambda_pop / 2 - 1 + 1 / lambda_pop) - c1) / ((self.dim + 2) ** 2 + lambda_pop))
            
            # Matrices
            C = np.eye(self.dim)
            # Covariance matrix update state
            ps = np.zeros(self.dim)
            phes = np.zeros(self.dim)
            # Eigen decomposition cache for step size adaptation
            # Simplified: use sqrtm or eig every few steps? 
            # For speed, we can approximate or just use standard update.
            # Let's use a simplified update for speed:
            # D = diag(sigma * sqrt(eig(C)))
            
            # Precompute inverse square root of C? 
            # To keep it simple and fast, we might just use the standard update
            # but computing eigenvalues every step is O(N^3). N=5, so it's fine.
            
            for _ in range(n_evals):
                if time.time() > deadline - 2.0:
                    break
                if n_evals <= 0:
                    break
                    
                # Generate population
                # Get eigenvalues of C
                eigvals, eigvecs = np.linalg.eigh(C)
                # Ensure positive
                eigvals = np.maximum(eigvals, 1e-12)
                D = sigma * np.sqrt(eigvals)
                B = eigvecs
                
                # Sample
                Y = rng.standard_normal((lambda_pop, self.dim))
                # Transform
                Z = (B * D) @ Y.T
                X = x[None, :] + Z.T
                X = np.clip(X, -5, 5)
                
                # Evaluate
                Fs = np.array([func(xi) for xi in X])
                
                # Update mean
                weights = np.zeros(lambda_pop)
                mu = max(1, lambda_pop // 2)
                # Logarithmic weights
                log_w = np.log(np.arange(lambda_pop, 0, -1))
                w = np.exp(log_w - np.max(log_w))
                w = w / np.sum(w)
                # Only top mu have non-zero weight in standard CMA-ES, but let's use all for simplicity? 
                # Standard: w_i > 0 for i <= mu.
                w[:mu] = w[:mu]
                w[mu:] = 0
                w = w / np.sum(w)
                mu_eff = 1 / np.sum(w ** 2)
                
                x_old = x.copy()
                x = np.sum(w[:, None] * X, axis=0)
                
                # Update step size and covariance
                # ps update
                # Need B^-1 D^-1
                # D_inv = 1 / D
                # B_inv = B.T
                # ps = (1-cs)*ps + cs * sqrt(mu_eff) * B.T @ (D_inv * (x - x_old).T).T
                # Note: (x - x_old) is row vector
                diff = x - x_old
                # Transform to normal space
                diff_normal = (B.T @ (diff / D)) # diff / D element-wise, then dot with B.T
                ps = (1 - cs) * ps + cs * np.sqrt(mu_eff) * diff_normal
                
                # hsig check
                chi_n = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (n_evals + 1)))) < (1.4 + 2 / (self.dim + 1)) * chi_n
                
                # phes update
                phes = (1 - cs) * phes + hsig * np.sqrt(cs * (2 - cs) * mu_eff) * x
                
                # Covariance update
                # Delta h = (1 - hsig) * cs * (2 - cs)
                delta_h = (1 - hsig) * cs * (2 - cs)
                
                # Rank-1 update
                C = (1 + c1 * delta_h - c1 - cmu) * C + c1 * np.outer(ps, ps)
                
                # Rank-mu update
                # z = (X - x_old) / D (element-wise division by D for each row)
                Z = (X - x_old[None, :]) / D[None, :]
                # Weighted outer sum
                # w is shape (lambda_pop,)
                # Z is (lambda_pop, dim)
                # We need sum(w_i * z_i * z_i.T)
                # Z_w = Z * np.sqrt(w)[:, None]
                Z_w = Z * np.sqrt(w)[:, None]
                C = C + cmu * (Z_w.T @ Z_w)
                
                # Update sigma
                sigma = sigma * np.exp(cs / (5 * self.dim) * (np.linalg.norm(ps) / chi_n - 1))
                
                # Update best
                if Fs.min() < fx:
                    fx = Fs.min()
                    # Find index
                    idx = np.argmin(Fs)
                    # x is already updated to weighted mean, but we track best found
                    # For next iteration, we use the weighted mean as the new center
                    # But we should keep track of the absolute best
                    # The CMA-ES mean is the center. The best point is just a sample.
                    # We return the best point found so far.
                    
                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                # Eigen check for stability
                # If eigenvalues become negative, project to PD
                # For N=5, this is cheap
                eigvals_C, eigvecs_C = np.linalg.eigh(C)
                if np.any(eigvals_C < 0):
                    eigvals_C = np.maximum(eigvals_C, 1e-12)
                    C = (eigvecs_C * eigvals_C) @ eigvecs_C.T
                
                n_evals -= lambda_pop

            return x, fx

        # Refine best
        x_best, f_best = run_cma_es(best_x, best_f, budget_best)
        if f_best < best_f:
             best_x, best_f = x_best, f_best
             
        # Refine second best if exists and budget remains
        if len(candidates) > 1 and budget_second > 0:
             x_second, f_second = candidates[1]
             x_s, f_s = run_cma_es(x_second, f_second, budget_second)
             if f_s < best_f:
                 best_x, best_f = x_s, f_s

        return best_x, best_f
