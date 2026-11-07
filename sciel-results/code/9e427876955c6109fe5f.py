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
        
        # CMA-ES parameters
        # Population size
        lam = max(10, 4 + 3 * int(np.log(dim)))
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        mueff = 1.0 / np.sum(weights**2)
        
        # Adaptation rates
        cs = (mueff + 2) / (lam + mueff + 5)
        cc = (mueff + 1.5) / (lam + mueff + 5.5)
        c1 = 2 / ((lam + 2)**2 + mueff)
        cmu = c1 * (mueff - 2 + 2 / mueff) / ((lam + 2)**2 - mueff)
        
        # Initial state
        # Start at 0, which is often a good guess for centered functions
        # Or random? The problem says minimizer is in [-4, 4]^5.
        # Let's start at 0.
        x_mean = np.zeros(dim)
        sigma = 2.0  # Initial step size
        
        # Evolution paths
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        D = 1.0
        B = np.eye(dim)
        
        # Precompute Cholesky
        def chol_update(C):
            try:
                L = np.linalg.cholesky(C)
                return L
            except np.linalg.LinAlgError:
                # If singular, add small jitter
                return np.linalg.cholesky(C + 1e-10 * np.eye(dim))
        
        L = chol_update(C)
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        
        # Budget management
        # We have `budget` evaluations left (1 used for best_f)
        evals_left = budget - 1
        
        while evals_left >= lam:
            # Check time
            if time.time() - start_time > 290:
                break
                
            # Sample population
            ys = np.array([L @ rng.standard_normal(dim) for _ in range(lam)])
            # rng needs to be defined
            # Let's use a local rng
            rng = np.random.default_rng(0) # Fixed seed for reproducibility in single run? 
            # Actually, the evaluator might call this multiple times with different instances.
            # Using a fixed seed for the optimizer's internal randomness is fine as the instance changes the function.
            
            # Wait, I need to define rng properly.
            rng = np.random.default_rng(42)
            
            # Re-sample with proper rng
            ys = np.array([L @ rng.standard_normal(dim) for _ in range(lam)])
            
            # Apply bounds?
            # The function clips, but for CMA-ES update, we should use the actual points.
            # If we clip the points before evaluation, the gradient information is lost.
            # Standard practice: evaluate clipped points, but update based on original samples?
            # No, usually we evaluate the clipped points.
            # Let's evaluate the clipped points.
            xs = np.clip(x_mean + sigma * ys, -5, 5)
            
            # Evaluate
            fs = np.array([func(x) for x in xs])
            
            # Update best
            idx_best = np.argmin(fs)
            if fs[idx_best] < best_f:
                best_f = fs[idx_best]
                best_x = xs[idx_best].copy()
            
            evals_left -= lam
            
            # Sort
            order = np.argsort(fs)
            xs_sorted = xs[order]
            fs_sorted = fs[order]
            
            # Update mean
            new_mean = np.sum(weights[:, None] * xs_sorted[:mu], axis=0)
            
            # Update evolution path ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (L.T @ (new_mean - x_mean) / sigma)
            
            # Check hsig
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (evals_left // lam + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1))
            
            # Update evolution path pc
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (L.T @ (new_mean - x_mean) / sigma)
            
            # Delta of Hesse
            delta_hess = (1 - hsig) * (1 - cc) + cc
            
            # Update covariance
            rank_one = np.outer(pc, pc)
            rank_mu = np.sum(weights[:, None, None] * np.outer(xs_sorted[:mu] - x_mean, xs_sorted[:mu] - x_mean) / sigma**2, axis=0)
            
            C = (1 + c1 * delta_hess - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu
            
            # Regularize C
            C = (C + C.T) / 2
            
            # Update L
            L = chol_update(C)
            
            # Update sigma
            sigma = sigma * np.exp(cs / (dim + 1) * (np.linalg.norm(ps) / np.sqrt(mueff) - 1))
            
            # Update mean
            x_mean = new_mean
            
            # Project mean to box?
            # If the mean goes out of bounds, the samples will be clipped heavily.
            # Clipping the mean helps keep the distribution centered in the valid region.
            x_mean = np.clip(x_mean, -5, 5)
            
            # Stagnation check
            # If sigma is very small, or no improvement for many generations
            if sigma < 1e-8:
                break

        # Use remaining budget for local refinement or more CMA steps?
        # If evals_left > 0 and < lam, we can do single steps or just return.
        # Given the small dim, a few local hill climbs might help.
        # But CMA-ES usually converges well.
        
        return best_x, best_f
