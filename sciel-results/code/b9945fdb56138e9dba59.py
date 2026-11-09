import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        def track_call(x):
            nonlocal calls_used, best_f, best_x
            if calls_used >= budget:
                return best_f
            # Ensure x is a valid numpy array of correct shape
            x = np.asarray(x, dtype=float)
            if x.shape != (dim,):
                x = x.reshape(dim)
            val = func(x)
            calls_used += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # CMA-ES Parameters
        # For dim=5, lambda=10 is standard.
        lam = 10
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights**2)
        
        # Initial parameters
        # Start near center, as minimizer is in [-4, 4]^5
        mean = np.zeros(dim)
        
        # Strategy: Two-phase CMA-ES
        # Phase 1: Broad exploration with large sigma to find the basin
        # Phase 2: Fine-grained exploitation with adaptive CMA-ES
        
        # Phase 1: Simple random search with larger steps to get into the basin
        # Schaffers F7 has a global minimum at the origin (before shift/rotation).
        # The shift is in [-4, 4], so starting at 0 with sigma=2.5 should cover most of the space.
        # We will run a few iterations of a simple (1+lambda)-ES or just CMA-ES with large sigma.
        
        sigma = 2.5
        C = np.eye(dim)
        sigma_sqrt = 1.0
        Cs = np.zeros((dim, dim))
        ps = np.zeros(dim)
        
        # CMA-ES Update Constants
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1) - 1) - 1) + cs
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Population generation
        def generate_population(mean, sigma, C):
            # Sample from N(0, C)
            # Use Cholesky decomposition for sampling
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback if C is not positive definite
                L = np.eye(dim)
            
            z = np.random.randn(lam, dim)
            # Transform z to have covariance C
            # y = L @ z.T -> z @ L.T
            y = z @ L.T
            pop = mean + sigma * y
            return pop
        
        # Main Loop
        while calls_used < budget and time.time() < deadline - 2.0:
            # Generate population
            pop = generate_population(mean, sigma, C)
            
            # Evaluate population
            fit = np.zeros(lam)
            for i in range(lam):
                if calls_used >= budget:
                    break
                fit[i] = track_call(pop[i])
            
            if calls_used >= budget:
                break
                
            # Sort by fitness
            order = np.argsort(fit)
            pop_sorted = pop[order]
            fit_sorted = fit[order]
            
            # Update best
            if fit_sorted[0] < best_f:
                best_f = fit_sorted[0]
                best_x = pop_sorted[0].copy()
            
            # Check termination criteria
            # If sigma is very small or variance is very small
            if sigma < 1e-12:
                break
            if np.linalg.norm(mean - best_x) < 1e-8 * (1 + np.linalg.norm(best_x)):
                # We are very close to the best found, might be converged
                # But we should still use remaining budget if possible?
                # No, if we are converged, we can stop or do local polish.
                # For now, let's just break if we are extremely close and sigma is small.
                if sigma < 0.01:
                    break

            # Update Mean
            mean_new = np.sum(weights[:, None] * pop_sorted[:mu], axis=0)
            
            # Update Step Size (Sigma)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / chiN * (C ** (-0.5)) @ (mean_new - mean)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_used // lam + 1)))) < (1.4 + 2 / (dim + 1)) * chiN
            
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chiN - 1))
            
            # Update Covariance
            if hsig:
                Cs = (1 - cs) * Cs + cs * (np.outer(ps, ps) - C)
            else:
                Cs = (1 - cs) * Cs
            
            # Rank-Mu update
            v = pop_sorted[:mu] - mean
            delta = np.sum(weights[:, None] * (np.outer(v, v)), axis=0)
            C = (1 + c1 * hsig * (np.outer(ps, ps) - C) + c1 * (Cs - C) + cmu * (delta - C))
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Add small identity for stability
            C += 1e-6 * np.eye(dim)
            
            # Update mean
            mean = mean_new
            
            # Adaptive Sigma Reduction (Optional, but good for convergence)
            # If we are making progress, we can keep sigma. If not, reduce it?
            # Standard CMA-ES handles this via the sigma update rule.
            
            # Early termination if progress is negligible
            # If the best fitness hasn't improved in the last few generations, we might be stuck.
            # But CMA-ES is robust. Let's just let it run until budget or time.

        return best_x, best_f
