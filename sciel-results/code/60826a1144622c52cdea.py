"""Hybrid Halton Exploration + CMA-ES Optimizer for BBOB f17."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Hybrid optimizer:
    1. Halton sequence for global exploration (fast initial progress).
    2. CMA-ES for local refinement (robust convergence).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _halton(self, n, dim, start=1):
        """
        Generate n Halton sequence points in [0, 1)^dim.
        """
        points = np.zeros((n, dim))
        for d in range(dim):
            base = d + 2  # Bases: 2, 3, 5, 7, 11 for dim=5
            for i in range(n):
                idx = i + start
                f = 1.0
                r = 0.0
                while idx > 0:
                    f /= base
                    r += f * (idx % base)
                    idx //= base
                points[i, d] = r
        return points

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        calls = 0
        best_x = None
        best_f = np.inf
        
        # Helper to evaluate and track
        def evaluate(x):
            nonlocal best_x, best_f, calls
            # Ensure input is a valid numpy array
            x = np.asarray(x, dtype=np.float64)
            # Clip to bounds as per problem description (func clips, but we ensure input is clean)
            x_clipped = np.clip(x, -5.0, 5.0)
            fx = float(func(x_clipped))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # --- Phase 1: Global Exploration with Halton Sequence ---
        # Use a fixed number of points for exploration. 
        # 50-100 points is usually enough to find a region close to the optimum in 5D.
        # We use 60 points to leave ~440 for CMA-ES.
        n_explore = 60
        # Generate Halton points in [0,1)^dim
        halton_points = self._halton(n_explore, dim)
        # Map to [-5, 5]^dim
        explore_points = halton_points * 10.0 - 5.0
        
        # Evaluate exploration points
        for i in range(n_explore):
            if calls >= budget or time.time() > deadline:
                break
            x = explore_points[i]
            evaluate(x)
            
        # If we haven't found any point (should not happen), fallback
        if best_x is None:
            x0 = np.zeros(dim)
            evaluate(x0)

        # --- Phase 2: Local Refinement with CMA-ES ---
        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Initialize CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Step size adaptation
        cs = (mueff + 1) / (dim + mueff + 2)
        ds = 1 + 2 * max(0, (mueff - 1) / (dim + 1)) + cs
        
        # Covariance matrix adaptation
        c1 = (2 / ((dim + 1.3)**2 + mueff)) * (1 - cs) / (1 + cs)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        rng = np.random.default_rng(42)
        
        # Start CMA-ES from the best point found in Phase 1
        xmean = best_x.copy()
        
        # Initial sigma: Start a bit larger to allow exploration around the Halton-found best
        # But not too large to avoid wandering away.
        sigma = 0.5 
        
        # Initial covariance
        C = np.eye(dim)
        
        # Path variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Main CMA-ES Loop
        while calls < budget:
            if time.time() > deadline:
                break
                
            # If remaining budget is less than population size, we can't do a full generation.
            # We could do partial, but standard CMA-ES expects full generations for stability.
            # However, to squeeze out last bits, if remaining < lam, we break.
            if budget - calls < lam:
                break

            # Compute Cinv_sqrt
            evals, evecs = np.linalg.eigh(C)
            evals = np.maximum(evals, 1e-12)
            Cinv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
            
            # Sample offspring
            ys = np.zeros((lam, dim))
            for i in range(lam):
                z = rng.standard_normal(dim)
                ys[i] = xmean + sigma * (Cinv_sqrt @ z)
            
            # Evaluate offspring
            fvals = np.array([evaluate(y) for y in ys])
            
            # Sort
            idx = np.argsort(fvals)
            ys = ys[idx]
            fvals = fvals[idx]
            
            # Update xmean
            xmean_new = xmean + sigma * sum(weights[i] * ys[i] for i in range(mu))
            
            # Update p_sigma
            d = (xmean_new - xmean) / sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * (Cinv_sqrt @ d)
            
            # Update sigma
            chiN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            sigma = sigma * np.exp(cs / ds * (np.linalg.norm(p_sigma) / chiN - 1))
            
            # Update C
            y = (xmean_new - xmean) / sigma
            rank1 = np.outer(y, y)
            
            y_selected = (ys[:mu] - xmean) / sigma
            rankmu = sum(weights[i] * np.outer(y_selected[i], y_selected[i]) for i in range(mu))
            
            C = (1 - c1 * (1 - cs)**2) * C + c1 * rank1 + cmu * rankmu
            C = (C + C.T) / 2
            
            # Update xmean
            xmean = xmean_new
            
            # Optional: Stagnation check / Sigma clamp
            if sigma < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
