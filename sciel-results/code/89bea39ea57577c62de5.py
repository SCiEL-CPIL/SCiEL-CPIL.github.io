"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Rastrigin-aware optimizer: periodic lattice scan + CMA-ES refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s kill
        dim = self.dim
        budget = self.budget
        calls = 0

        def safe_call(x):
            nonlocal calls
            if time.time() > deadline:
                return None
            calls += 1
            return func(np.clip(x, -5.0, 5.0))

        # Phase 1: Periodic lattice scan exploiting Rastrigin structure
        # Rastrigin has local minima at integer coordinates.
        # Scan integer grid points and half-integer offsets to find good basins.
        best_x = None
        best_f = np.inf

        # Integer grid points in [-4, 4]
        # 9^5 = 59049 points - too many. Use a coarser strategy.
        # Actually, let's be smarter: evaluate a subset first.
        
        # Strategy: Evaluate all integer points in [-4,4]^5 but that's 9^5=59049
        # We have 500 budget. Let's use a different approach.
        
        # Better: Use a random subset of integer and half-integer points,
        # then refine the best one with CMA-ES.
        
        rng = np.random.default_rng(42)
        
        # Generate candidate starting points from periodic structure
        # Integer points and half-integer offsets
        candidates = []
        
        # Add some random integer points
        for _ in range(30):
            pt = rng.integers(-4, 5, size=dim).astype(float)
            candidates.append(pt)
        
        # Add half-integer offsets
        for _ in range(20):
            pt = (rng.integers(-4, 5, size=dim) + 0.5).astype(float)
            candidates.append(pt)
        
        # Add some fully random points
        for _ in range(10):
            pt = rng.uniform(-5, 5, size=dim)
            candidates.append(pt)
        
        # Evaluate candidates
        for pt in candidates:
            if calls >= budget - 50:  # Reserve budget for refinement
                break
            if time.time() > deadline:
                break
            f = safe_call(pt)
            if f is not None and f < best_f:
                best_f = f
                best_x = pt.copy()
        
        # Phase 2: CMA-ES refinement from the best point
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_call(best_x)
            if best_f is None:
                return best_x, np.inf
        
        # CMA-ES parameters
        x = best_x.copy()
        f = best_f
        sigma = 0.5
        n = dim
        # CMA-ES population size
        lam = max(4, int(4 + 3 * np.log(n)))
        
        # Covariance matrix
        C = np.eye(n)
        # Step sizes
        cs = (n + 2) / (n + 8)
        ds = 1 + 2 * max(0, np.sqrt((n - 1) / (n + 3)) - 1) + cs
        cc = (4 + n / 4) / (n + 4 + 2)
        cmu = min(0.5, (mu - 2) / (n + mu - 5)) if (n + mu - 5) > 0 else 0.1
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Sort indices
        idx = np.arange(lam)
        
        for gen in range(budget - calls - 10):
            if calls >= budget - 5 or time.time() > deadline:
                break
            
            # Generate population
            ys = np.zeros((lam, n))
            for i in range(lam):
                if calls >= budget - 5 or time.time() > deadline:
                    break
                ys[i] = x + sigma * rng.multivariate_normal(np.zeros(n), C)
                ys[i] = np.clip(ys[i], -5, 5)
            
            # Evaluate
            fs = np.full(lam, np.inf)
            for i in range(lam):
                if calls >= budget - 5 or time.time() > deadline:
                    break
                f_val = safe_call(ys[i])
                if f_val is not None:
                    fs[i] = f_val
            
            # Sort
            order = np.argsort(fs)
            ys = ys[order]
            fs = fs[order]
            
            # Update best
            if fs[0] < f:
                f = fs[0]
                x = ys[0].copy()
            
            # Mean update
            x_new = np.sum(weights[:, None] * ys[:mu], axis=0)
            
            # Evolution path updates
            y_diff = (x_new - x) / sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * y_diff
            hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (gen + 1))
                    < (1.4 + 2 / (n + 1)) * np.sqrt((n + 1) / (n - 1)))
            p_c = (1 - cc) * p_c + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_diff
            
            # Covariance update
            delta_h = (1 - hsig) * cc * (1 - cc)
            rank1 = np.outer(p_c, p_c)
            rankmu = np.sum(weights[:, None, None] * (ys[:mu] - x)[:, None, :] @ (ys[:mu] - x)[None, :, :], axis=0) / sigma**2
            
            C = (1 + cs * ds * delta_h - cs - cmu) * C + \
                cmu * rankmu + (1 + cs * ds * delta_h - cs) * rank1
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Sigma update
            sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / np.sqrt(n) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            x = x_new
        
        return x, f
# EVOLVE-BLOCK-END
