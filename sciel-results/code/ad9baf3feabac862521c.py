"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-strategy optimizer for BBOB f21 (Gallagher 101 peaks).
    
    Strategy:
    1. Global exploration with Halton sequence + random sampling to find good basins.
    2. Local refinement around the best found point using adaptive step size.
    3. Deadline guard to ensure we finish within time limits.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    @staticmethod
    def _halton_sequence(n, dim, base=None):
        """Generate n Halton sequence points in [0,1)^dim."""
        if base is None:
            # Use first 'dim' primes
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            base = primes[:dim]
        
        points = np.zeros((n, dim))
        for d, p in enumerate(base):
            inv_p = 1.0 / p
            f = inv_p
            for i in range(1, n + 1):
                while f >= 1.0:
                    f = (f * p) % 1.0
                    f /= p
                points[i - 1, d] = f
                f = (f * p) % 1.0
                f /= p
        return points

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0  # Safety margin before 300s kill
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        # Track best
        best_x = None
        best_f = np.inf
        
        def eval_point(x):
            nonlocal best_x, best_f
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx
        
        # Phase 1: Global exploration
        # Use a combination of Halton sequence and random sampling
        n_global = min(100, budget // 5)  # Use about 20% budget for global search
        
        # Halton points
        halton_pts = self._halton_sequence(n_global // 2, dim)
        # Map to [-5, 5]^dim
        halton_pts = halton_pts * 10.0 - 5.0
        
        # Random points
        n_random = n_global - len(halton_pts)
        random_pts = rng.uniform(-5, 5, (n_random, dim))
        
        # Evaluate global points
        for x in halton_pts:
            if time.time() > deadline:
                break
            eval_point(x)
        
        for x in random_pts:
            if time.time() > deadline:
                break
            eval_point(x)
        
        if best_x is None:
            # Fallback: evaluate one point
            x0 = np.zeros(dim)
            eval_point(x0)
            return best_x, best_f
        
        # Phase 2: Local refinement around best point
        # Use adaptive step size local search
        remaining_budget = budget - n_global
        sigma = 1.0  # Initial step size
        min_sigma = 1e-4
        max_sigma = 5.0
        
        # Try multiple local refinements from top few points
        # But we only tracked the best, so let's just refine the best
        x = best_x.copy()
        fx = best_f
        
        # Adaptive local search
        for i in range(remaining_budget):
            if time.time() > deadline:
                break
            
            # Generate candidate
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            
            if fy < fx:
                x, fx = y, fy
                best_x, best_f = x.copy(), fx
                sigma = min(sigma * 1.2, max_sigma)
            else:
                sigma = max(sigma * 0.85, min_sigma)
            
            # Occasionally try a larger jump to escape local minima
            if i % 50 == 0 and i > 0:
                jump = x + 2.0 * rng.standard_normal(dim)
                jump = np.clip(jump, -5, 5)
                fj = func(jump)
                if fj < fx:
                    x, fx = jump, fj
                    best_x, best_f = x.copy(), fx
                    sigma = 2.0
        
        return best_x, best_f
# EVOLVE-BLOCK-END
