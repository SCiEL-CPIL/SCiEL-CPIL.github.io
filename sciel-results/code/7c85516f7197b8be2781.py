"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Sobol global search followed by L-BFGS-B local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _sobol_points(self, n, dim, rng):
        """Generate n quasi-random points in [-5, 5]^dim using Halton-like sequence."""
        points = np.zeros((n, dim))
        # Use a simple Halton sequence for low-discrepancy sampling
        for d in range(dim):
            base = d + 2  # prime bases: 2, 3, 5, 7, 11
            for i in range(n):
                f = 1.0
                result = 0.0
                idx = i + 1
                while idx > 0:
                    f /= base
                    result += f * (idx % base)
                    idx //= base
                points[i, d] = -5.0 + 10.0 * result
        # Add small jitter to avoid exact duplicates
        points += rng.uniform(0, 0.01, size=points.shape)
        points = np.clip(points, -5, 5)
        return points

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 300s limit, keep 20s margin
        
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Global search with Sobol/Halton points
        n_global = min(100, budget // 5)
        global_points = self._sobol_points(n_global, dim, rng)
        
        # Also add some random points for diversity
        n_random = min(20, budget // 10)
        random_points = rng.uniform(-5, 5, size=(n_random, dim))
        all_points = np.vstack([global_points, random_points])
        
        best_x = None
        best_f = np.inf
        
        for i in range(len(all_points)):
            if time.time() > deadline - 60:
                break
            x = all_points[i]
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Also evaluate origin and a few strategic points
        strategic_points = [
            np.zeros(dim),
            rng.uniform(-5, 5, dim),
            rng.uniform(-3, 3, dim),
        ]
        for sp in strategic_points:
            if time.time() > deadline - 60:
                break
            fx = func(sp)
            if fx < best_f:
                best_f = fx
                best_x = sp.copy()
        
        calls_made = len(all_points) + len(strategic_points)
        
        # Phase 2: Local refinement with L-BFGS-B
        remaining_budget = budget - calls_made
        if remaining_budget > 10 and best_x is not None:
            try:
                from scipy.optimize import minimize
                result = minimize(
                    func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': min(200, remaining_budget), 'maxls': 50}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
        
        # Phase 3: If we have budget left, do a second local refinement from best
        elapsed = time.time() - start_time
        if elapsed < 200 and best_x is not None:
            # Try a few more local refinements from slightly perturbed best
            for _ in range(3):
                if time.time() > deadline - 30:
                    break
                perturbation = rng.normal(0, 0.5, dim)
                x_pert = np.clip(best_x + perturbation, -5, 5)
                try:
                    from scipy.optimize import minimize
                    result = minimize(
                        func,
                        x_pert,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 100, 'maxls': 50}
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except Exception:
                    pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
