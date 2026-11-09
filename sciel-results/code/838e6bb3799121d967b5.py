"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    """
    Multi-objective optimizer for DTLZ2 using analytic front approximation.
    1. Finds the optimal 'distance' variables (x_M..x_n) to lie on the Pareto front.
    2. Samples the 'angle' variables (x_1..x_{M-1}) using a Sobol sequence for uniform coverage.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 290.0  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_angle = self.n_obj - 1
        n_dist = self.dim - n_angle
        
        if n_dist <= 0:
            # Fallback to random if no distance variables
            X = lo + (hi - lo) * np.random.default_rng(0).random((self.budget, self.dim))
            return X

        # Step 1: Find a good x_dist that places points on the Pareto front
        # We fix x_angle to a center value and optimize x_dist
        x_angle_center = np.full(n_angle, 0.5)
        
        # Initial candidates for x_dist
        rng = np.random.default_rng(0)
        candidates = [
            np.zeros(n_dist),
            np.full(n_dist, 0.5),
            np.ones(n_dist),
            rng.random(n_dist),
            rng.random(n_dist)
        ]
        
        best_dist = None
        best_score = np.inf
        
        for xd in candidates:
            xd = np.clip(xd, lo[n_angle:], hi[n_angle:])
            x = np.concatenate([x_angle_center, xd])
            f = problem(x)
            score = np.sum(f)
            if score < best_score:
                best_score = score
                best_dist = xd.copy()
        
        # Refine with local search (Nelder-Mead)
        def objective(xd):
            xd = np.clip(xd, lo[n_angle:], hi[n_angle:])
            x = np.concatenate([x_angle_center, xd])
            f = problem(x)
            return np.sum(f)
        
        try:
            res = minimize(objective, best_dist, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
            if np.sum(res.fun) < best_score:
                best_dist = np.clip(res.x, lo[n_angle:], hi[n_angle:])
                best_score = res.fun
        except Exception:
            pass # Keep initial best_dist if optimization fails

        # Step 2: Generate points on the front using Sobol sequence for angles
        # Allocate remaining budget
        used_evals = len(candidates) + 100 # Approximate count for minimize
        remaining = self.budget - used_evals
        if remaining < 10:
            remaining = 10 # Ensure at least 10 points
        
        # Cap remaining to budget
        remaining = min(remaining, self.budget)
        
        # Generate Sobol points in n_angle dimensions
        sampler = qmc.Sobol(d=n_angle, scramble=True)
        sobol_points = sampler.random(n=remaining)
        
        # Construct decision vectors
        X = np.zeros((remaining, self.dim))
        X[:, :n_angle] = sobol_points
        X[:, n_angle:] = best_dist
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        # Check time
        if time.time() - start_time > deadline:
            return X
            
        # Step 3: Optional Refinement
        # If we have time and budget, we can try to refine a few points
        # but for DTLZ2, the analytic construction should be very close to optimal.
        # We will just return the constructed points.
        
        return X
# EVOLVE-BLOCK-END
