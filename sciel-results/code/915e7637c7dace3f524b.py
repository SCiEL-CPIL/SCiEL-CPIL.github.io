"""Hybrid Multi-Start Nelder-Mead Optimizer for BBOB f20 (Schwefel)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

class Optimizer:
    """
    Optimizer using a multi-start strategy with Nelder-Mead local refinement.
    Designed for the BBOB f20 (Schwefel) function on [-5, 5]^5.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin
        calls_made = 0
        best_x = None
        best_f = np.inf

        def safe_call(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = np.array(x)
            return fx

        # Phase 1: Diverse Initial Exploration
        # The prompt states the minimizer lies in [-4, 4]^5.
        # We will sample this region densely.
        
        initial_points = []
        
        # 1. Origin
        initial_points.append(np.zeros(self.dim))
        
        # 2. Vertices of the hypercube [-4, 4]^5
        # 2^5 = 32 vertices
        for i in range(2 ** self.dim):
            p = np.array([(i >> j) & 1 for j in range(self.dim)], dtype=float)
            # Map {0, 1} to {-1, 1} then scale to [-4, 4]
            p = (p * 2 - 1) * 4.0
            initial_points.append(p)
            
        # 3. Random points in [-4, 4]^5
        rng = np.random.default_rng(42)
        for _ in range(18):
            initial_points.append(rng.uniform(-4.0, 4.0, self.dim))
            
        # 4. Points near the theoretical Schwefel optimum (4.21) just in case
        # Although prompt says [-4,4], 4.21 is close.
        schwefel_opt = 4.209687
        for scale in [0.8, 0.9, 1.0, 1.1]:
             p = np.full(self.dim, schwefel_opt * scale)
             p = np.clip(p, -5, 5)
             initial_points.append(p)

        # Evaluate initial points
        evaluated_points = []
        for p in initial_points:
            if time.time() > deadline or calls_made >= self.budget:
                break
            fx = safe_call(p)
            evaluated_points.append((p, fx))

        if best_x is None:
            # Fallback if no calls were made (should not happen)
            best_x = np.zeros(self.dim)
            best_f = safe_call(best_x)

        # Sort by function value to get the best starting points for local search
        evaluated_points.sort(key=lambda x: x[1])
        
        # Phase 2: Local Refinement using Nelder-Mead
        # We will run Nelder-Mead on the top K best initial points.
        # K is chosen based on remaining budget.
        # Nelder-Mead in D dims uses approx D+1 function calls per iteration.
        # We limit iterations to ensure we don't exceed budget.
        
        remaining_budget = self.budget - calls_made
        if remaining_budget > 10:
            # Allocate budget to top 3 candidates
            # Let's give each candidate up to (remaining_budget // 3) calls
            calls_per_candidate = remaining_budget // 3
            # Ensure at least a few iterations
            if calls_per_candidate < 10:
                calls_per_candidate = min(10, remaining_budget)
            
            top_k = min(3, len(evaluated_points))
            
            for i in range(top_k):
                if time.time() > deadline or calls_made >= self.budget:
                    break
                
                start_point = evaluated_points[i][0]
                
                # Define the objective function for minimize
                # We need to clip inputs to [-5, 5] as per task description
                def obj(x):
                    x_clipped = np.clip(x, -5.0, 5.0)
                    return safe_call(x_clipped)
                
                try:
                    # Nelder-Mead options
                    # maxiter: maximum number of iterations
                    # Each iteration involves roughly 1 function evaluation (updating simplex)
                    # So maxiter ~ max calls
                    res = minimize(
                        obj, 
                        start_point, 
                        method='Nelder-Mead', 
                        options={
                            'maxiter': calls_per_candidate,
                            'xatol': 1e-6,
                            'fatol': 1e-6
                        }
                    )
                    # safe_call updates best_x, best_f automatically
                except Exception:
                    pass

        # Phase 3: Final Polish / Random Perturbations
        # If we have calls left, do a few random steps around the best point
        if best_x is not None and calls_made < self.budget and time.time() < deadline:
            sigma = 0.5
            while calls_made < self.budget and time.time() < deadline:
                # Adaptive sigma
                x_pert = np.clip(best_x + sigma * rng.standard_normal(self.dim), -5, 5)
                fx = safe_call(x_pert)
                if fx < best_f:
                    # Accept, shrink sigma
                    sigma *= 0.7
                else:
                    # Reject, expand sigma slightly to escape local min
                    sigma *= 1.2
                    if sigma > 2.0:
                        sigma = 2.0

        return best_x, best_f
# EVOLVE-BLOCK-END
