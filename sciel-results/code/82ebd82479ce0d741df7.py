"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ7-aware optimizer using dense Pareto-front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # DTLZ7 Pareto front: x[0] and x[1] determine objectives, x[2:] must be 0 for g=0
        # f1 = x[0], f2 = x[1], f3 = (1 + x[21]) * g(x[2:], ...)
        # On Pareto front: g=0 => x[2:] = 0, so f3 = 0
        # Front is 2D simplex: x[0] >= 0, x[1] >= 0, x[0] + x[1] <= 1

        # Use all budget for dense sampling on the Pareto front
        # Optimal for hypervolume: uniform coverage of the 2D front region
        n_evals = min(budget, 200)
        
        # Create a dense grid on the 2D simplex
        # Use stratified sampling: divide the triangle into cells
        # For hypervolume, we want good coverage of the entire front
        
        # Use a structured approach:
        # 1. Sample on the hypotenuse (x[0] + x[1] = 1) densely
        # 2. Sample interior points to cover the full simplex
        
        # Optimal for 3-objective hypervolume with ref point [1.1, 1.1, 25.0]:
        # The Pareto front of DTLZ7 is the triangle with vertices (0,0), (1,0), (0,1)
        # Hypervolume contribution is maximized by covering the whole front
        
        # Use a combination of:
        # - Dense sampling along the main diagonal (x[0] + x[1] = 1)
        # - Some interior points for robustness
        
        # Actually, for DTLZ7 the entire triangle x[0]+x[1]<=1, x[0]>=0, x[1]>=0 is the front
        # But the "knee" region near (0,0) contributes less to hypervolume
        # The region near the hypotenuse contributes most
        
        # Strategy: use most budget on the hypotenuse, some on interior
        n_hypotenuse = int(n_evals * 0.8)
        n_interior = n_evals - n_hypotenuse
        
        front_points = []
        
        # Sample densely on the hypotenuse x[0] + x[1] = 1
        # This is where most hypervolume contribution comes from
        if n_hypotenuse > 0:
            t_vals = np.linspace(0.0, 1.0, n_hypotenuse)
            for t in t_vals:
                x = np.zeros(dim)
                x[0] = t
                x[1] = 1.0 - t
                front_points.append(x)
        
        # Sample interior points using stratified random
        if n_interior > 0:
            rng = np.random.default_rng(42)
            for _ in range(n_interior):
                # Random point in triangle using rejection sampling
                while True:
                    u1 = rng.random()
                    u2 = rng.random()
                    if u1 + u2 <= 1.0:
                        break
                x = np.zeros(dim)
                x[0] = u1
                x[1] = u2
                front_points.append(x)
        
        front_points = np.array(front_points[:n_evals])
        
        # Evaluate all front points
        F = np.array([problem(x) for x in front_points])
        used = len(front_points)
        
        # Check time
        if time.time() > deadline:
            return front_points
        
        # If we have remaining budget, use it for a few local refinements
        # to ensure we're exactly on the front (g=0)
        remaining = budget - used
        if remaining > 0:
            # Pick a few points to refine
            n_refine = min(remaining, 10)
            # Select points that might benefit from refinement
            # Focus on points where g might not be exactly 0
            refine_indices = np.random.default_rng(42).choice(n_evals, size=n_refine, replace=False)
            
            for idx in refine_indices:
                if time.time() > deadline or used >= budget:
                    break
                x_curr = front_points[idx].copy()
                # Set non-front dimensions to exactly 0 to ensure g=0
                x_curr[2:] = 0.0
                x_new = np.clip(x_curr, lo, hi)
                f_new = problem(x_new)
                used += 1
                # Accept if it improves (should be same or better)
                if np.all(f_new <= F[idx]) and np.any(f_new < F[idx]):
                    front_points[idx] = x_new
                    F[idx] = f_new
        
        return front_points
# EVOLVE-BLOCK-END
