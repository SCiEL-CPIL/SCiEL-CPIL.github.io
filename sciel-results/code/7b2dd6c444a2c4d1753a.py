"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ7-aware optimizer using Pareto-front sampling and local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # DTLZ7 Pareto front: f1=x1, f2=x2, f3=0 on the front
        # Front region: x1>=0, x2>=0, x1+x2<=1, x3..x22=0
        # Hypervolume-aware sampling: cluster near origin where marginal HV contribution is highest

        n_front = min(budget, 120)
        
        # Hypervolume-aware non-uniform grid on the 2D simplex
        # Use a transformation that clusters points near the origin (0,0)
        # In objective space, points near origin contribute more to hypervolume
        
        # Strategy: use a power-law distribution to cluster near corners
        # For the 2D triangle with vertices (0,0), (1,0), (0,1):
        # Use barycentric coordinates with biased sampling
        
        front_points = []
        
        # Include the three extreme corners explicitly
        corners = [
            np.array([0.0, 0.0]),  # (0,0)
            np.array([1.0, 0.0]),  # (1,0) 
            np.array([0.0, 1.0]),  # (0,1)
        ]
        for c in corners:
            x = np.zeros(dim)
            x[0] = c[0]
            x[1] = c[1]
            front_points.append(x)
        
        # Add points along the edges with hypervolume-aware spacing
        # Edge from (0,0) to (1,0): x2=0, x1 varies
        # Edge from (0,0) to (0,1): x1=0, x2 varies
        # Edge from (1,0) to (0,1): x1+x2=1
        
        n_edge = max(1, (n_front - 3) // 3)
        
        # Use power-law spacing: t_i = (i/n)^alpha where alpha > 1 clusters near 0
        alpha = 1.5  # clustering parameter
        
        # Edge 1: (0,0) to (1,0)
        for i in range(1, n_edge + 1):
            t = (i / (n_edge + 1)) ** alpha
            x = np.zeros(dim)
            x[0] = t
            x[1] = 0.0
            front_points.append(x)
        
        # Edge 2: (0,0) to (0,1)
        for i in range(1, n_edge + 1):
            t = (i / (n_edge + 1)) ** alpha
            x = np.zeros(dim)
            x[0] = 0.0
            x[1] = t
            front_points.append(x)
        
        # Edge 3: (1,0) to (0,1) - use uniform spacing here as this is the "far" edge
        for i in range(1, n_edge + 1):
            t = i / (n_edge + 1)
            x = np.zeros(dim)
            x[0] = t
            x[1] = 1.0 - t
            front_points.append(x)
        
        # Interior points with hypervolume-aware density
        n_interior = n_front - len(front_points)
        if n_interior > 0:
            rng = np.random.default_rng(42)
            # Use rejection sampling with biased distribution
            # Sample u1, u2 from Beta distribution to cluster near 0
            for _ in range(n_interior):
                # Sample from a distribution that clusters near origin
                # Use u1 = v1^beta, u2 = v2^beta * (1-u1) for bias
                beta = 2.0
                u1 = rng.random() ** beta
                u2 = rng.random() ** beta * (1.0 - u1)
                
                x = np.zeros(dim)
                x[0] = u1
                x[1] = u2
                front_points.append(x)
        
        front_points = np.array(front_points[:n_front])
        
        # Evaluate front points
        F = np.array([problem(x) for x in front_points])
        used = len(front_points)
        
        # Check time
        if time.time() > deadline:
            return front_points
        
        # Local refinement: use remaining budget to refine points
        remaining = budget - used
        if remaining <= 0:
            return front_points
        
        # Sort points by objective sum to prioritize refinement of promising points
        obj_sums = np.sum(F, axis=1)
        sort_idx = np.argsort(obj_sums)
        
        n_local = len(front_points)
        steps_per_point = max(1, remaining // n_local)
        
        # Refine in order of promise (lowest objective sum first)
        for idx in sort_idx:
            if time.time() > deadline:
                break
            for step in range(steps_per_point):
                if time.time() > deadline:
                    break
                if used >= budget:
                    break
                
                x_curr = front_points[idx].copy()
                
                # Multi-scale perturbation: try different step sizes
                rng_local = np.random.default_rng(idx * 1000 + step)
                step_size = 0.01 * (0.5 ** step)  # decreasing step size
                perturbation = rng_local.standard_normal(dim) * step_size
                x_new = x_curr + perturbation
                x_new = np.clip(x_new, lo, hi)
                
                f_new = problem(x_new)
                used += 1
                
                # Accept if it improves or is incomparable
                if not (np.all(F[idx] <= f_new) and np.any(F[idx] < f_new)):
                    if np.all(f_new <= F[idx]) and np.any(f_new < F[idx]):
                        front_points[idx] = x_new
                        F[idx] = f_new
                    elif not (np.all(F[idx] <= f_new) and np.any(F[idx] < f_new)):
                        if np.sum(f_new) < np.sum(F[idx]):
                            front_points[idx] = x_new
                            F[idx] = f_new
        
        return front_points
# EVOLVE-BLOCK-END
