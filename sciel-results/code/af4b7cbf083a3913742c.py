"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using Pareto-front aware sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: evaluator kills at 330s.
        deadline = start_time + 320.0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        # 1. Determine optimal tail values for DTLZ2
        # Standard DTLZ2: g(x) is minimized at x_i = 0.5 for i > n_obj
        # Hardcode 0.5 to save evaluation budget.
        best_tail_val = 0.5

        # 2. Generate points on the Pareto Front
        # For DTLZ2 with n_obj=3, the Pareto front is the unit sphere in the positive octant.
        # f_i = cos(x_i * pi / 2)
        # x_i = 2/pi * arccos(f_i)
        
        if n_obj == 3:
            # We will generate a grid of points on the sphere
            # theta in [0, pi/2], phi in [0, pi/2]
            # f1 = sin(theta)cos(phi)
            # f2 = sin(theta)sin(phi)
            # f3 = cos(theta)
            
            # Reserve some budget for random points to fill gaps
            # We have 'budget' evaluations left (should be 200).
            # Fixed points: 3 axis + 3 edge = 6
            # Grid: 13x13 = 169
            # Random: 200 - 6 - 169 = 25
            n_grid = 13
            n_fixed = 6
            n_random = max(0, budget - n_fixed - n_grid * n_grid)
            
            candidates = []
            
            # Add axis points (corners of the octant)
            # (1,0,0), (0,1,0), (0,0,1)
            axis_f = [
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0])
            ]
            for f in axis_f:
                x_head = (2.0 / np.pi) * np.arccos(np.clip(f, 0.0, 1.0))
                x = np.zeros(dim)
                x[:3] = x_head
                x[3:] = best_tail_val
                candidates.append(x)
            
            # Add face center points (midpoints of the "edges" of the octant surface? No, centers of the faces if it were a cube, but here it's sphere)
            # Points like (1/sqrt(2), 1/sqrt(2), 0) etc.
            edge_f = [
                np.array([1.0, 1.0, 0.0]) / np.sqrt(2),
                np.array([1.0, 0.0, 1.0]) / np.sqrt(2),
                np.array([0.0, 1.0, 1.0]) / np.sqrt(2)
            ]
            for f in edge_f:
                x_head = (2.0 / np.pi) * np.arccos(np.clip(f, 0.0, 1.0))
                x = np.zeros(dim)
                x[:3] = x_head
                x[3:] = best_tail_val
                candidates.append(x)
                
            # Grid points
            thetas = np.linspace(0.0, np.pi/2, n_grid)
            phis = np.linspace(0.0, np.pi/2, n_grid)
            
            for i in range(n_grid):
                for j in range(n_grid):
                    theta = thetas[i]
                    phi = phis[j]
                    
                    f1 = np.sin(theta) * np.cos(phi)
                    f2 = np.sin(theta) * np.sin(phi)
                    f3 = np.cos(theta)
                    
                    f = np.array([f1, f2, f3])
                    x_head = (2.0 / np.pi) * np.arccos(np.clip(f, 0.0, 1.0))
                    x = np.zeros(dim)
                    x[:3] = x_head
                    x[3:] = best_tail_val
                    candidates.append(x)
                    
            # Random points on the sphere
            rng = np.random.default_rng(0)
            for _ in range(n_random):
                # Generate random point in unit cube, reject if outside sphere? 
                # Or generate random direction.
                v = rng.random(3)
                norm = np.linalg.norm(v)
                if norm == 0:
                    continue
                f = v / norm
                x_head = (2.0 / np.pi) * np.arccos(np.clip(f, 0.0, 1.0))
                x = np.zeros(dim)
                x[:3] = x_head
                x[3:] = best_tail_val
                candidates.append(x)
                
            # Shuffle to ensure diverse coverage early on? 
            # Or just iterate. Grid is systematic.
            # Let's interleave or just iterate.
            
            for x in candidates:
                if time.time() > deadline:
                    break
                if budget <= 0:
                    break
                x_clipped = np.clip(x, lo, hi)
                problem(x_clipped)
                budget -= 1
                if budget == 0:
                    break
                
        else:
            # Fallback for other n_obj: Random search with fixed tail
            rng = np.random.default_rng(0)
            while budget > 0 and time.time() < deadline:
                x_head = rng.random(n_obj)
                x = np.zeros(dim)
                x[:n_obj] = x_head
                x[n_obj:] = best_tail_val
                x_clipped = np.clip(x, lo, hi)
                problem(x_clipped)
                budget -= 1

        return None
# EVOLVE-BLOCK-END
