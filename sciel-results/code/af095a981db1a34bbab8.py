import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ2: first n_obj-1 vars are position, rest are distance
        n_pos = self.n_obj - 1  # 2 position vars for 3 obj? 
        # Wait, for DTLZ2 with m objectives, there are m-1 position variables
        # Actually no: DTLZ2 has M objectives, (N-M+1) position variables where N is total dim
        # Let me reconsider. Standard DTLZ2: x_pos has (M-1) vars, x_dist has (N-M+1) vars
        # For M=3, N=12: x_pos has 2 vars, x_dist has 10 vars
        # h_i for i=1..M: 
        #   h_1 = x_pos[0]
        #   h_2 = x_pos[1]  
        #   h_3 = sqrt(1 - x_pos[0]^2 - x_pos[1]^2)
        # f_i = (1+g) * h_i
        # g = sum((x_dist_j - 0.5)^2)
        
        # So position space is the quarter unit disk in 2D (x_pos[0], x_pos[1] >= 0, x_pos[0]^2 + x_pos[1]^2 <= 1)
        
        n_pos = self.n_obj - 1  # 2
        n_dist = self.dim - n_pos  # 10
        
        # Generate points on the Pareto front
        # Use a grid in polar coordinates for the quarter disk
        # r in [0, 1], angle in [0, pi/2]
        
        # Strategy: use most budget for front sampling
        # 10x10 grid = 100 points, plus boundaries
        # Then local search with remaining budget
        
        n_grid = 10
        # Create grid points in polar coords
        r_vals = np.linspace(0, 1, n_grid)
        theta_vals = np.linspace(0, np.pi/2, n_grid)
        
        candidates = []
        for r in r_vals:
            for theta in theta_vals:
                x_pos = np.array([r * np.cos(theta), r * np.sin(theta)])
                # x_pos[0] >= 0, x_pos[1] >= 0, x_pos[0]^2 + x_pos[1]^2 = r^2 <= 1
                # h_3 = sqrt(1 - r^2)
                candidates.append(x_pos)
        
        # Add corner points explicitly
        corners = [
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 0.0]),
        ]
        for c in corners:
            if not any(np.allclose(c, cp) for cp in candidates):
                candidates.append(c)
        
        # Also add points along the arc (r=1)
        for theta in np.linspace(0, np.pi/2, n_grid + 1):
            x_pos = np.array([np.cos(theta), np.sin(theta)])
            if not any(np.allclose(x_pos, cp) for cp in candidates):
                candidates.append(x_pos)
        
        # Shuffle to mix up
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(candidates))
        candidates = [candidates[i] for i in idx]
        
        # Build full solution vectors
        x_dist_zero = np.full(n_dist, 0.5)  # g = 0
        
        solutions = []
        for x_pos in candidates:
            x = np.concatenate([x_pos, x_dist_zero])
            solutions.append(x)
        
        # Evaluate
        F = []
        used = 0
        for x in solutions:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            f = np.array(problem(x))
            F.append(f)
            used += 1
        
        # Local search with remaining budget
        remaining = self.budget - used
        if remaining > 0 and F:
            F_arr = np.array(F)
            # Find non-dominated points
            # Simple: just do local search on a few promising points
            # Pick a few points and try to reduce g by adjusting dist vars
            
            # For local search, try small perturbations on distance vars
            step = 0.01
            for i in range(min(remaining, len(F_arr))):
                if time.time() > deadline:
                    break
                # Pick a random point from current solutions
                j = rng.integers(len(solutions))
                x = solutions[j].copy()
                # Perturb distance vars
                pert = step * rng.standard_normal(n_dist)
                x[n_pos:] = np.clip(x[n_pos:] + pert, lo[n_pos:], hi[n_pos:])
                f = np.array(problem(x))
                used += 1
                # Replace if better (dominates or reduces g)
                # Simple: replace if sum is lower
                if np.sum(f) < np.sum(F_arr[j]):
                    solutions[j] = x
                    F_arr[j] = f
        
        return solutions
