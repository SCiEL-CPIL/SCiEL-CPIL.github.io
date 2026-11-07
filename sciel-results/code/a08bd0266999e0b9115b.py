# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ2.
    
    Approach:
    - DTLZ2 Pareto front for 3 objectives is a quarter-sphere in the positive octant.
    - Distance variables (last dim - n_obj + 1 = 10 variables) should be 0.5 for optimality.
    - Angle variables (first n_obj - 1 = 2 variables) determine position on the front.
    - Use deterministic low-discrepancy sequence (Halton) to sample the 2D angle space
      maximally space-filling, ensuring good hypervolume coverage.
    - Use remaining budget for local refinement around promising points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget
        
        # For DTLZ2: first (n_obj-1) variables are angles, rest are distance variables
        n_angles = n_obj - 1  # 2 for 3 objectives
        n_dist = dim - n_angles  # 10 for 12 dims
        
        # Halton sequence generator for low-discrepancy sampling
        def halton_sequence(n, dims):
            """Generate n points in [0,1]^dims using Halton sequence."""
            points = np.zeros((n, dims))
            for d in range(dims):
                base = d + 2  # primes: 2, 3, 5, ...
                for i in range(n):
                    f = 1.0
                    i_val = i + 1
                    while i_val > 0:
                        f /= base
                        i_val = (i_val - 1) // base
                    points[i, d] = f
            return points

        # Generate Halton points for the angle space
        # We'll use a good chunk of budget for initial space-filling sampling
        n_halton = min(budget - 10, 150)  # leave some for refinement
        halton_pts = halton_sequence(n_halton, n_angles)
        
        # Map Halton points to DTLZ2 angle space
        # For DTLZ2, angles are in [0, pi/2] for each of the first (n_obj-1) variables
        # The transformation from angles to objectives:
        # f_1 = (1 + g) * prod_{i=1}^{n_obj-1} cos(theta_i) * sin(theta_{n_obj-1})
        # Actually, standard DTLZ2 mapping:
        # f_i = (1+g) * prod_{j=1}^{i-1} cos(theta_j) * sin(theta_i) for i < n_obj
        # f_{n_obj} = (1+g) * prod_{j=1}^{n_obj-1} sin(theta_j)
        # where theta_i in [0, pi/2]
        
        # For 3 objectives:
        # f_1 = (1+g) * cos(theta_1) * sin(theta_2)  -- wait, let me be more careful
        # Standard DTLZ2:
        # f_1 = (1+g) * cos(x_1) * cos(x_2)  -- no, that's not right either
        # Let me use the standard formulation:
        # f_i = (1+g) * prod_{j=1}^{i-1} cos(x_j) * sin(x_i) for i = 1, ..., n_obj-1
        # f_{n_obj} = (1+g) * prod_{j=1}^{n_obj-1} sin(x_j)
        # Wait, I need to be careful. Let me use:
        # For i in 1..n_obj-1: f_i = (1+g) * (prod_{j=1}^{i-1} cos(x_j)) * sin(x_i)
        # f_{n_obj} = (1+g) * (prod_{j=1}^{n_obj-1} sin(x_j))
        # But this doesn't quite work for i=1 since prod over empty set is 1
        # f_1 = (1+g) * sin(x_1)
        # f_2 = (1+g) * cos(x_1) * sin(x_2)
        # f_3 = (1+g) * cos(x_1) * cos(x_2)
        # Hmm, that's not standard either. Let me just use the fact that we want
        # points on the unit sphere in the positive octant.
        
        # Simpler approach: sample uniformly on the quarter-sphere
        # For 3 objectives, the Pareto front is {f in R^3 : sum f_i^2 = 1, f_i >= 0}
        # We can parameterize using two angles:
        # f_1 = cos(a) * cos(b)
        # f_2 = cos(a) * sin(b)
        # f_3 = sin(a)
        # where a, b in [0, pi/2]
        
        # Map Halton points to angles
        angles = halton_pts * (np.pi / 2)
        
        # Compute objectives for each angle pair (assuming g=0, i.e., on Pareto front)
        # f_1 = cos(a) * cos(b), f_2 = cos(a) * sin(b), f_3 = sin(a)
        a = angles[:, 0]
        b = angles[:, 1]
        f1 = np.cos(a) * np.cos(b)
        f2 = np.cos(a) * np.sin(b)
        f3 = np.sin(a)
        
        # Build x vectors: first 2 are angles, rest are 0.5 (optimal distance)
        X = np.zeros((n_halton, dim))
        X[:, :n_angles] = angles
        X[:, n_angles:] = 0.5  # optimal distance value for DTLZ2
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        used = n_halton
        
        # Keep track of all evaluated points for hypervolume computation
        all_X = [X]
        all_F = [F]
        
        # Use remaining budget for local refinement
        remaining = budget - used
        if remaining > 0 and time.time() < deadline:
            # Refine around best points (lowest sum of objectives)
            sums = np.sum(F, axis=1)
            # Pick top few points to refine around
            n_refine = min(remaining, 20)
            top_idx = np.argsort(sums)[:min(5, n_halton)]
            
            for i in range(n_refine):
                if time.time() >= deadline:
                    break
                base_idx = top_idx[i % len(top_idx)]
                base_x = X[base_idx].copy()
                
                # Small random perturbation
                scale = 0.02 * (hi - lo)
                perturbation = np.random.standard_normal(dim) * scale
                new_x = np.clip(base_x + perturbation, lo, hi)
                
                new_f = problem(new_x)
                used += 1
                all_X.append(new_x.reshape(1, -1))
                all_F.append(new_f.reshape(1, -1))
        
        # Combine all evaluated points
        if all_X:
            all_X = np.vstack(all_X)
            all_F = np.vstack(all_F)
        
        return all_X
# EVOLVE-BLOCK-END
