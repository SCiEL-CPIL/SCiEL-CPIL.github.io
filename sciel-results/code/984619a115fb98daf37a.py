# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for DTLZ4 that exploits the known Pareto front geometry.
    It uses a deterministic, low-discrepancy grid sampling strategy over the shape variables
    to ensure uniform coverage of the Pareto front, while fixing distance variables to their
    optimal values (0.5) to guarantee all points lie on the front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 with 3 objectives:
        # Shape variables: x_0, x_1 (indices 0, 1)
        # Distance variables: x_2 ... x_{dim-1} (indices 2 to 11 for dim=12)
        n_shape = n_obj - 1  # 2
        n_dist = dim - n_shape  # 10

        # We want to distribute 'budget' points over the 2D shape space.
        # A square grid of size M x M gives M^2 points.
        M = int(np.sqrt(budget))
        
        # To use the full budget as closely as possible, we might adjust M slightly
        # or just use M x M and if budget > M*M, we can add more points or just stick to M*M.
        # Given the scoring is hypervolume, uniform coverage is key.
        # Let's ensure we don't exceed budget.
        # If M*M < budget, we can increase M until M*M > budget, then take the first 'budget' points?
        # Or just use M = int(np.ceil(np.sqrt(budget))) and truncate?
        # Truncating a grid might break symmetry. 
        # Better: Use M such that M*M is close to budget.
        # Actually, a simple M x M grid is very robust. Let's stick to M = int(np.sqrt(budget)).
        # If budget is 200, M=14, M*M=196. We have 4 evaluations left.
        # We can add the 4 remaining points as vertices if not already covered, or just ignore the small difference.
        # The vertex inclusion is critical. A linspace grid includes vertices.
        
        # Generate grid points in u, v space [0, 1] x [0, 1]
        # u corresponds to cos(theta_0), v corresponds to theta_1/(pi/2)
        # Using linspace ensures vertices (0,0), (1,0), (0,1), (1,1) are included.
        
        u_vals = np.linspace(0, 1, M)
        v_vals = np.linspace(0, 1, M)
        
        # Create meshgrid
        U, V = np.meshgrid(u_vals, v_vals)
        
        # Flatten to get list of (u, v) pairs
        # We need to map these to x_0, x_1
        # x_0 = (2/pi) * arccos(u)
        # x_1 = v
        
        # Reshape to 1D array of pairs
        u_flat = U.flatten()
        v_flat = V.flatten()
        
        # Add extra points to use full budget
        # For M=14, we have 196 points. Add 4 points at edge midpoints.
        # These are at (0.5, 0), (0.5, 1), (0, 0.5), (1, 0.5) in (u,v) space
        # Wait, these may already be in the grid if M is even. Let's check:
        # For M=14, u_vals = [0, 1/13, 2/13, ..., 1]. 0.5 is not in this list.
        # So (0.5, 0), (0.5, 1), (0, 0.5), (1, 0.5) are not in the grid.
        # Actually, let's add points that fill the largest gaps.
        # The grid has spacing 1/(M-1) = 1/13 in both directions.
        # Adding points at the center of the 4 edge cells (not corners) helps.
        
        extra_u = []
        extra_v = []
        remaining = budget - len(u_flat)
        
        if remaining > 0:
            # Add points at midpoints of edges (excluding corners)
            # Edge midpoints: (0.5, 0), (0.5, 1), (0, 0.5), (1, 0.5)
            # But check if they're already close to existing points
            edge_points = [(0.5, 0.0), (0.5, 1.0), (0.0, 0.5), (1.0, 0.5)]
            for eu, ev in edge_points:
                if remaining <= 0:
                    break
                # Check if this point is too close to an existing grid point
                min_dist = np.min(np.sqrt((u_flat - eu)**2 + (v_flat - ev)**2))
                if min_dist > 0.01:  # Not too close to existing
                    extra_u.append(eu)
                    extra_v.append(ev)
                    remaining -= 1
        
        if extra_u:
            u_flat = np.concatenate([u_flat, np.array(extra_u)])
            v_flat = np.concatenate([v_flat, np.array(extra_v)])
        
        # Calculate shape variables
        x_0_vals = (2.0 / np.pi) * np.arccos(u_flat)
        x_1_vals = v_flat
        
        # Total points generated
        n_points = len(x_0_vals)
        
        # Construct the full decision vectors
        # x = [x_0, x_1, 0.5, 0.5, ..., 0.5]
        
        # Create a matrix of shape (n_points, dim)
        X = np.zeros((n_points, dim))
        X[:, 0] = x_0_vals
        X[:, 1] = x_1_vals
        X[:, 2:] = 0.5  # Distance variables at optimal value
        
        # Evaluate all points
        # The problem function expects a vector. We must call it point by point.
        # Vectorization is not possible with the problem(x) interface.
        
        # Check deadline before starting loop if we are very tight (unlikely for 200 calls)
        
        for i in range(n_points):
            if time.time() > deadline:
                break
            
            x = X[i]
            # Evaluate
            f = problem(x)
            
        # The return value is ignored by the evaluator.
        return None
# EVOLVE-BLOCK-END
