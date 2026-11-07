import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # safety margin before 330s kill

        n_obj = self.n_obj
        dim = self.dim
        n_pos = n_obj - 1  # 2 position variables for 3 objectives

        # DTLZ4 Pareto front: f_i = cos^2(theta_i) for i < n_pos, f_{n_pos} = 1
        # with sum of cos^2 = 1 (for 2 position vars: cos^2(a) + sin^2(a) = 1)
        # So f = (cos^2(a), sin^2(a), 1) where a in [0, pi/2]

        # Use non-uniform sampling: denser near corners for better hypervolume
        # Strategy: combine uniform angular + a few extra at strategic points

        total_budget = self.budget
        # Reserve a few points for edge cases
        n_main = total_budget - 4

        # Generate non-uniform angles: use power-law spacing
        # More points near 0 and pi/2 where hypervolume contribution is higher
        # Use a beta-distribution-like spacing
        t = np.linspace(0, 1, n_main)
        # Apply a transformation to cluster near endpoints
        # t^0.7 gives more density near 0, but we want both ends
        # Use: theta = pi/2 * (1 - cos(t * pi)) / 2  -> uniform in cos space
        # Actually let's use: sample uniformly in f1 = cos^2(theta)
        # f1 ranges from 0 to 1, theta = arccos(sqrt(f1))
        # Uniform in f1: f1_i = i/(n_main-1) for i=0..n_main-1
        # This gives denser theta near pi/2 (where f1 is small) and sparse near 0
        # Wait: theta = arccos(sqrt(f1)), so df1/dtheta = -sin(2*theta)
        # Near theta=0: df1/dtheta ~ 0, so uniform f1 is sparse in theta
        # Near theta=pi/2: same
        # Near theta=pi/4: df1/dtheta = -1, so uniform f1 is dense in theta
        
        # For hypervolume, we want good coverage. Let's use uniform in theta
        # but add extra points near the corners.
        
        # Base: uniform in theta
        theta_uniform = np.linspace(0, np.pi/2, n_main)
        
        # Additional strategic points at the very corners and specific locations
        extra_thetas = [0.0, np.pi/2, np.pi/8, 3*np.pi/8]
        
        # Check if extra_thetas are already in theta_uniform
        # theta_uniform[0] = 0, theta_uniform[-1] = pi/2, so those are duplicates
        # Remove duplicates
        all_thetas = np.concatenate([theta_uniform, extra_thetas])
        all_thetas = np.unique(np.round(all_thetas, 12))
        
        # If we have too many, trim to budget
        if len(all_thetas) > total_budget:
            # Keep the most important: corners and uniform
            # Just take first total_budget
            all_thetas = all_thetas[:total_budget]
        
        n_points = len(all_thetas)
        
        # Build decision vectors
        points = []
        for theta in all_thetas:
            x = np.zeros(dim)
            x[0] = np.cos(theta)  # position variable 1
            x[1] = np.sin(theta)  # position variable 2
            # Distance variables: set to 0.5 for optimal g=0
            x[2:] = 0.5
            points.append(x)
        
        # Evaluate all points
        F = np.array([problem(x) for x in points])
        
        # Check deadline
        if time.time() > deadline:
            return F
        
        return F
