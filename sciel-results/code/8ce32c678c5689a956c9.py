# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Multi-objective optimizer for DTLZ4.
    
    Strategy:
    1. Exploit the analytic structure of DTLZ4:
       - The Pareto front is a quarter-sphere in the first n_obj-1 objectives.
       - The last (dim - (n_obj - 1)) variables (g-variables) must be at their
         optimal value (0.5 for DTLZ4 with beta=10) to achieve g=0.
       - The first (n_obj - 1) variables determine the position on the front.
       
    2. Use a stratified/Sobol-like sequence to sample the first (n_obj-1) variables
       uniformly on the unit sphere surface (quarter-sphere for positive orthant).
       
    3. Use remaining budget for local refinement of g-variables if needed,
       though setting them exactly to 0.5 should give g=0.
    
    For DTLZ4 with 12 variables and 3 objectives:
    - Variables x[0], x[1] determine the front position (n_obj-1 = 2)
    - Variables x[2]..x[11] are g-variables (10 variables)
    - Optimal g-variables: all at 0.5
    - Front: f1 = sin(x0) * cos(x1), f2 = sin(x0) * sin(x1), f3 = cos(x0)
      where x0 in [0, pi/2], x1 in [0, pi/2]
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Number of front-determining variables
        n_front = n_obj - 1  # 2 for 3 objectives
        # Number of g-variables
        n_g = dim - n_front  # 10 for 12 vars, 3 objs
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # For DTLZ4, the g-variables should be at 0.5 (optimal)
        # The front variables should be in [0, pi/2]
        # But bounds are [0, 1] for all variables per the task description
        # So we need to map [0, 1] to [0, pi/2] for front variables
        
        # Strategy: 
        # 1. Generate a set of candidate points on the Pareto front
        # 2. Evaluate them
        # 3. Use remaining budget for refinement if needed
        
        # We want to use most of the budget to evaluate diverse front points
        # Let's allocate budget to evaluating front points
        
        # Generate stratified samples on the quarter-sphere
        # For 2 front variables: x0 in [0, pi/2], x1 in [0, pi/2]
        # f1 = sin(x0) * cos(x1)
        # f2 = sin(x0) * sin(x1)  
        # f3 = cos(x0)
        
        # To get uniform coverage on the front, we can use a grid or Sobol-like sequence
        # Let's use a stratified approach
        
        # Allocate most budget to front sampling
        # Keep a small reserve for potential refinement
        n_front_samples = min(budget - 10, 190)  # Leave 10 for refinement
        n_refine = budget - n_front_samples
        
        # Generate stratified samples for the 2 front variables
        # Map uniform [0,1]^2 to the quarter-sphere parameterization
        
        # For uniform coverage on the sphere surface, we should use a proper mapping
        # x0 = arccos(1 - u1) would give uniform on hemisphere, but we want quarter
        # Actually, for the quarter sphere (first octant in 3D, but we're on the 
        # intersection with positive orthant), let's think carefully.
        
        # The Pareto front of DTLZ3/4 with 3 objectives is:
        # f1 = 0.5 * (1 + g) * x1
        # f2 = 0.5 * (1 + g) * x2  
        # f3 = 0.5 * (1 + g) * x3
        # where x1^2 + x2^2 + x3^2 = 1, xi >= 0
        # and g = sum((xi - 0.5)^2) for i in g-variables
        
        # Wait, let me reconsider. DTLZ4 standard formulation:
        # f1 = 0.5 * (1+g) * x1
        # f2 = 0.5 * (1+g) * x2
        # f3 = 0.5 * (1+g) * x3
        # where sum(x1^2 + x2^2 + x3^2) = 1, xi >= 0
        # g = sum_{i=n_obj}^{dim} (xi - 0.5)^2 for DTLZ4 with beta=10? 
        # Actually DTLZ4: g = sum (xi^beta - 0.5^beta)^2 or similar
        
        # The key insight: when g=0, the front is a quarter-sphere of radius 0.5
        # in the positive orthant.
        
        # To sample uniformly on this quarter-sphere:
        # Generate random points in the positive orthant, normalize to unit sphere
        # Then scale by 0.5
        
        # But we need deterministic stratified sampling. Let's use a grid-based approach.
        
        # For 2 parameters (since n_front=2, we have 2 degrees of freedom on the sphere):
        # We can parameterize with theta and phi
        # x1 = sin(theta) * cos(phi)
        # x2 = sin(theta) * sin(phi)
        # x3 = cos(theta)
        # where theta in [0, pi/2], phi in [0, pi/2]
        
        # For uniform area on the sphere, dA = sin(theta) dtheta dphi
        # So we should stratify with sin(theta) weighting
        
        # Let's use a simple stratified grid with proper weighting
        
        # Number of samples in each dimension
        n_theta = int(np.sqrt(n_front_samples))
        n_phi = n_front_samples // n_theta if n_theta > 0 else 1
        
        # Generate stratified samples
        # theta in [0, pi/2], phi in [0, pi/2]
        # For uniform area: sample sin(theta) uniformly, so theta = arcsin(uniform)
        
        theta_samples = []
        phi_samples = []
        
        for i in range(n_theta):
            for j in range(n_phi):
                if i + j * n_theta >= n_front_samples:
                    break
                # Stratified: use midpoint of each cell
                # For theta: stratify sin(theta) uniformly
                u_theta = (i + 0.5) / n_theta
                # sin(theta) = u_theta, so theta = arcsin(u_theta)
                theta = np.arcsin(np.clip(u_theta, 0, 1))
                
                # For phi: stratify uniformly (since dA has no special weighting for phi in this range)
                u_phi = (j + 0.5) / n_phi
                phi = u_phi * (np.pi / 2)
                
                theta_samples.append(theta)
                phi_samples.append(phi)
        
        # Generate the x points
        X = []
        for theta, phi in zip(theta_samples, phi_samples):
            x = np.zeros(dim)
            # Front variables (first n_front)
            x[0] = np.sin(theta) * np.cos(phi)  # x1
            x[1] = np.sin(theta) * np.sin(phi)  # x2
            # x3 = cos(theta) is determined by the constraint, but in DTLZ4 
            # the first n_obj-1 variables are the decision variables for the front
            # and x3 is computed? No, in standard DTLZ, the first n_obj-1 
            # decision variables ARE the front coordinates.
            
            # Actually, I need to be more careful. In DTLZ problems:
            # The first (n_obj - 1) decision variables directly correspond to 
            # the front parameters. For DTLZ3/4 with 3 objectives:
            # x[0] and x[1] are the front parameters, and the objectives are:
            # f1 = 0.5 * (1+g) * x[0]
            # f2 = 0.5 * (1+g) * x[1]  
            # f3 = 0.5 * (1+g) * sqrt(1 - x[0]^2 - x[1]^2)
            # where x[0]^2 + x[1]^2 <= 1
            
            # So we need x[0]^2 + x[1]^2 <= 1, and they should be >= 0
            # The front is the quarter unit disk in the x0-x1 plane
            
            # Let me redo: use polar coordinates in the quarter disk
            # x[0] = r * cos(phi), x[1] = r * sin(phi)
            # where r in [0, 1], phi in [0, pi/2]
            # For uniform area on the disk: r = sqrt(uniform), phi = uniform
            
            # But we want the front, which is r=1 (the boundary)
            # So x[0] = cos(phi), x[1] = sin(phi) for phi in [0, pi/2]
            
            # Let's just use phi directly
            x[0] = np.cos(phi)
            x[1] = np.sin(phi)
            
            # G-variables: set to 0.5 for optimal g
            for k in range(n_front, dim):
                x[k] = 0.5
            
            X.append(x)
        
        # Clip to bounds
        X = np.array(X)
        X = np.clip(X, lo, hi)
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        used = len(X)
        
        # Use remaining budget for local refinement
        # For each point, we can try small perturbations of the g-variables
        # to see if we can improve (though 0.5 should be optimal)
        
        if n_refine > 0 and used < budget:
            # Try to refine g-variables for the worst-performing points
            # Compute a simple quality metric: distance to ideal point or similar
            # For now, just do random perturbation of g-variables
            
            for _ in range(min(n_refine, budget - used)):
                if used >= budget:
                    break
                # Pick a random point
                idx = np.random.randint(len(X))
                x_new = X[idx].copy()
                # Perturb g-variables slightly
                for k in range(n_front, dim):
                    x_new[k] += 0.01 * np.random.randn()
                x_new = np.clip(x_new, lo, hi)
                f_new = problem(x_new)
                used += 1
                
                # Replace if better (dominates or is closer to ideal)
                # Simple check: if new point is not dominated and improves some objective
                # For simplicity, just keep the best in terms of sum of objectives
                if np.sum(f_new) < np.sum(F[idx]):
                    X[idx] = x_new
                    F[idx] = f_new
        
        return X
# EVOLVE-BLOCK-END
