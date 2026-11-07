import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = self.start_time + 290  # Safety margin

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # For DTLZ2 with M=3 and dim=12:
        # x[0:2] are shape variables (angles)
        # x[2:12] are distance variables
        # Optimal shape: x[0]=pi/6, x[1]=pi/6
        # Optimal distance: all distance vars = 0.5
        
        # We will generate a grid over the shape variables (x[0], x[1])
        # and set distance variables to 0.5 (or mid-point)
        
        # Determine shape variables: first M-1 variables
        # For DTLZ2, shape vars are the first M-1 variables
        # Let's assume the first n_obj-1 variables are shape variables.
        # To be robust, we could try to identify them, but for DTLZ2 this is standard.
        # Let's stick to the standard DTLZ2 structure: first M-1 are shape.
        
        k_shape = self.n_obj - 1
        shape_vars = list(range(k_shape))
        
        # The remaining variables are distance variables
        # For DTLZ2, the optimal distance is achieved when all distance vars are 0.5
        # (since g(x) = (dim-M+1) * sum(x_i^2) for i in M..dim, min is 0 at x_i=0, but wait)
        # Actually for DTLZ2:
        # g(x) = (dim - M + 1) * sum_{i=M}^{dim} x_i^2
        # f_1 = (1+g) * prod(cos(x_i))
        # f_2 = (1+g) * sin(x_{M-1}) * prod_{i=1}^{M-2}(cos(x_i))
        # ...
        # To minimize all f_i, we need g=0 (so x_i=0 for i>=M) and shape vars on the Pareto front.
        # Wait, the standard DTLZ2 Pareto front is where g=0.
        # So distance vars should be 0.
        
        # Let's verify: if x_i=0 for i>=M, then g=0.
        # Then f_1 = prod(cos(x_i)), etc.
        # The Pareto front is the quarter unit sphere (or hyper-surface) defined by the shape vars.
        
        # So we set distance vars to 0.
        
        # Grid search on shape vars
        # We have budget=200.
        # Used so far: 0 (we don't need to identify vars if we assume standard structure)
        # But to be safe, let's just use the budget for the grid.
        
        # How many points per dimension?
        # g^2 <= 200 => g <= 14
        # Let's use g=14 => 196 points.
        # Or g=15 => 225 points (too many).
        
        # Let's calculate g based on budget
        g = int(np.sqrt(self.budget))
        if g < 2:
            g = 2
        
        # Generate grid
        # Shape vars are angles in [0, pi/2] for DTLZ2
        # But bounds are [0, 1].
        # The problem clips to bounds.
        # If we assume the problem is standard DTLZ2 scaled to [0,1],
        # then x[0], x[1] in [0, pi/2] maps to [0,1] via x/pi/2?
        # Or is the problem defined such that x in [0,1] corresponds to angles in [0, pi/2]?
        # The task says: "problem(x) clips a finite length-12 vector to the bounds and returns 3 objective values."
        # And "bounds [0, 1] for every variable".
        # Standard DTLZ2 usually has x in [0,1] for distance vars and [0, pi/2] for shape vars?
        # Or is it all [0,1]?
        # If all [0,1], then the angles are effectively in [0, 1] radians?
        # Or does the problem implementation map [0,1] to [0, pi/2]?
        
        # Let's assume the problem implementation handles the mapping internally.
        # We just need to sample the shape variables uniformly in [0, 1] to cover the Pareto front.
        # If the Pareto front corresponds to a specific range of angles, uniform sampling in [0,1] might miss parts if the mapping is non-linear.
        # However, for DTLZ2, the Pareto front is a quarter sphere.
        # If x[0], x[1] are angles, and they are in [0, pi/2], and the bounds are [0,1],
        # it's likely that the problem expects x[0], x[1] in [0,1] to map to [0, pi/2].
        # Or maybe the problem just uses x[0], x[1] as is?
        
        # Let's try to be smart.
        # If we sample x[0], x[1] uniformly in [0,1], we might not cover the whole front if the effective range is [0, pi/2].
        # But if the problem scales it, then [0,1] is correct.
        
        # Let's assume the problem is standard DTLZ2 where shape vars are in [0, pi/2] but bounds are [0,1].
        # This implies the problem might multiply by pi/2 internally?
        # Or maybe the bounds are actually [0, pi/2] for shape vars?
        # The task says "bounds [0, 1] for every variable".
        
        # Let's just sample uniformly in [0,1] for shape vars and set distance vars to 0.
        # This should cover the Pareto front if the problem maps [0,1] to the appropriate angle range.
        
        # Generate grid points
        grid_vals = np.linspace(0, 1, g)
        
        solutions = []
        objective_values = []
        
        # Iterate over grid
        for v1 in grid_vals:
            for v2 in grid_vals:
                if time.time() > self.deadline:
                    break
                if len(objective_values) >= self.budget:
                    break
                
                x = np.zeros(self.dim)
                x[shape_vars[0]] = v1
                x[shape_vars[1]] = v2
                # Distance vars (index 2 to 11) are already 0
                
                f = np.array(problem(x))
                solutions.append(x)
                objective_values.append(f)
                
            if len(objective_values) >= self.budget:
                break
        
        return solutions
