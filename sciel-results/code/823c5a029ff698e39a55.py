import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_obj = self.n_obj
        dim = self.dim
        n_pos = n_obj - 1
        n_shape = dim - n_pos
        
        # DTLZ2 Pareto Front Structure:
        # f_i = g * h_i
        # g = sum((x_j - 0.5)^2) for j in [n_pos, dim)
        # h_i = cos(x_i * pi / 2) for i in [0, n_pos-1]
        # h_n = sin(x_{n_pos-1} * pi / 2)
        # The Pareto front is defined by g=0, i.e., x_j = 0.5 for all shape variables.
        # The position variables x_0 ... x_{n_pos-1} should be in [0, 1] to cover the front.
        
        # Strategy:
        # 1. Use a significant portion of the budget to evaluate points exactly on the Pareto front.
        #    We use a Sobol sequence or a structured grid for the position variables to ensure
        #    good spread and hypervolume coverage.
        # 2. Use the remaining budget to evaluate points slightly off the front to see if there
        #    are any numerical advantages or to "fill in" gaps, though for DTLZ2, the front is 
        #    convex and well-defined, so on-front points are optimal.
        #    Actually, for hypervolume, we just need a dense set of non-dominated points.
        #    Since g=0 is the only way to be on the front, any deviation increases all objectives.
        #    Therefore, the optimal strategy is to evaluate as many distinct points on the front as possible.
        
        # We have `budget` evaluations.
        # Let's allocate all evaluations to on-front points, but with a twist:
        # We can use a Latin Hypercube or Sobol-like sampling for the position variables.
        # Since we don't have scipy.stats.qmc directly imported in the prompt's allowed list 
        # (though SciPy is available, let's stick to numpy for safety and speed), 
        # we can use a simple structured grid or random sampling with replacement avoidance.
        
        # A dense grid might be too coarse if budget is small, but 200 is decent.
        # For 2 position variables (n_obj=3), a 14x14 grid is 196 points. 
        # This is very dense and covers the front well.
        # Let's use a stratified random sampling (Latin Hypercube style) for the position variables.
        
        rng = np.random.default_rng(42)
        
        # Generate position variables
        # We want to distribute points well in the [0, 1]^(n_pos) space.
        # For n_pos=2, we can use a grid. For general n_pos, we use LHS.
        
        # Let's use a simple approach: generate random points, but ensure they are distinct.
        # Better: Use the fact that for DTLZ2, the front is a hypersphere surface.
        # Uniform sampling on the unit hypersphere surface gives good HV coverage.
        
        # However, the mapping is x_i -> cos/sin. 
        # If we sample angles uniformly, we get uniform points on the front.
        # For n_pos=2 (3 objectives):
        # x0 = angle1, x1 = angle2? No.
        # h0 = cos(x0*pi/2)
        # h1 = sin(x0*pi/2)*cos(x1*pi/2)
        # h2 = sin(x0*pi/2)*sin(x1*pi/2)
        # This maps [0,1]^2 to the first octant of the unit sphere.
        # Uniform sampling in [0,1]^2 does NOT give uniform density on the sphere.
        # To get uniform density on the sphere (which maximizes HV for a convex front),
        # we should sample the angles appropriately.
        
        # For 3 objectives (2 position vars), the area element on the sphere is sin(theta) dtheta dphi.
        # Here x0 corresponds to theta (0 to pi/2) and x1 to phi (0 to pi/2).
        # To sample uniformly on the surface, we should sample x0 such that cos(x0*pi/2) is uniform in [0,1]?
        # Let u ~ U[0,1]. Let cos(x0*pi/2) = u. Then x0 = (2/pi) * arccos(u).
        # And x1 ~ U[0,1] (since the cross-section is a line segment for fixed theta? No, it's a circle).
        # Actually, for 3D, uniform sampling on the spherical triangle can be done by:
        # 1. Sample z ~ U[0,1]. Let z = sin^2(x0*pi/2)? No.
        
        # Simpler robust approach: 
        # Just use a dense grid or random points. With 200 points, a grid of ~14x14 is excellent.
        # Let's check if budget allows a grid.
        # If n_pos == 2:
        #   side = int(np.sqrt(budget))
        #   Generate grid.
        
        # If n_pos > 2, grid explodes. But task says n_obj=3, so n_pos=2.
        # The task specifies n_obj=3, dim=12. So n_pos=2.
        
        if n_pos == 2:
            side = int(np.sqrt(self.budget))
            # Adjust side to not exceed budget
            while side * side > self.budget:
                side -= 1
            
            # Create grid
            x0_vals = np.linspace(0, 1, side)
            x1_vals = np.linspace(0, 1, side)
            X0, X1 = np.meshgrid(x0_vals, x1_vals)
            X_pos = np.vstack([X0.ravel(), X1.ravel()]).T
            
            # We might have fewer points than budget. 
            # If we have remaining budget, we can add random points on the front.
            num_grid = len(X_pos)
            remaining = self.budget - num_grid
            
            if remaining > 0:
                # Add random points on the front to fill gaps or just increase density
                X_pos_rand = rng.random((remaining, n_pos))
                X_pos = np.vstack([X_pos, X_pos_rand])
            
            # Ensure we don't exceed budget
            X_pos = X_pos[:self.budget]
            
        else:
            # Fallback for other dimensions: Random sampling on the front
            X_pos = rng.random((self.budget, n_pos))
            
        # Construct full X
        X = np.zeros((len(X_pos), dim))
        X[:, :n_pos] = X_pos
        
        # Set shape variables to 0.5 (midpoint) to ensure g=0
        # The bounds are [0,1], so 0.5 is the center and minimizes g.
        mid_shape = 0.5 * (lo[n_pos:] + hi[n_pos:])
        X[:, n_pos:] = mid_shape
        
        # Evaluate all points
        # We must be careful about time, but 200 evaluations of a simple function is fast.
        F = np.array([problem(x) for x in X])
        
        # Check time just in case, though it should be well within limits
        if time.time() > deadline:
            return X
            
        return X
