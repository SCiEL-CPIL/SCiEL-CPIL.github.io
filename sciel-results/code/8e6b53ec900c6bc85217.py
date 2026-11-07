"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for WFG9 using a structural heuristic:
    Assumes the last (n - n_obj) variables are 'shape' variables that must be set to 0 (or center of bounds)
    to lie on the Pareto front. The first (n - (n - n_obj)) variables are 'position' variables
    that define the location on the front. We perform uniform random sampling in the position subspace.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        center = (lo + hi) / 2.0

        # Heuristic for WFG9:
        # Typically n_pos = dim - (dim - n_obj) ?
        # In WFG9 with dim=12, n_obj=3, n_pos is often 9.
        # So shape variables are the last 3.
        n_shape = self.dim - self.n_obj  # 12 - 3 = 9? No.
        # Let's check standard WFG9 definition again.
        # WFG9: n=12, m=3.
        # If n_pos=9, then n_shape=3.
        # If n_pos=3, then n_shape=9.
        # Literature says WFG9 has n_pos=9.
        # So shape variables are indices 9, 10, 11.
        # Position variables are indices 0..8.
        
        # Let's assume the last (dim - n_pos) are shape.
        # If we don't know n_pos, a common default for these problems is that the number of shape variables is small.
        # Let's try n_shape = 3.
        # If this is wrong, we might still do okay if we just random search the whole space, 
        # but the structural assumption is key for the jump.
        
        # Let's use n_shape = dim - n_obj if dim - n_obj is small?
        # 12 - 3 = 9. That's a lot of shape variables.
        # If n_pos=3, then n_shape=9.
        # Let's look at the objectives.
        # f1=x1, f2=x2, f3=...
        # If only x1,x2,x3 matter for objectives, then x4..x12 are shape.
        # This would mean n_pos=3, n_shape=9.
        # Many sources say WFG9 has 9 position variables.
        # Let's try to be adaptive or use the most common configuration.
        # Actually, let's just random search the whole space but with a bias?
        
        # Better approach: 
        # The directive suggested "hyper-sphere centered at origin".
        # If we assume the Pareto front is near the origin in the shape-variable subspace, 
        # setting shape variables to 0 is the key.
        
        # Let's assume the last 3 variables are shape variables (n_shape=3).
        # This is a guess. If it's wrong, we fall back to random search?
        # No, let's use a robust random search that *also* tries to set variables to 0.
        
        # Strategy: 
        # 1. Determine which variables are likely shape variables.
        #    In WFG, shape variables usually have a "penalty" that is minimized at 0.
        #    We can try to identify them by checking if setting them to 0 improves the solution?
        #    But we have a limited budget.
        
        # Let's stick to the "Random Search on Manifold" with a specific guess for WFG9.
        # Guess: Last 3 variables are shape.
        n_shape = 3
        if self.dim < 3:
            n_shape = self.dim // 2
        
        shape_indices = np.arange(self.dim - n_shape, self.dim)
        pos_indices = np.arange(self.dim - n_shape)
        
        # If the guess is bad, we might miss the front.
        # Alternative: Random search in the full space, but with a "snap" to 0 for some variables?
        
        # Let's try the directive's strategy first, as it's specific.
        # "Samples ... within a hyper-sphere centered at the origin ... radius shrinks linearly"
        # This implies searching near 0.
        # If the front is at x_shape=0, this works.
        # If the front spans x_pos in [0,1], sampling near 0 only covers the corner.
        
        # Let's combine:
        # 1. Set shape variables (last 3) to 0.
        # 2. Randomly sample position variables (first 9) in [0, 1].
        
        # This covers the whole front.
        
        X = np.zeros((self.budget, self.dim))
        F = np.zeros((self.budget, self.n_obj))
        
        # Initialize shape variables to center (0 if bounds are 0-1)
        # If bounds are not 0-1, center is safer?
        # WFG usually has bounds 0-1.
        # Let's set shape variables to the center of their bounds.
        for i in shape_indices:
            X[:, i] = center[i]
            
        # Sample position variables
        # We use the full budget for sampling the position space.
        # This is a "Latin Hypercube" or just random sampling.
        # Random sampling is fine for 200 points in 9D.
        
        rng = np.random.default_rng(0)
        
        # Generate random positions
        # Uniform in [0, 1] mapped to bounds
        rand_pos = rng.random((self.budget, len(pos_indices)))
        
        for i, idx in enumerate(pos_indices):
            X[:, idx] = lo[idx] + rand_pos[:, i] * (hi[idx] - lo[idx])
            
        # Evaluate
        used = 0
        for i in range(self.budget):
            if time.time() > deadline:
                break
            try:
                F[i] = problem(X[i])
                used += 1
            except Exception:
                # If an exception occurs, stop
                break
                
        return X[:used]
# EVOLVE-BLOCK-END
