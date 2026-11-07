"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic multi-scale grid search with local hill climb for f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill

        global_best_x = None
        global_best_f = np.inf
        calls_made = 0
        total_budget = self.budget
        dim = self.dim

        def try_eval(x):
            nonlocal global_best_x, global_best_f, calls_made
            if calls_made >= total_budget:
                return
            if time.time() > deadline - 1.0:
                return
            
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            
            if fx < global_best_f:
                global_best_f = fx
                global_best_x = x.copy()

        # Phase 1: Deterministic Grid Search
        # We use a hierarchical approach:
        # Level 0: Vertices of the hypercube (2^dim points)
        # Level 1: Midpoints of edges, faces, etc. (or a coarser grid)
        
        # For dim=5, 2^5 = 32 points. This is very cheap.
        # Let's define a few levels of granularity.
        
        # Level 0: The 32 corners of [-5, 5]^5
        corners = np.array([[ -5 if i else 5 for i in range(dim)] for i in range(1 << dim)])
        # Wait, the list comprehension above is slightly wrong for generating all combinations.
        # Let's use itertools.product or a proper binary generation.
        
        # Generate all 2^dim vertices
        vertices = []
        for i in range(1 << dim):
            v = np.array([ -5.0 if (i >> j) & 1 else 5.0 for j in range(dim) ])
            vertices.append(v)
        
        for v in vertices:
            try_eval(v)
            if calls_made >= total_budget or time.time() > deadline - 1.0:
                break

        # Level 1: Midpoints of the box (0,0,0,0,0) and midpoints of axes?
        # Let's add the origin.
        try_eval(np.zeros(dim))
        
        # Let's add points that are 0 in all but one dimension, and +/- 2.5 in that dimension?
        # Or simply a grid of {-5, -2.5, 0, 2.5, 5}^dim? That is 5^5 = 3125 points. Too many.
        
        # Let's stick to a smaller set of structured points.
        # How about midpoints of the edges?
        # An edge is defined by fixing dim-1 coordinates to +/-5 and varying the last.
        # Midpoint of such an edge has dim-1 coords at +/-5 and 1 coord at 0.
        
        # Let's generate all points with coordinates in {-5, 0, 5} that have at least one 0?
        # No, let's just do a few specific structured sets.
        
        # Set 2: Points with coordinates in {-2.5, 0, 2.5}
        # 3^5 = 243 points. Total so far 32 + 1 + 243 = 276.
        # This is a good chunk of the budget (500).
        
        coords = [-2.5, 0.0, 2.5]
        # Generate all combinations
        import itertools
        for combo in itertools.product(coords, repeat=dim):
            if calls_made >= total_budget or time.time() > deadline - 1.0:
                break
            try_eval(np.array(combo))
            
        # Set 3: If we have budget left, add points with coords in {-3.75, -1.25, 1.25, 3.75}?
        # Or refine the best region.
        
        # If global_best_x is not set (shouldn't happen), fallback
        if global_best_x is None:
            global_best_x = np.zeros(dim)
            global_best_f = func(global_best_x)
            calls_made += 1

        # Phase 2: Local Hill Climb from Global Best
        # Use remaining budget for a simple (1+1)-ES or fixed step descent.
        # Since the grid search is deterministic and covers the space reasonably,
        # the global best should be in a good basin.
        
        x = global_best_x.copy()
        fx = global_best_f
        
        # Adaptive step size local search
        sigma = 1.0
        no_improve = 0
        patience = 20
        
        while calls_made < total_budget and time.time() < deadline - 0.5:
            # Generate candidate
            # Use a deterministic sequence of directions if possible? 
            # Or just use a fixed seed RNG for reproducibility within the run?
            # The task allows stochasticity, but the prompt asked for "avoiding stochastic initialization".
            # Local search can be stochastic or deterministic. Let's use a simple deterministic 
            # coordinate-wise search or a small random step.
            # To be safe and fast, let's do a coordinate-wise hill climb with adaptive step.
            
            # Try moving along each axis
            improved = False
            for d in range(dim):
                if calls_made >= total_budget or time.time() > deadline - 0.5:
                    break
                    
                # Try positive step
                y = x.copy()
                y[d] += sigma
                try_eval(y)
                if global_best_f < fx:
                    x = global_best_x.copy()
                    fx = global_best_f
                    improved = True
                    no_improve = 0
                else:
                    no_improve += 1
                
                # Try negative step
                y = x.copy()
                y[d] -= sigma
                try_eval(y)
                if global_best_f < fx:
                    x = global_best_x.copy()
                    fx = global_best_f
                    improved = True
                    no_improve = 0
                else:
                    no_improve += 1
            
            if improved:
                sigma *= 1.2 # Expand step if we made progress? Or keep it?
                # Usually in hill climbing, if you make progress, you might keep the step or increase slightly.
                # If you don't, you decrease.
            else:
                sigma *= 0.5
                if sigma < 1e-4:
                    break
                if no_improve > patience * dim:
                    break

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
