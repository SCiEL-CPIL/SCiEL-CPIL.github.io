import numpy as np
import time

class Optimizer:
    """
    Lattice-based deterministic grid search exploiting Rastrigin's periodic structure.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(7)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def try_point(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= budget:
                return
            if time.time() > deadline:
                return
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 1: Lattice search exploiting Rastrigin's periodicity
        # Rastrigin has minima at integer coordinates (shifted by hidden offset)
        # Try integer lattice points first, then half-integer offsets
        
        # Generate candidate points: integers and half-integers in [-5, 5]
        # But we don't know the shift, so try a structured grid
        
        # Integer lattice points
        int_coords = np.arange(-5, 6)  # -5 to 5
        half_coords = np.arange(-5, 5.5, 0.5)  # -5 to 5 in steps of 0.5
        
        # Phase 1a: Try integer lattice points (21^5 = 4084111, too many)
        # Instead, use a smarter approach: try combinations of a subset
        
        # Since dim=5 and budget=500, we can only evaluate 500 points
        # Strategy: Use a space-filling or structured approach
        
        # Phase 1: Coarse grid search
        # Divide [-5, 5] into segments and try representative points
        # For 500 calls in 5D, we can do a structured exploration
        
        # Approach: Try points on a lattice with spacing 2 (covering [-4, 4] roughly)
        # Then refine around the best found
        
        # Lattice points with spacing 2 in [-4, 4]
        lattice_spacing = 2.0
        lattice_min = -4.0
        lattice_max = 4.0
        n_per_dim = int((lattice_max - lattice_min) / lattice_spacing) + 1
        # n_per_dim = 5, so 5^5 = 3125 points, still too many
        
        # Use a smaller lattice: spacing 4, so [-4, 0, 4] -> 3 points per dim
        # 3^5 = 243 points, fits in budget
        lattice_vals = np.array([-4.0, 0.0, 4.0])
        
        # Generate all combinations
        from itertools import product
        lattice_points = list(product(lattice_vals, repeat=dim))
        
        # Shuffle to get diverse early progress
        rng.shuffle(lattice_points)
        
        for pt in lattice_points:
            x = np.array(pt, dtype=float)
            try_point(x)
            if calls_made >= budget or time.time() > deadline:
                break
        
        # Phase 2: Local refinement around the best point found
        if best_x is not None and calls_made < budget:
            # Nelder-Mead style local search
            # Create simplex around best_x
            eps = 0.5
            simplex = [best_x.copy()]
            for i in range(dim):
                x_plus = best_x.copy()
                x_plus[i] += eps
                simplex.append(np.clip(x_plus, -5, 5))
            
            # Evaluate simplex
            simplex_fits = []
            for x in simplex:
                if calls_made >= budget or time.time() > deadline:
                    break
                fx = func(x)
                calls_made += 1
                simplex_fits.append(fx)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            
            # Simple local refinement: try moving in coordinate directions
            step_sizes = [0.5, 0.25, 0.125, 0.0625, 0.03125]
            for step in step_sizes:
                if calls_made >= budget or time.time() > deadline:
                    break
                for i in range(dim):
                    for sign in [1, -1]:
                        if calls_made >= budget or time.time() > deadline:
                            break
                        x_new = best_x.copy()
                        x_new[i] += sign * step
                        x_new = np.clip(x_new, -5, 5)
                        try_point(x_new)
        
        # Phase 3: Random sampling if budget remains
        while calls_made < budget and time.time() <= deadline:
            x = rng.uniform(-5, 5, size=dim)
            try_point(x)
        
        return best_x, best_f
