"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Deterministic Nelder-Mead Simplex with coarse grid seeding for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # State
        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0
        rng = np.random.default_rng(42)

        def do_call(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget or time.time() > deadline:
                return None
            x = np.clip(x, -5, 5)
            try:
                f = func(x)
            except Exception:
                f = np.inf
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Coarse Exploration with Grid Seeding
        # For BBOB f22 (Gallagher 21 peaks), the landscape is multimodal.
        # We use a grid of starting points to identify promising basins.
        
        # Generate a coarse grid of initial points
        # 3 points per dimension -> 3^5 = 243 points
        # This covers the space [-5, 5]^5 reasonably well.
        grid_values = np.array([-4.0, 0.0, 4.0])
        grid_points = []
        for v0 in grid_values:
            for v1 in grid_values:
                for v2 in grid_values:
                    for v3 in grid_values:
                        for v4 in grid_values:
                            grid_points.append(np.array([v0, v1, v2, v3, v4]))
        
        # Evaluate grid points
        # We will keep the best 10 points as seeds for Nelder-Mead
        # To save budget, we don't evaluate all 243 if we find good ones early,
        # but for f22, a full coarse pass is often worth it.
        # However, 243 calls is half the budget. Let's be smarter.
        
        # Strategy: Evaluate a subset first, then refine.
        # Let's use a Halton sequence for better space-filling than a regular grid
        # or just stick to the grid but limit the number of seeds.
        
        # Let's use the grid but only keep the top 5 seeds.
        # To avoid wasting calls, we'll evaluate them and track the best.
        
        # Actually, let's use a smaller set of strategic points first.
        # Center + Axis points + Random
        initial_points = []
        initial_points.append(np.zeros(dim))
        
        # Axis points
        for i in range(dim):
            for val in [-3.5, 3.5]:
                p = np.zeros(dim)
                p[i] = val
                initial_points.append(p)
        
        # Random points
        for _ in range(10):
            initial_points.append(rng.uniform(-4, 4, size=dim))
            
        # Evaluate initial points
        for p in initial_points:
            f = do_call(p)
            if f is None:
                break
        
        if call_count >= budget or time.time() > deadline:
            return best_x, best_f
        
        # Phase 2: Multi-Start Nelder-Mead
        # We will run Nelder-Mead from the best point found so far.
        # If it stagnates, we might try another restart, but NM is usually fast.
        
        def nelder_mead(start_x, max_iter=100):
            nonlocal call_count, best_f, best_x, deadline
            
            # Initialize simplex
            # Vertices: start_x + perturbations
            alpha = 0.1
            vertices = [start_x.copy()]
            for i in range(dim):
                v = start_x.copy()
                v[i] += alpha
                vertices.append(v)
            
            # Evaluate vertices
            f_values = []
            for v in vertices:
                f = do_call(v)
                if f is None:
                    return
                f_values.append(f)
            
            # Sort vertices by function value
            indices = np.argsort(f_values)
            vertices = [vertices[i] for i in indices]
            f_values = [f_values[i] for i in indices]
            
            # Parameters
            alpha_nm = 1.0  # Reflection
            gamma = 2.0     # Expansion
            rho = 0.5       # Contraction
            sigma = 0.5     # Shrink
            
            for _ in range(max_iter):
                if call_count >= budget or time.time() > deadline:
                    return
                
                # Check convergence
                if max(f_values) - min(f_values) < 1e-8:
                    break
                
                # Centroid of all vertices except the worst
                centroid = np.mean(vertices[:-1], axis=0)
                
                # Worst vertex
                worst_v = vertices[-1]
                worst_f = f_values[-1]
                
                # Reflection
                reflected_v = centroid + alpha_nm * (centroid - worst_v)
                f_reflected = do_call(reflected_v)
                if f_reflected is None:
                    return
                
                if f_reflected < f_values[0]:
                    # Expansion
                    expanded_v = centroid + gamma * (reflected_v - centroid)
                    f_expanded = do_call(expanded_v)
                    if f_expanded is None:
                        return
                    
                    if f_expanded < f_reflected:
                        new_v = expanded_v
                        new_f = f_expanded
                    else:
                        new_v = reflected_v
                        new_f = f_reflected
                elif f_reflected < f_values[-2]:
                    new_v = reflected_v
                    new_f = f_reflected
                else:
                    # Contraction
                    contracted_v = centroid + rho * (worst_v - centroid)
                    f_contracted = do_call(contracted_v)
                    if f_contracted is None:
                        return
                    
                    if f_contracted < worst_f:
                        new_v = contracted_v
                        new_f = f_contracted
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            vertices[i] = vertices[0] + sigma * (vertices[i] - vertices[0])
                            f_shrink = do_call(vertices[i])
                            if f_shrink is None:
                                return
                            f_values[i] = f_shrink
                        
                        # Re-sort
                        indices = np.argsort(f_values)
                        vertices = [vertices[i] for i in indices]
                        f_values = [f_values[i] for i in indices]
                        continue
                
                # Replace worst vertex
                vertices[-1] = new_v
                f_values[-1] = new_f
                
                # Re-sort
                indices = np.argsort(f_values)
                vertices = [vertices[i] for i in indices]
                f_values = [f_values[i] for i in indices]
            
            # After NM, update global best
            # The best vertex is vertices[0]
            # do_call already updated best_x/best_f if it was better

        # Run NM from the current best
        nelder_mead(best_x.copy(), max_iter=100)
        
        # Phase 3: If budget remains, try a few more random restarts with NM
        # This helps if the first NM got stuck in a local minimum
        remaining_budget = budget - call_count
        if remaining_budget > 50 and time.time() < deadline:
            # Try 2 more restarts
            for _ in range(2):
                if call_count >= budget or time.time() > deadline:
                    break
                
                # Random restart point
                start = rng.uniform(-4, 4, size=dim)
                f_start = do_call(start)
                if f_start is None:
                    break
                
                # Run a shorter NM
                nelder_mead(start, max_iter=50)
        
        return best_x, best_f

# EVOLVE-BLOCK-END
