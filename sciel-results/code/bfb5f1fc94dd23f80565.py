"""Multi-objective optimizer on DTLZ7 with structural exploitation and budget tracking."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for DTLZ7.
    
    Key insights:
    1. DTLZ7 has m objectives, n variables.
       Position variables: x_0, ..., x_{m-2} (m-1 variables)
       G variables: x_{m-1}, ..., x_{n-1} (k = n - m + 1 variables)
       
       f_1 = x_0
       f_2 = x_1
       ...
       f_{m-1} = x_{m-2}
       f_m = g(x_M) * (1 + h(f_1, ..., f_{m-1}, g))
       
       where g(x_M) = sum_{i=1}^{k} [(x_i - 0.5)^2 - cos(20*pi*(x_i + 0.5))]
       
       The minimum of each term is -1 at x_i = 0.5, so g_min = -k.
       
       Wait, let me reconsider. The standard DTLZ7 definition:
       g(x) = sum_{i=n-k+1}^{n} [(x_i - 0.5)^2 - cos(20*pi*(x_i + 0.5))]
       
       At x_i = 0.5: (0)^2 - cos(20*pi*1.0) = -cos(20*pi) = -1
       So g_min = -k where k = n - m + 1.
       
       For n=22, m=3: k = 20, so g_min = -20.
       
       The Pareto front is characterized by g = g_min and the trade-off among f_1, ..., f_{m-1}.
       
    2. Strategy:
       - Set all g-variables to 0.5 to achieve g_min.
       - Sweep position variables (f_1, f_2 for m=3) over a grid in [0,1]^{m-1}.
       - Use remaining budget for local refinement.
    """
    
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.n_g_vars = dim - n_obj + 1  # k = n - m + 1
        self.n_pos_vars = n_obj - 1      # number of position variables
    
    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310  # safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n = self.dim
        m = self.n_obj
        k = self.n_g_vars  # number of g-variables
        n_pos = self.n_pos_vars  # number of position variables
        
        evals_used = 0
        max_evals = self.budget
        all_f = []  # list of objective vectors
        
        # Helper: create a point with given position values and g-variables at 0.5
        def make_point(pos_vals):
            """Create a point with given position variable values, g-variables at 0.5."""
            x = np.zeros(n)
            for i in range(n_pos):
                x[i] = pos_vals[i]
            for i in range(n_pos, n):
                x[i] = 0.5  # g-variables at minimum
            return np.clip(x, lo, hi)
        
        # Phase 1: Grid on the Pareto front
        # For m=3, we have 2 position variables. Use a 2D grid.
        # For general m, use a grid in (m-1) dimensions.
        
        # Determine grid size based on budget
        # Use about 80% of budget for grid, 20% for refinement
        grid_fraction = 0.8
        grid_budget = int(self.budget * grid_fraction)
        
        if n_pos == 2:
            # 2D grid
            n_grid = int(np.sqrt(grid_budget))
            n_grid = max(2, min(n_grid, 50))
            
            grid_points = []
            for i in range(n_grid):
                for j in range(n_grid):
                    f1 = (i + 0.5) / n_grid
                    f2 = (j + 0.5) / n_grid
                    grid_points.append((f1, f2))
        elif n_pos == 1:
            n_grid = min(grid_budget, 100)
            grid_points = [((i + 0.5) / n_grid,) for i in range(n_grid)]
        else:
            # For higher dimensions, use a stratified sample
            n_grid = min(grid_budget, 100)
            rng_init = np.random.default_rng(42)
            grid_points = [tuple(rng_init.uniform(0, 1, n_pos)) for _ in range(n_grid)]
        
        # Evaluate grid points
        for pos_vals in grid_points:
            if evals_used >= max_evals:
                break
            if time.time() > deadline - 30:
                break
            
            x = make_point(np.array(pos_vals))
            f = problem(x)
            evals_used += 1
            all_f.append(np.array(f))
        
        # Phase 2: Local refinement
        remaining_budget = max_evals - evals_used
        
        if remaining_budget > 0 and len(all_f) > 0:
            F = np.array(all_f)
            n_pts = len(F)
            
            # Find non-dominated points
            dominated = np.zeros(n_pts, dtype=bool)
            for i in range(n_pts):
                if dominated[i]:
                    continue
                for j in range(n_pts):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            
            non_dom = [i for i in range(n_pts) if not dominated[i]]
            
            if non_dom:
                rng = np.random.default_rng(42)
                step_sizes = [0.1, 0.05, 0.02, 0.01]
                
                for idx in non_dom:
                    if evals_used >= max_evals:
                        break
                    if time.time() > deadline - 20:
                        break
                    
                    # Reconstruct the point
                    if n_pos == 2:
                        # Find which grid point this was
                        # Just create a new point from scratch
                        base_x = make_point(np.array([
                            (idx % n_grid + 0.5) / n_grid,
                            (idx // n_grid + 0.5) / n_grid
                        ]))
                    else:
                        base_x = make_point(np.array([
                            (idx // (n_grid ** (n_pos - 1 - i)) % n_grid + 0.5) / n_grid
                            for i in range(n_pos)
                        ]))
                    
                    base_f = problem(base_x)
                    evals_used += 1
                    if evals_used >= max_evals:
                        break
                    
                    # Try perturbations
                    for step in step_sizes:
                        if evals_used >= max_evals:
                            break
                        if time.time() > deadline - 10:
                            break
                        
                        for trial in range(2):
                            if evals_used >= max_evals:
                                break
                            if time.time() > deadline - 10:
                                break
                            
                            new_x = base_x + rng.standard_normal(n) * step
                            new_x = np.clip(new_x, lo, hi)
                            
                            f = problem(new_x)
                            evals_used += 1
                            
                            # Accept if not dominated by base
                            if np.any(f < base_f) and np.all(f <= base_f + 1e-8):
                                base_x = new_x.copy()
                                base_f = f
        
        return None
# EVOLVE-BLOCK-END
