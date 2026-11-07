import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        global_best_x = None
        global_best_f = float('inf')
        calls = 0

        def check_deadline():
            return time.time() > deadline

        def record(y, fy):
            nonlocal global_best_x, global_best_f
            if fy < global_best_f:
                global_best_f = fy
                global_best_x = y.copy()

        def run_es(x_start, fx_start, max_calls, sigma_init=2.0, window=15):
            """Run (1+1)-ES with windowed 1/5 success rule for up to max_calls evaluations."""
            nonlocal calls, global_best_x, global_best_f

            x = x_start.copy()
            fx = fx_start
            sigma = sigma_init
            min_sigma = 1e-8
            max_sigma = 10.0
            successes = 0
            local_best_x = x.copy()
            local_best_f = fx

            for _ in range(max_calls):
                if calls >= budget:
                    break
                if check_deadline():
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1

                if fy < local_best_f:
                    local_best_f = fy
                    local_best_x = y.copy()

                record(y, fy)

                if fy < fx:
                    x = y
                    fx = fy
                    successes += 1

                # Windowed 1/5 success rule
                if calls % window == 0:
                    success_rate = successes / window
                    if success_rate > 0.2:
                        sigma *= 1.3
                    elif success_rate < 0.2:
                        sigma *= 0.7
                    sigma = np.clip(sigma, min_sigma, max_sigma)
                    successes = 0

            # Update global best with local best
            if local_best_f < global_best_f:
                global_best_f = local_best_f
                global_best_x = local_best_x.copy()

            return local_best_x, local_best_f

        def run_grid_refinement(center, half_width, max_calls):
            """
            Grid-based adaptive search:
            1. Evaluate a coarse grid (3^dim points) within the current cell.
            2. Find the best point.
            3. If budget remains, subdivide the cell around the best point by half.
            4. Repeat until budget is exhausted or cell is very small.
            """
            nonlocal calls, global_best_x, global_best_f
            
            # 3^dim grid points
            grid_size = 3 ** dim
            min_half_width = 0.05
            
            while calls < budget and not check_deadline() and half_width > min_half_width:
                # Check if we have enough budget for a full grid pass
                # We need at least grid_size calls. If not, do a few random steps or break.
                if grid_size > (budget - calls):
                    # Not enough budget for full grid, do a few random local steps or break
                    # Actually, just break and let ES handle the final bits? 
                    # Or do a smaller grid? Let's just break to be safe and let ES refine.
                    break
                
                # Define grid points
                # linspace from center - half_width to center + half_width
                # Clip to [-5, 5]
                points = []
                # Generate all combinations
                # For dim=5, 3^5 = 243 points.
                # Use np.meshgrid or itertools.product
                # It is faster to generate indices and map them.
                indices = np.array(np.unravel_index(np.arange(grid_size), [3]*dim))
                
                # Map indices 0,1,2 to -1, 0, 1
                # Then scale by half_width and add to center
                # Offset: -1 -> -half_width, 0 -> 0, 1 -> half_width
                offsets = indices - 1  # -1, 0, 1
                
                # Generate all points
                # points[i] = center + offsets[i] * half_width
                # offsets is (dim, grid_size)
                # center is (dim,)
                # We want (grid_size, dim)
                pts = center[None, :] + offsets.T * half_width
                pts = np.clip(pts, -5, 5)
                
                local_best_pt = None
                local_best_val = float('inf')
                
                for pt in pts:
                    if calls >= budget:
                        break
                    if check_deadline():
                        break
                    fy = func(pt)
                    calls += 1
                    record(pt, fy)
                    
                    if fy < local_best_val:
                        local_best_val = fy
                        local_best_pt = pt.copy()
                
                if local_best_pt is None:
                    break
                    
                # Subdivide: new center is the best point, new half_width is half
                center = local_best_pt
                half_width *= 0.5
                
                # Optional: if the improvement is very small, we might stop early?
                # The grid always evaluates the center (index 1 in each dim is 0 offset).
                # So the center is always evaluated.
                
                # If we are getting very close, we might switch to ES for final polish?
                # The prompt suggests combining grid with ES.
                # Let's let the grid run until budget is low, then use ES for the last few steps?
                # Or just let the grid run until it breaks due to budget.
                # Since grid_size is fixed (243), and budget is 500, we can do at most 2 full grids.
                # 500 / 243 = 2.
                # So we do 2 levels of subdivision.
                # Level 1: 243 calls.
                # Level 2: 243 calls.
                # Total 486 calls.
                # Remaining 14 calls for ES.
                # This seems reasonable.
                
            return center, local_best_val

        # Phase 1: Initial exploration
        # Start at origin
        x0 = np.zeros(dim)
        fx0 = func(x0)
        calls += 1
        record(x0, fx0)

        # Phase 2: Multi-restart ES to find a good basin
        # Use a portion of the budget for this.
        # Let's say 200 calls for exploration.
        n_restarts = 4
        budget_per_restart = 40
        # Total 160 calls. Plus initial 1. Total 161.
        # Remaining ~339 for refinement.
        
        # Run first ES from origin
        if calls < budget:
            run_es(x0, fx0, budget_per_restart, sigma_init=2.0)

        # Additional restarts
        for restart in range(n_restarts - 1):
            if calls >= budget:
                break
            if check_deadline():
                break

            x_start = rng.uniform(-3, 3, dim)
            fx_start = func(x_start)
            calls += 1
            record(x_start, fx_start)

            remaining = budget - calls
            if remaining <= 0:
                break

            alloc = min(remaining, budget_per_restart)
            run_es(x_start, fx_start, alloc, sigma_init=1.5)

        # Phase 3: Grid-based refinement around global best
        if global_best_x is not None and calls < budget and not check_deadline():
            # Use the remaining budget for grid refinement
            # The grid function will use up to budget - calls
            run_grid_refinement(global_best_x, 1.0, budget - calls)
            
            # Phase 4: Final polish with ES if there is any budget left
            # The grid might stop early if budget is not a multiple of grid_size or if it breaks.
            # Let's use a small ES run to polish.
            if calls < budget and not check_deadline() and global_best_x is not None:
                remaining = budget - calls
                # Use a very small sigma for final polish
                run_es(global_best_x, global_best_f, remaining, sigma_init=0.1, window=5)

        return global_best_x, global_best_f
