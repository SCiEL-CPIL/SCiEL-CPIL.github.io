"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Sobol quasi-random global search with adaptive centering and Rastrigin-aware refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Time guard
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 330s kill

        def time_ok():
            return time.time() < deadline

        calls = 0
        best_x = np.zeros(dim)
        best_fx = np.inf

        # Sobol sequence for systematic coverage
        # scipy.spatial.qhull has no Sobol, but we can use a simple quasi-random approach
        # Actually, let's use a stratified sampling approach that mimics Sobol coverage
        # We'll divide the space into a grid and sample within cells
        
        # Phase 1: Systematic exploration with adaptive centering
        # Start with a broad grid, then refine around the best point
        
        # Initial broad sampling
        n_initial = 50
        if budget > 100:
            n_initial = min(100, budget // 5)
        
        # Use stratified sampling for initial coverage
        # Divide each dimension into intervals and sample one point per cell
        n_per_dim = 4  # 4^5 = 1024 cells, but we only sample n_initial of them
        
        # Create grid points
        grid_points = []
        for i in range(n_initial):
            # Use a space-filling approach
            t = i / max(n_initial - 1, 1)
            # Use a simple low-discrepancy sequence (Halton-like)
            point = np.zeros(dim)
            for d in range(dim):
                # Halton sequence
                base = d + 2  # 2, 3, 5, 7, 11 for dim=5
                f = 1.0
                r = 0.0
                i_val = i + 1
                while i_val > 0:
                    f /= base
                    r += f * (i_val % base)
                    i_val //= base
                point[d] = -5 + 10 * r
            
            grid_points.append(point)
        
        # Evaluate initial stratified samples
        for x in grid_points:
            if not time_ok() or calls >= budget:
                break
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_fx = func(best_x)
            calls += 1

        # Phase 2: Adaptive local search around best point with Rastrigin-aware steps
        # Rastrigin has local minima at integer coordinates
        # Use a combination of:
        # 1. Fine local search around current best
        # 2. Integer-snapping attempts
        # 3. Occasional larger jumps to escape local minima
        
        x = best_x.copy()
        fx = best_fx
        sigma = 0.5
        no_improve = 0
        max_no_improve = 20
        
        # Track visited integer points to avoid redundant evaluations
        visited_integers = set()
        
        while calls < budget and time_ok():
            # Adaptive step size
            if no_improve >= max_no_improve:
                # Stagnation: try a larger jump or integer snap
                if sigma < 0.1:
                    # Try snapping to nearby integers
                    snapped = np.round(x).astype(int)
                    key = tuple(snapped)
                    if key not in visited_integers and calls < budget:
                        visited_integers.add(key)
                        x_test = np.clip(snapped.astype(float), -5, 5)
                        fx_test = func(x_test)
                        calls += 1
                        if fx_test < fx:
                            x = x_test
                            fx = fx_test
                            no_improve = 0
                            sigma = 0.3
                        else:
                            # Try a random jump
                            x = best_x + rng.uniform(-1.5, 1.5, dim)
                            x = np.clip(x, -5, 5)
                            fx = func(x)
                            calls += 1
                            if fx < best_fx:
                                best_fx = fx
                                best_x = x.copy()
                            else:
                                x = best_x.copy()
                                fx = best_fx
                            sigma = 0.8
                            no_improve = 0
                else:
                    sigma *= 0.7
                    no_improve = 0
            else:
                # Normal step
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    no_improve = 0
                    sigma *= 1.05  # Slight increase on success
                else:
                    sigma *= 0.95
                    no_improve += 1
        
        # Phase 3: Final Rastrigin-aware refinement
        # Try all nearby integer combinations
        if time_ok() and calls < budget - 5:
            # Get the current best and try integer neighbors
            base = np.round(x).astype(int)
            
            # Try all combinations of floor and ceil for each coordinate
            # This is 2^dim = 32 combinations for dim=5
            for mask in range(2 ** dim):
                if not time_ok() or calls >= budget:
                    break
                
                x_test = np.zeros(dim)
                for d in range(dim):
                    if mask & (1 << d):
                        x_test[d] = np.floor(x[d])
                    else:
                        x_test[d] = np.ceil(x[d])
                
                x_test = np.clip(x_test, -5, 5)
                key = tuple(np.round(x_test).astype(int))
                
                if key not in visited_integers:
                    visited_integers.add(key)
                    fx_test = func(x_test)
                    calls += 1
                    if fx_test < fx:
                        x = x_test
                        fx = fx_test
                        if fx < best_fx:
                            best_fx = fx
                            best_x = x.copy()
        
        # Phase 4: Final polish with very small steps
        if time_ok() and calls < budget:
            sigma_small = 0.02
            while calls < budget and time_ok():
                y = np.clip(x + sigma_small * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                    sigma_small *= 1.02
                else:
                    sigma_small *= 0.9
                    if sigma_small < 1e-10:
                        break

        return best_x, best_fx

# EVOLVE-BLOCK-END
