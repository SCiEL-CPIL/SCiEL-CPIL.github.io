import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, max_evals, time_limit):
        """
        Nelder-Mead simplex method with adaptive step size.
        """
        dim = self.dim
        # Create initial simplex
        simplex = [x0.copy()]
        fx0 = func(x0)
        simplex_f = [fx0]
        eval_count = 1
        
        # Initial step size
        initial_step = 1.0
        for i in range(dim):
            xi = x0.copy()
            xi[i] += initial_step
            simplex.append(xi)
            simplex_f.append(func(xi))
            eval_count += 1
        
        # Sort simplex by function value
        idx = np.argsort(simplex_f)
        simplex = [simplex[i] for i in idx]
        simplex_f = [simplex_f[i] for i in idx]
        
        best_x = simplex[0].copy()
        best_fx = simplex_f[0]
        
        # Nelder-Mead parameters
        alpha = 1.0   # Reflection
        gamma = 2.0   # Expansion
        rho = 0.5     # Contraction
        sigma = 0.5   # Shrinkage
        
        start_time = time.time()
        
        while eval_count < max_evals:
            if time.time() - start_time > time_limit:
                break
            
            # Check for convergence
            if abs(simplex_f[-1] - simplex_f[0]) < 1e-10:
                break
            
            # Compute centroid of all but the worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            fxr = func(xr)
            eval_count += 1
            
            if fxr < simplex_f[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                fxe = func(xe)
                eval_count += 1
                
                if fxe < fxr:
                    simplex[-1] = xe
                    simplex_f[-1] = fxe
                else:
                    simplex[-1] = xr
                    simplex_f[-1] = fxr
            elif fxr < simplex_f[-2]:
                # Simple reflection
                simplex[-1] = xr
                simplex_f[-1] = fxr
            else:
                # Contraction
                if fxr < simplex_f[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - simplex[-1])
                
                fxc = func(xc)
                eval_count += 1
                
                if fxc < min(fxr, simplex_f[-1]):
                    simplex[-1] = xc
                    simplex_f[-1] = fxc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex_f[i] = func(simplex[i])
                        eval_count += 1
                        if eval_count >= max_evals:
                            break
            
            # Sort simplex again
            idx = np.argsort(simplex_f)
            simplex = [simplex[i] for i in idx]
            simplex_f = [simplex_f[i] for i in idx]
            
            # Update best
            if simplex_f[0] < best_fx:
                best_fx = simplex_f[0]
                best_x = simplex[0].copy()
        
        return best_x, best_fx, eval_count

    def __call__(self, func):
        # Deadline guard: 300s limit, use 290s to be safe
        deadline = time.time() + 290.0
        total_budget = self.budget
        dim = self.dim
        
        # Phase 1: Multi-start global exploration
        # Use 50% of budget for exploration
        exploration_budget = total_budget // 2
        n_starts = min(exploration_budget, 20)
        
        # Generate candidate starting points
        rng = np.random.default_rng(42)
        
        # Initial point at center
        candidates = [np.zeros(dim)]
        candidate_fxs = [func(np.zeros(dim))]
        
        # Grid-based initial points for better coverage
        grid_points = []
        for i in range(1, 4):
            for j in range(1, 4):
                for k in range(1, 4):
                    if dim >= 3:
                        pt = np.array([-3.0 + i * 2.0, -3.0 + j * 2.0, -3.0 + k * 2.0])
                        if dim == 5:
                            pt = np.append(pt, [0.0, 0.0])
                        grid_points.append(pt)
        
        # Add random points
        for _ in range(n_starts - 1):
            pt = rng.uniform(-5, 5, size=dim)
            grid_points.append(pt)
        
        # Evaluate and select top basins
        basin_results = []
        for pt in grid_points:
            if time.time() > deadline - 60:  # Leave time for exploitation
                break
            fx = func(pt)
            basin_results.append((fx, pt.copy()))
        
        # Sort by function value and select top basins
        basin_results.sort(key=lambda x: x[0])
        top_n = min(5, len(basin_results))
        
        # Phase 2: Local exploitation with Nelder-Mead
        best_x = basin_results[0][1].copy()
        best_fx = basin_results[0][0]
        
        remaining_budget = total_budget - len(grid_points) - 1
        time_per_start = (deadline - time.time()) / top_n
        
        for i in range(top_n):
            if i >= len(basin_results):
                break
            if time.time() > deadline - 5:
                break
            
            start_point = basin_results[i][1]
            time_limit = min(50.0, (deadline - time.time()) / (top_n - i))
            
            # Run Nelder-Mead from this basin
            local_x, local_fx, evals_used = self._nelder_mead(
                func, start_point, remaining_budget, time_limit
            )
            
            if local_fx < best_fx:
                best_fx = local_fx
                best_x = local_x.copy()
            
            remaining_budget -= evals_used
        
        return best_x, best_fx
