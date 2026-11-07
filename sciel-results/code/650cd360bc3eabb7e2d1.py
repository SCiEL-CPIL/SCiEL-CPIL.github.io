"""Optimizer for BBOB-style f7 (Step ellipsoidal) on [-5,5]^5."""
import numpy as np
import time


class Optimizer:
    """Coarse-to-fine coordinate search with Nelder-Mead exploitation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        d = self.dim
        budget = self.budget

        call_count = 0

        def f(x):
            nonlocal call_count
            call_count += 1
            return func(np.asarray(x, dtype=float))

        # Phase 1: Coarse grid search to identify the basin
        # For step ellipsoidal, the function has plateaus; a coarse grid helps
        # find the right region before fine-tuning.
        # Use a 3-level grid in each dimension for initial exploration.
        grid_levels = 3
        grid_points = np.linspace(-5, 5, grid_levels)
        
        best_x = None
        best_f = np.inf
        
        # Evaluate grid points
        # With dim=5 and 3 levels, that's 3^5 = 243 points
        # We have 500 budget, so we can afford this
        for i0 in range(grid_levels):
            for i1 in range(grid_levels):
                for i2 in range(grid_levels):
                    for i3 in range(grid_levels):
                        for i4 in range(grid_levels):
                            if call_count >= budget:
                                break
                            x = np.array([grid_points[i0], grid_points[i1],
                                          grid_points[i2], grid_points[i3],
                                          grid_points[i4]])
                            fx = f(x)
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                        if call_count >= budget:
                            break
                    if call_count >= budget:
                        break
                if call_count >= budget:
                    break
            if call_count >= budget:
                break

        if best_x is None:
            # Fallback: random start
            rng = np.random.default_rng(0)
            best_x = rng.uniform(-5, 5, d)
            best_f = f(best_x)

        # Phase 2: Nelder-Mead refinement
        # Create initial simplex around best_x
        simplex = [best_x.copy()]
        for i in range(d):
            point = best_x.copy()
            step = 0.5 if best_x[i] >= 0 else -0.5
            point[i] += step
            point[i] = np.clip(point[i], -5, 5)
            simplex.append(point)
        
        simplex = [np.clip(p, -5, 5) for p in simplex]
        fvals = [f(p) for p in simplex]
        
        # Track best
        best_idx = int(np.argmin(fvals))
        if fvals[best_idx] < best_f:
            best_f = fvals[best_idx]
            best_x = simplex[best_idx].copy()

        # Nelder-Mead parameters
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma_nm = 0.5  # shrink

        # Main Nelder-Mead loop
        while call_count < budget:
            if time.time() > deadline:
                break
            
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Check convergence
            if len(set(fvals)) == 1 or (fvals[-1] - fvals[0]) < 1e-12:
                break
            
            # Check time
            if time.time() > deadline:
                break

            # Centroid of all points except worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = f(x_r)
            
            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = f(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                if f_r < fvals[-1]:
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    x_c = centroid - rho * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = f(x_c)
                
                if f_c < min(f_r, fvals[-1]):
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, d + 1):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = f(simplex[i])
            
            # Update best
            for i in range(d + 1):
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = simplex[i].copy()
            
            if call_count >= budget:
                break

        # Phase 3: If budget remains, do fine coordinate descent
        if call_count < budget and best_x is not None:
            # Adaptive step size coordinate descent
            step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
            for ss in step_sizes:
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break
                improved = True
                while improved and call_count < budget:
                    if time.time() > deadline:
                        break
                    improved = False
                    for i in range(d):
                        if call_count >= budget:
                            break
                        # Try positive step
                        x_pos = best_x.copy()
                        x_pos[i] += ss
                        x_pos[i] = np.clip(x_pos[i], -5, 5)
                        f_pos = f(x_pos)
                        if f_pos < best_f:
                            best_f = f_pos
                            best_x = x_pos
                            improved = True
                        # Try negative step
                        if call_count >= budget:
                            break
                        x_neg = best_x.copy()
                        x_neg[i] -= ss
                        x_neg[i] = np.clip(x_neg[i], -5, 5)
                        f_neg = f(x_neg)
                        if f_neg < best_f:
                            best_f = f_neg
                            best_x = x_neg
                            improved = True

        return best_x, best_f
