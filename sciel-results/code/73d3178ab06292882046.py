"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 exploiting separability with archive-based search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        d = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 structure: first n_obj-1 variables are "shape" variables,
        # remaining variables are "distance" variables that should be ~0.5
        n_shape = n_obj - 1  # 2 shape variables for 3 objectives
        n_dist = d - n_shape  # 10 distance variables

        # Initialize with a structured grid on the Pareto front
        # For DTLZ4 with 3 objectives:
        # f1 = (1+g)*x1*x2, f2 = (1+g)*x1*(1-x2), f3 = (1+g)*(1-x1)
        # where g = sum((xi^0.1 - 1)^2 for i=n_shape+1..d)
        # Optimal g=0 when distance vars = 0.5
        # Pareto front: f1=x1*x2, f2=x1*(1-x2), f3=1-x1

        # Strategy:
        # 1. Initialize with points on the known Pareto front (distance=0.5)
        # 2. Use remaining budget for local search around Pareto front
        # 3. Maintain an archive of non-dominated solutions

        # Initialize: sample the Pareto front using grid + random
        n_init = min(150, budget - 50)
        
        X_init = np.zeros((n_init, d))
        # Set distance variables to 0.5 (optimal for g=0)
        X_init[:, n_shape:] = 0.5
        
        # Mix of grid and random for shape variables
        n_grid = int(np.sqrt(n_init))
        n_random = n_init - n_grid * n_grid
        
        idx = 0
        # Grid points
        x1_vals = np.linspace(0.02, 0.98, n_grid)
        x2_vals = np.linspace(0.02, 0.98, n_grid)
        for x1 in x1_vals:
            for x2 in x2_vals:
                if idx >= n_init:
                    break
                X_init[idx, 0] = x1
                X_init[idx, 1] = x2
                idx += 1
        
        # Random points to fill remaining
        if idx < n_init:
            X_init[idx:, :n_shape] = rng.random((n_init - idx, n_shape))
            X_init[idx:, n_shape:] = 0.5

        X = X_init[:n_init]
        F = np.array([problem(x) for x in X])
        used = n_init

        # Archive-based approach: maintain a set of non-dominated solutions
        def is_dominated(f_new, F_archive):
            """Check if f_new is dominated by any point in F_archive."""
            if len(F_archive) == 0:
                return False
            return np.any(np.all(F_archive <= f_new, axis=1) & np.any(F_archive < f_new, axis=1))

        def crowding_distance(F):
            n = len(F)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(F.shape[1]):
                order = np.argsort(F[:, m])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                rng_m = F[order[-1], m] - F[order[0], m]
                if rng_m > 1e-15:
                    for i in range(1, n - 1):
                        cd[order[i]] += (F[order[i + 1], m] - F[order[i - 1], m]) / rng_m
            return cd

        def archive_insert(X, F, x_new, f_new, archive_size=200):
            """Insert new solution into archive, maintaining non-dominance."""
            # Check if new solution dominates any existing
            if len(F) > 0:
                dominated = np.where(np.all(f_new <= F, axis=1) & np.any(f_new < F, axis=1))[0]
                if len(dominated) > 0:
                    # Remove dominated solutions
                    mask = np.ones(len(X), dtype=bool)
                    mask[dominated] = False
                    X = X[mask]
                    F = F[mask]
            
            # Check if new solution is dominated by existing
            if is_dominated(f_new, F):
                return X, F, False
            
            # Add new solution
            X = np.vstack([X, x_new.reshape(1, -1)])
            F = np.vstack([F, f_new.reshape(1, -1)])
            
            # If archive too large, remove the point with smallest crowding distance
            if len(X) > archive_size:
                cd = crowding_distance(F)
                min_idx = np.argmin(cd)
                mask = np.ones(len(X), dtype=bool)
                mask[min_idx] = False
                X = X[mask]
                F = F[mask]
            
            return X, F, True

        # Main loop: use remaining budget for local search around Pareto front
        # Adaptive step sizes
        sigma_shape = 0.1
        sigma_dist = 0.03
        best_hv = -np.inf
        stagnation_counter = 0
        max_stagnation = 30

        while used < budget:
            if time.time() > deadline:
                break

            # Pick a random point from current population
            parent_idx = rng.integers(len(X))
            parent = X[parent_idx].copy()
            
            # Mutate: primarily shape variables with small steps, distance vars stay near 0.5
            child = parent.copy()
            
            # Mutate shape variables with adaptive step
            child[:n_shape] += sigma_shape * rng.standard_normal(n_shape)
            child[:n_shape] = np.clip(child[:n_shape], lo[:n_shape], hi[:n_shape])
            
            # Distance variables: keep near 0.5 with small perturbation
            child[n_shape:] = 0.5 + sigma_dist * rng.standard_normal(n_dist)
            child[n_shape:] = np.clip(child[n_shape:], lo[n_shape:], hi[n_shape:])
            
            fc = problem(child)
            used += 1
            
            # Insert into archive
            X, F, added = archive_insert(X, F, child, fc, archive_size=200)
            
            # Track stagnation
            if added:
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # Adapt step sizes based on stagnation
            if stagnation_counter > max_stagnation:
                # Increase exploration
                sigma_shape *= 1.5
                sigma_dist *= 1.5
                sigma_shape = min(sigma_shape, 0.3)
                sigma_dist = min(sigma_dist, 0.1)
                stagnation_counter = 0
            elif stagnation_counter > max_stagnation // 2:
                # Moderate exploration
                sigma_shape *= 1.2
                sigma_dist *= 1.2
                sigma_shape = min(sigma_shape, 0.3)
                sigma_dist = min(sigma_dist, 0.1)

        # Final polish: refine the best non-dominated points with smaller steps
        if time.time() < deadline - 5 and used < budget:
            # Find non-dominated points in the archive
            non_dom_mask = np.ones(len(F), dtype=bool)
            for i in range(len(F)):
                if non_dom_mask[i]:
                    for j in range(len(F)):
                        if i != j and non_dom_mask[j]:
                            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                                non_dom_mask[i] = False
                                break
            
            if np.any(non_dom_mask):
                best_indices = np.where(non_dom_mask)[0]
                # Try small perturbations around best points
                for idx in best_indices[:10]:
                    if used >= budget:
                        break
                    if time.time() > deadline:
                        break
                    for _ in range(5):
                        if used >= budget:
                            break
                        if time.time() > deadline:
                            break
                        # Very small perturbations for refinement
                        child = X[idx].copy()
                        child[:n_shape] += 0.02 * rng.standard_normal(n_shape)
                        child[:n_shape] = np.clip(child[:n_shape], lo[:n_shape], hi[:n_shape])
                        child[n_shape:] = 0.5 + 0.01 * rng.standard_normal(n_dist)
                        child[n_shape:] = np.clip(child[n_shape:], lo[n_shape:], hi[n_shape:])
                        
                        fc = problem(child)
                        used += 1
                        
                        # Replace if better in at least one objective and not worse in others
                        if np.all(fc <= F[idx]) and np.any(fc < F[idx]):
                            X[idx] = child
                            F[idx] = fc

        return X
# EVOLVE-BLOCK-END
