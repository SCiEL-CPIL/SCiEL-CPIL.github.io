"""Multi-objective optimizer on WFG4 using constrained Chebyshev scalarization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Constrained Chebyshev scalarization with smooth epigraph local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 290.0  # Safety margin from 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for scalarization
        ref_obj = np.array([3.0, 5.0, 7.0])

        # Generate systematic weight vectors along the Pareto front's convex hull
        # For 3 objectives, we use barycentric coordinates that sum to 1
        n_weights = min(20, budget // 6)
        weights = []
        # Generate points on the 2-simplex (sum of weights = 1)
        # Use a grid approach for systematic coverage
        n_per_axis = max(4, int(np.ceil(n_weights ** 0.5)))
        step = 1.0 / (n_per_axis - 1) if n_per_axis > 1 else 1.0
        for i in range(n_per_axis):
            for j in range(n_per_axis - i):
                w1 = i * step
                w2 = j * step
                w3 = 1.0 - w1 - w2
                if w3 >= -1e-9:
                    weights.append(np.array([max(0, w1), max(0, w2), max(0, w3)]))
        
        # Add some random weights for diversity
        rng = np.random.default_rng(42)
        while len(weights) < n_weights:
            w = rng.random(n_obj)
            w = w / w.sum()
            weights.append(w)
        
        weights = np.array(weights[:n_weights])

        # Archive of non-dominated points
        archive_X = []
        archive_F = []

        def is_pareto_optimal(f_new, F_archive):
            """Check if f_new is non-dominated with respect to F_archive."""
            if len(F_archive) == 0:
                return True
            F_arr = np.array(F_archive)
            # f_new is dominated if there exists a point in F_archive that is better in all objectives
            # and strictly better in at least one
            for f in F_arr:
                if np.all(f <= f_new) and np.any(f < f_new):
                    return False
            return True

        def add_to_archive(x, f):
            """Add point to archive if non-dominated."""
            if is_pareto_optimal(f, archive_F):
                archive_X.append(x.copy())
                archive_F.append(f.copy())
                # Remove any points in archive dominated by the new point
                new_archive_X = []
                new_archive_F = []
                for xi, fi in zip(archive_X, archive_F):
                    if np.all(f <= fi) and np.any(f < fi):
                        continue  # fi is dominated by f
                    new_archive_X.append(xi)
                    new_archive_F.append(fi)
                archive_X.clear()
                archive_F.clear()
                archive_X.extend(new_archive_X)
                archive_F.extend(new_archive_F)

        # Evaluate initial population
        used = 0
        pop_size = min(12, budget // 8)
        X_init = lo + (hi - lo) * rng.random((pop_size, dim))
        for x in X_init:
            if used >= budget:
                break
            if time.time() - start_time > deadline:
                break
            f = np.array(problem(x))
            used += 1
            add_to_archive(x, f)

        # Smooth Chebyshev scalarization using log-sum-exp for soft-max
        def soft_max(w, f, temp=0.1):
            """Soft-max approximation of max(w_i * f_i) using log-sum-exp."""
            z = w * f
            return temp * np.log(np.sum(np.exp(z / temp)))

        # For each weight vector, solve the constrained Chebyshev scalarization
        # min max_i (w_i * f_i)  s.t. f_i <= ref_obj_i for all i
        # We use a smooth local search approach from multiple starting points

        n_restarts_per_weight = max(2, budget // (len(weights) * 4))
        
        for w in weights:
            if used >= budget:
                break
            if time.time() - start_time > deadline:
                break
            
            # Try multiple starting points for this weight
            for restart in range(n_restarts_per_weight):
                if used >= budget:
                    break
                if time.time() - start_time > deadline:
                    break
                
                # Start from a random point or from the best archive point
                if rng.random() < 0.5 and len(archive_X) > 0:
                    x_cur = archive_X[rng.integers(len(archive_X))].copy()
                else:
                    x_cur = lo + (hi - lo) * rng.random(dim)
                
                f_cur = np.array(problem(x_cur))
                used += 1
                
                # Local search with adaptive step size using smooth objective
                step_size = 0.05 * (hi - lo)
                n_local_steps = max(3, budget // (len(weights) * n_restarts_per_weight * 3))
                
                for step in range(n_local_steps):
                    if used >= budget:
                        break
                    if time.time() - start_time > deadline:
                        break
                    
                    # Try several candidate moves
                    best_x = x_cur
                    best_f = f_cur
                    best_scalar = soft_max(w, f_cur)
                    
                    n_candidates = 6
                    for c in range(n_candidates):
                        if used >= budget:
                            break
                        if time.time() - start_time > deadline:
                            break
                        
                        # Random perturbation
                        delta = step_size * rng.standard_normal(dim)
                        x_candidate = np.clip(x_cur + delta, lo, hi)
                        f_candidate = np.array(problem(x_candidate))
                        used += 1
                        
                        # Check constraint: must be within reference point
                        if np.any(f_candidate >= ref_obj):
                            continue
                        
                        scalar_val = soft_max(w, f_candidate)
                        if scalar_val < best_scalar:
                            best_scalar = scalar_val
                            best_x = x_candidate
                            best_f = f_candidate
                    
                    if not np.array_equal(best_x, x_cur):
                        x_cur = best_x
                        f_cur = best_f
                        add_to_archive(x_cur, f_cur)
                    else:
                        # No improvement, shrink step size
                        step_size *= 0.8
                        if np.all(step_size < 1e-7 * (hi - lo)):
                            break

        # If we have budget left, do some random exploration
        while used < budget:
            if time.time() - start_time > deadline:
                break
            x_rand = lo + (hi - lo) * rng.random(dim)
            f_rand = np.array(problem(x_rand))
            used += 1
            if np.all(f_rand < ref_obj):
                add_to_archive(x_rand, f_rand)

        # Return the archive (or a subset if too large)
        if len(archive_X) == 0:
            # Fallback: return best single point
            if used > 0:
                # We didn't track the best, so return an empty array
                return np.array([])
            return np.array([])
        
        # Keep at most budget points
        if len(archive_X) > budget:
            # Select diverse points from archive
            idx = np.random.default_rng(0).permutation(len(archive_X))[:budget]
            archive_X = [archive_X[i] for i in idx]
        
        return np.array(archive_X) if archive_X else np.array([])
# EVOLVE-BLOCK-END
