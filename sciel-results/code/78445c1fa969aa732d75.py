"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ7 using structured initialization and local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ7 structure:
        # - First (n_obj - 1) = 2 variables define the Pareto front shape
        # - Remaining (dim - n_obj + 1) = 20 variables affect g() (distance to front)
        n_shape_vars = self.n_obj - 1  # 2
        n_g_vars = self.dim - n_shape_vars  # 20

        # Strategy:
        # 1. Use Sobol-like quasi-random points for shape variables
        # 2. Set g-variables to 0.5 (midpoint) to minimize g()
        # 3. Use remaining budget for local refinement of promising points

        budget = self.budget

        # Phase 1: Structured initialization with quasi-random shape variables
        # Use a simple LHS-like approach for shape variables
        n_init = min(budget // 2, 100)
        
        # Generate quasi-random points for shape variables using stratified sampling
        shape_points = np.zeros((n_init, n_shape_vars))
        for j in range(n_shape_vars):
            # Stratified sampling: divide [0,1] into n_init bins, pick one point per bin
            bins = np.linspace(0, 1, n_init + 1)
            shape_points[:, j] = (bins[:-1] + np.random.rand(n_init) * (bins[1:] - bins[:-1]))

        # Build initial population
        X = np.zeros((n_init, self.dim))
        X[:, :n_shape_vars] = lo[:n_shape_vars] + (hi[:n_shape_vars] - lo[:n_shape_vars]) * shape_points
        X[:, n_shape_vars:] = (lo[n_shape_vars:] + hi[n_shape_vars:]) / 2.0

        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = n_init

        # Track best points (non-dominated set approximation)
        # Keep track of all evaluated points
        all_X = X.copy()
        all_F = F.copy()

        # Phase 2: Local refinement using remaining budget
        remaining = budget - used
        if remaining <= 0:
            return all_X

        # Simple local search: perturb best points and evaluate
        # Use a small step size for refinement
        step_size = 0.05

        # Get non-dominated points from initial population
        def is_dominated(F_arr, f):
            """Check if any point in F_arr dominates f."""
            # A point dominates f if it is <= in all objectives and < in at least one
            better_or_equal = np.all(F_arr <= f, axis=1)
            strictly_better = np.any(F_arr < f, axis=1)
            return np.any(better_or_equal & strictly_better)

        # Iteratively refine
        for _ in range(remaining):
            if time.time() > deadline:
                break

            # Find non-dominated points
            non_dom_idx = []
            for i in range(len(all_F)):
                if not is_dominated(all_F, all_F[i]):
                    non_dom_idx.append(i)
            
            if not non_dom_idx:
                non_dom_idx = [np.argmin(np.sum(all_F, axis=1))]

            # Pick a random non-dominated point to refine
            parent_idx = non_dom_idx[np.random.randint(len(non_dom_idx))]
            parent = all_X[parent_idx].copy()

            # Generate a perturbed child
            # Focus perturbation on shape variables and some g-variables
            child = parent.copy()
            
            # Perturb shape variables with small steps
            for j in range(n_shape_vars):
                if np.random.rand() < 0.7:  # 70% chance to perturb each shape var
                    child[j] += np.random.randn() * step_size * (hi[j] - lo[j])
                    child[j] = np.clip(child[j], lo[j], hi[j])
            
            # Occasionally perturb g-variables (to explore different g values)
            if np.random.rand() < 0.3:
                g_idx = n_shape_vars + np.random.randint(n_g_vars)
                child[g_idx] += np.random.randn() * step_size * (hi[g_idx] - lo[g_idx])
                child[g_idx] = np.clip(child[g_idx], lo[g_idx], hi[g_idx])

            fc = problem(child)
            used += 1

            # Add to archive
            all_X = np.vstack([all_X, child])
            all_F = np.vstack([all_F, fc])

            # Periodically prune dominated points to keep archive manageable
            if len(all_F) > 500:
                # Keep only non-dominated points
                keep_idx = []
                for i in range(len(all_F)):
                    if not is_dominated(all_F, all_F[i]):
                        keep_idx.append(i)
                if keep_idx:
                    all_X = all_X[keep_idx]
                    all_F = all_F[keep_idx]

        return all_X
# EVOLVE-BLOCK-END
