"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective evolutionary search with archive and deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # Archive of non-dominated points
        # X_archive: list of arrays
        # F_archive: list of arrays
        X_archive = []
        F_archive = []
        
        used = 0
        
        def dominates(a, b):
            # a dominates b if a <= b in all objectives and a < b in at least one
            # a and b are 1D arrays of length n_obj
            return np.all(a <= b) and np.any(a < b)

        def update_archive(x, f):
            nonlocal X_archive, F_archive
            # Check if f is dominated by any point in archive
            for fa in F_archive:
                if dominates(fa, f):
                    return False
            
            # Check if f dominates any points in archive
            new_x_archive = []
            new_f_archive = []
            for xa, fa in zip(X_archive, F_archive):
                if dominates(f, fa):
                    continue
                new_x_archive.append(xa)
                new_f_archive.append(fa)
            
            X_archive = new_x_archive
            F_archive = new_f_archive
            X_archive.append(x)
            F_archive.append(f)
            return True

        # 1. Initialization
        init_size = 20
        X_init = lo + span * rng.random((init_size, self.dim))
        F_init = np.array([problem(x) for x in X_init])
        used += init_size
        
        for x, f in zip(X_init, F_init):
            update_archive(x, f)
        
        # 2. Evolutionary Loop
        # We will use the archive as the parent pool.
        # If archive is empty (shouldn't happen), use random.
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Select parent from archive
            if len(X_archive) == 0:
                x_parent = lo + span * rng.random(self.dim)
            else:
                # Random selection from non-dominated front
                idx = rng.integers(len(X_archive))
                x_parent = X_archive[idx]
            
            # Mutation
            # Mix of Gaussian and uniform
            if rng.random() < 0.5:
                # Gaussian
                # Step size: 10% of span
                x_child = x_parent + 0.1 * span * rng.standard_normal(self.dim)
            else:
                # Uniform
                # Pick a random point in the box
                x_child = lo + span * rng.random(self.dim)
            
            x_child = np.clip(x_child, lo, hi)
            f_child = problem(x_child)
            used += 1
            
            update_archive(x_child, f_child)
            
            # Optional: If we have plenty of time and budget, do a local polish
            # But with 200 budget, the loop above is the main driver.
            # Let's add a simple local refinement if we are near the end of budget
            # and have time.
            
        # 3. Final Polish (if time allows)
        # Use remaining budget for fine-grained search around best points
        # "Best" points can be defined by their contribution to hypervolume or simply random sampling from archive
        # Let's just do a few more random mutations with smaller step size
        if used < self.budget and len(X_archive) > 0:
            polish_budget = min(self.budget - used, 20)
            for _ in range(polish_budget):
                if time.time() > deadline:
                    break
                idx = rng.integers(len(X_archive))
                x_parent = X_archive[idx]
                # Small Gaussian step
                x_child = x_parent + 0.05 * span * rng.standard_normal(self.dim)
                x_child = np.clip(x_child, lo, hi)
                f_child = problem(x_child)
                used += 1
                update_archive(x_child, f_child)

        # Return the archive (list of x vectors)
        # The problem statement says "Every returned objective vector is recorded... return value of __call__ is ignored"
        # But usually we return the points or the front. The seed program returns X.
        # Let's return the X_archive as a list or array.
        if X_archive:
            return np.array(X_archive)
        else:
            return np.array([])

# EVOLVE-BLOCK-END
