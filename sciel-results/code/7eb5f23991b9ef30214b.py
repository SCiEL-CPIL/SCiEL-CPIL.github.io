"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Archive-based multi-objective optimizer tuned for RE21 (budget=200)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo

        # Archive of non-dominated points
        archive_X = []
        archive_F = []

        def add_to_archive(x, f):
            """Add point to archive if not dominated, remove dominated points."""
            f = np.array(f, dtype=np.float64)
            x = np.array(x, dtype=np.float64)
            
            if archive_F:
                A = np.array(archive_F)
                # Check for exact duplicates
                if np.any(np.all(A == f, axis=1)):
                    return False
                
                # Check if f is dominated by any in archive
                # a <= f and a < f (strictly better in at least one)
                dominated_by_existing = np.any(np.all(A <= f, axis=1) & np.any(A < f, axis=1))
                if dominated_by_existing:
                    return False
                
                # Check which existing points are dominated by f
                # f <= a and f < a
                dominated_existing = np.all(f <= A, axis=1) & np.any(f < A, axis=1)
                mask = ~dominated_existing
                
                # Keep only non-dominated existing points
                if not np.all(mask):
                    archive_F[:] = [archive_F[i] for i in range(len(archive_F)) if mask[i]]
                    archive_X[:] = [archive_X[i] for i in range(len(archive_X)) if mask[i]]
            
            archive_F.append(f)
            archive_X.append(x)
            return True

        # Initialization: Latin Hypercube Sampling
        n_init = 40
        if n_init > self.budget:
            n_init = self.budget
        
        # Generate LHS points
        # For each dimension, we need a permutation of 0..n_init-1
        perms = [np.random.permutation(n_init) for _ in range(self.dim)]
        
        for i in range(n_init):
            x = np.zeros(self.dim)
            for d in range(self.dim):
                # Uniform distribution in the i-th stratum
                x[d] = lo[d] + span[d] * (perms[d][i] + np.random.random()) / n_init
            f = problem(x)
            add_to_archive(x, f)
        
        used = n_init

        # Main Loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Selection
            # 10% random exploration, 90% exploitation from archive
            if np.random.random() < 0.1 or not archive_X:
                x_parent = np.random.uniform(lo, hi)
            else:
                idx = np.random.randint(len(archive_X))
                x_parent = archive_X[idx].copy()
            
            # Mutation
            # Gaussian mutation
            # Start with larger sigma, decay it?
            # Simple: sigma = 0.1 * span
            sigma = 0.1 * span
            x_child = x_parent + np.random.normal(0, 1, self.dim) * sigma
            x_child = np.clip(x_child, lo, hi)
            
            # Evaluate
            f_child = problem(x_child)
            used += 1
            
            # Update Archive
            add_to_archive(x_child, f_child)

        # Return the archive points
        if archive_X:
            return np.array(archive_X)
        else:
            return np.array([np.random.uniform(lo, hi)])
# EVOLVE-BLOCK-END
