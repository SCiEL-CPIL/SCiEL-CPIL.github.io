"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer using archive-based search with dominance
    checking and adaptive mutation for efficient front coverage.
    """
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 330s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Phase 1: Initial random population for broad coverage
        init_pop = min(50, budget)
        X_init = lo + (hi - lo) * np.random.random((init_pop, dim))
        F_init = np.array([problem(x) for x in X_init])
        evals_used = init_pop

        # Store all evaluated points (for scoring)
        all_X = [x.copy() for x in X_init]
        all_F = [f.copy() for f in F_init]

        # Archive: maintain only non-dominated points for efficient parent selection
        archive_X = [x.copy() for x in X_init]
        archive_F = [f.copy() for f in F_init]
        
        # Prune archive to keep only non-dominated points
        def prune_archive(arch_X, arch_F):
            n = len(arch_F)
            if n <= 1:
                return arch_X, arch_F
            F_arr = np.array(arch_F)
            keep = []
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated = True
                        break
                if not dominated:
                    keep.append(i)
            return [arch_X[i] for i in keep], [arch_F[i] for i in keep]

        archive_X, archive_F = prune_archive(archive_X, archive_F)

        # Phase 2: Evolutionary refinement with archive-based selection
        remaining = budget - evals_used
        mutation_scale_start = 0.25
        mutation_scale_end = 0.01

        while evals_used < budget:
            if time.time() > deadline:
                break

            # Select parent from archive (non-dominated points)
            if len(archive_X) > 0:
                parent_idx = np.random.randint(len(archive_X))
            else:
                parent_idx = np.random.randint(len(all_X))
                parent_X = all_X[parent_idx]
                parent_F = all_F[parent_idx]
            if len(archive_X) > 0:
                parent_X = archive_X[parent_idx]
                parent_F = archive_F[parent_idx]

            # Adaptive mutation step size
            progress = 1.0 - (budget - evals_used) / max(1, remaining)
            scale = mutation_scale_start + (mutation_scale_end - mutation_scale_start) * progress
            scale = max(scale, mutation_scale_end)

            # Gaussian mutation
            mutation = scale * (hi - lo) * np.random.standard_normal(dim)
            child_X = np.clip(parent_X + mutation, lo, hi)

            # Evaluate child
            child_F = problem(child_X)
            evals_used += 1

            # Add to full archive (for scoring)
            all_X.append(child_X)
            all_F.append(child_F)

            # Check if child is non-dominated with respect to archive
            is_dominated = False
            for af in archive_F:
                if np.all(af <= child_F) and np.any(af < child_F):
                    is_dominated = True
                    break
            
            if not is_dominated:
                # Add to archive and remove any dominated points
                archive_X.append(child_X)
                archive_F.append(child_F)
                # Prune archive
                archive_X, archive_F = prune_archive(archive_X, archive_F)

        return None
# EVOLVE-BLOCK-END
