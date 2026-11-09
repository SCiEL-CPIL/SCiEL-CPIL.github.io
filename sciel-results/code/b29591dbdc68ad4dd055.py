"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer combining archive-based search with LHS initialization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin before 330s limit
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Archive of non-dominated points: list of (x, f)
        archive = []
        
        def is_dominated(f_new, archive):
            for _, f_old in archive:
                if np.all(f_old <= f_new) and np.any(f_old < f_new):
                    return True
            return False
            
        def update_archive(x_new, f_new):
            # Check if new point is dominated
            if is_dominated(f_new, archive):
                return archive
            
            # Remove points dominated by the new point
            new_archive = []
            for x_old, f_old in archive:
                if np.all(f_new <= f_old) and np.any(f_new < f_old):
                    continue
                new_archive.append((x_old, f_old))
                
            new_archive.append((x_new, f_new))
            return new_archive

        # 1. Initialization: Latin Hypercube Sampling
        n_init = max(10, int(budget * 0.1))
        
        # Generate LHS points
        # For each dimension, divide [0,1] into n_init intervals, pick random point in each
        perm = np.argsort(np.argsort(np.random.rand(n_init, dim)), axis=0)
        u = (perm + np.random.rand(n_init, dim)) / n_init
        X_init = lo + (hi - lo) * u
        
        # Evaluate and populate archive
        used = 0
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = X_init[i]
            f = problem(x)
            used += 1
            # Basic validity check
            if np.any(np.isnan(f)) or np.any(np.isinf(f)):
                continue
            archive = update_archive(x, f)
            
        # 2. Iterative Refinement
        # Strategy: Population-based search with crossover and mutation.
        
        while used < budget:
            if time.time() > deadline:
                break
            
            if not archive:
                # If archive is empty, try random
                x = lo + (hi - lo) * np.random.rand(dim)
                f = problem(x)
                used += 1
                if np.any(np.isnan(f)) or np.any(np.isinf(f)):
                    continue
                archive = update_archive(x, f)
                continue
            
            # Select two parents from the archive
            if len(archive) == 1:
                parent1 = archive[0][0]
                parent2 = archive[0][0]
            else:
                idx1 = np.random.randint(0, len(archive))
                idx2 = np.random.randint(0, len(archive))
                parent1 = archive[idx1][0]
                parent2 = archive[idx2][0]
            
            # Crossover: BLX-Alpha in normalized space
            # Normalize parents to [0, 1]
            p1_norm = (parent1 - lo) / (hi - lo)
            p2_norm = (parent2 - lo) / (hi - lo)
            
            # BLX-Alpha
            alpha = 0.5
            lower = np.minimum(p1_norm, p2_norm) - alpha * np.abs(p1_norm - p2_norm)
            upper = np.maximum(p1_norm, p2_norm) + alpha * np.abs(p1_norm - p2_norm)
            lower = np.clip(lower, 0, 1)
            upper = np.clip(upper, 0, 1)
            
            x_child_norm = lower + (upper - lower) * np.random.rand(dim)
            
            # Denormalize
            x_child = lo + (hi - lo) * x_child_norm
            
            # Mutation: Gaussian noise
            # Adaptive step size
            progress = used / budget
            step_size = 0.1 * (1.0 - 0.5 * progress)
            
            noise = np.random.randn(dim) * step_size
            x_child = x_child + noise * (hi - lo)
            x_child = np.clip(x_child, lo, hi)
            
            f_child = problem(x_child)
            used += 1
            
            if np.any(np.isnan(f_child)) or np.any(np.isinf(f_child)):
                continue
                
            archive = update_archive(x_child, f_child)
            
        return archive
# EVOLVE-BLOCK-END
