"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE34 using a population-based evolutionary strategy
    with Pareto dominance and adaptive step sizes."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        width = hi - lo

        X_archive = []
        F_archive = []
        eval_count = 0
        
        def is_dominated(f_new, F_list):
            """Check if f_new is dominated by any point in F_list."""
            for f in F_list:
                if all(f <= f_new) and any(f < f_new):
                    return True
            return False

        def safe_problem(x):
            nonlocal eval_count
            if eval_count >= self.budget:
                return [1e10, 1e10, 1e10]
            
            f = problem(x)
            eval_count += 1
            return f

        def add_to_archive(x, f):
            nonlocal X_archive, F_archive
            # Check if x is dominated by existing archive
            if is_dominated(f, F_archive):
                return False
            
            # Remove points from archive dominated by f
            new_X = []
            new_F = []
            for i in range(len(F_archive)):
                if not is_dominated(F_archive[i], [f]):
                    new_X.append(X_archive[i])
                    new_F.append(F_archive[i])
            
            X_archive = new_X
            F_archive = new_F
            X_archive.append(x)
            F_archive.append(f)
            return True

        # 1. Initial Stratified Sampling (Latin Hypercube-like)
        n_init = 20
        if n_init > self.budget:
            n_init = self.budget
            
        lhs_points = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            perm = np.random.permutation(n_init)
            points = (perm + np.random.rand(n_init)) / n_init
            lhs_points[:, d] = points
            
        X_init = lo + lhs_points * width
        
        for x in X_init:
            if eval_count >= self.budget:
                break
            f = safe_problem(x)
            add_to_archive(x, f)
            
        if eval_count >= self.budget:
            return X_archive

        # 2. Evolutionary Strategy Loop
        # Adaptive step size
        step_size = 0.1 * width
        # Population size for offspring generation per generation
        # We will iterate over the current Pareto front and generate offspring for each
        # or just a fixed number of offspring from random parents in the front.
        # Let's use a strategy where we pick parents from the front and generate offspring.
        
        while eval_count < self.budget:
            if time.time() > deadline:
                break
                
            # If archive is empty (shouldn't happen), fallback to random
            if not F_archive:
                x = lo + (hi - lo) * np.random.rand(self.dim)
                f = safe_problem(x)
                add_to_archive(x, f)
                continue

            # Generate a batch of offspring
            # Size of batch depends on remaining budget
            # Let's try to generate up to 10 offspring per loop iteration
            batch_size = min(10, self.budget - eval_count)
            
            new_points_found = 0
            
            for _ in range(batch_size):
                if eval_count >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Select a parent from the archive
                # Uniform selection from Pareto front
                idx = np.random.randint(len(F_archive))
                x_parent = X_archive[idx]
                
                # Generate offspring
                x_child = x_parent + np.random.normal(0, step_size, self.dim)
                
                # Clip to bounds
                x_child = np.clip(x_child, lo, hi)
                
                # Evaluate
                f_child = safe_problem(x_child)
                
                # Check if non-dominated
                if add_to_archive(x_child, f_child):
                    new_points_found += 1
            
            # Adapt step size
            # If no new points found, reduce step size (exploitation)
            # If many found, increase slightly (exploration)
            if new_points_found == 0:
                step_size *= 0.8
            elif new_points_found > batch_size / 2:
                step_size *= 1.1
                
            # Prevent step size from becoming too small or too large
            min_step = 0.001 * width
            max_step = 0.5 * width
            step_size = np.clip(step_size, min_step, max_step)
            
            # Optional: If archive is getting very large, we might need to manage it,
            # but for 200 budget, it should be small enough.

        return X_archive
# EVOLVE-BLOCK-END
