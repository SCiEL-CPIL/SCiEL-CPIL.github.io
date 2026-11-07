"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for WFG9 using a structured random search approach.
    WFG9 has a known structure where the Pareto front is defined by specific variable
    combinations. This approach uses a hybrid of random sampling and local refinement
    to efficiently explore the search space within the tight budget.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # WFG9 specific structure: 
        # The Pareto front is typically achieved when certain variables are set to specific values.
        # For WFG9, the Pareto front is a 2D manifold in the 12D space.
        # A common heuristic is to fix some variables and optimize others.
        
        # Strategy: 
        # 1. Use a smaller population for faster exploration.
        # 2. Use a mutation strategy that encourages movement along the Pareto front.
        # 3. Keep a non-dominated archive to maintain diversity.
        
        pop_size = 30
        if self.budget < pop_size:
            pop_size = self.budget
            
        # Initialize population with random points
        X = (lo + (hi - lo) * np.random.random((pop_size, self.dim)))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Non-dominated archive
        archive_X = []
        archive_F = []
        
        def is_dominated(f, archive_F):
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False
        
        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)
        
        # Add initial population to archive
        for i in range(pop_size):
            if not is_dominated(F[i], archive_F):
                # Remove dominated points from archive
                new_archive_X = []
                new_archive_F = []
                for j in range(len(archive_F)):
                    if not dominates(F[i], archive_F[j]):
                        new_archive_X.append(archive_X[j])
                        new_archive_F.append(archive_F[j])
                archive_X.append(X[i])
                archive_F.append(F[i])
                archive_X = new_archive_X
                archive_F = new_archive_F
        
        # Main loop: structured random search with local refinement
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Choose a base point from the archive or population
            if archive_X and np.random.random() < 0.7:
                base_idx = np.random.randint(len(archive_X))
                base = archive_X[base_idx]
            else:
                base_idx = np.random.randint(pop_size)
                base = X[base_idx]
            
            # Generate a perturbation
            # Use a larger step size initially, then decrease
            progress = used / self.budget
            step_size = 0.5 * (1 - progress)
            
            # Random direction
            direction = np.random.randn(self.dim)
            direction /= np.linalg.norm(direction)
            
            # New candidate
            u = base + step_size * (hi - lo) * direction
            u = np.clip(u, lo, hi)
            
            # Evaluate
            f_u = problem(u)
            used += 1
            
            # Update population (simple replacement)
            if np.random.random() < 0.5:
                X[base_idx] = u
                F[base_idx] = f_u
            
            # Update archive
            if not is_dominated(f_u, archive_F):
                # Remove dominated points from archive
                new_archive_X = []
                new_archive_F = []
                for j in range(len(archive_F)):
                    if not dominates(f_u, archive_F[j]):
                        new_archive_X.append(archive_X[j])
                        new_archive_F.append(archive_F[j])
                archive_X.append(u)
                archive_F.append(f_u)
                archive_X = new_archive_X
                archive_F = new_archive_F
        
        return X
# EVOLVE-BLOCK-END
