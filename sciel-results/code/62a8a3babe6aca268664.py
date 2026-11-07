"""Multi-objective optimizer on DTLZ7 using Differential Evolution with a Pareto Archive."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Multi-objective optimizer using Differential Evolution (DE/rand/1) 
    with a Pareto Archive and density-based selection.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # Track evaluations
        eval_count = 0
        all_points = []
        all_objectives = []
        
        # Archive stores (x, f) for non-dominated solutions
        archive_X = []
        archive_F = []
        
        def eval_point(x):
            nonlocal eval_count
            eval_count += 1
            f = np.array(problem(x))
            all_points.append(x.copy())
            all_objectives.append(f)
            return f

        def dominates(f1, f2):
            """Check if f1 dominates f2 (minimization)."""
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def update_archive(x, f):
            """
            Update the Pareto archive.
            Returns True if the point was added or replaced.
            """
            nonlocal archive_X, archive_F
            n_arch = len(archive_F)
            
            # Check if f is dominated by any existing point
            for i in range(n_arch):
                if dominates(archive_F[i], f):
                    return False
            
            # Remove points dominated by f
            new_X = []
            new_F = []
            for i in range(n_arch):
                if not dominates(f, archive_F[i]):
                    new_X.append(archive_X[i])
                    new_F.append(archive_F[i])
            
            # If the archive is full, we might need to prune based on density
            # For simplicity and speed, we keep all non-dominated points 
            # but limit the population size for DE operations.
            # However, for DE, we usually maintain a population of fixed size.
            # This archive is for the final result and potentially for selection if we use it.
            # In this implementation, the 'population' for DE is separate from the archive.
            # We will just add to the global list of all evaluated points for the return value.
            # The 'archive' here is actually just the list of all non-dominated points found so far.
            
            archive_X = new_X
            archive_F = new_F
            
            # Add the new point
            archive_X.append(x)
            archive_F.append(f)
            
            # Optional: Limit archive size to prevent memory issues or slow dominance checks
            # DTLZ7 with 200 evals won't have a huge archive, but let's be safe.
            # If archive is too large, we can downsample. 
            # For 200 evals, max 200 points, O(N^2) dominance check is 40000 ops, which is fine.
            
            return True

        # Population size for DE
        pop_size = 40
        F_factor = 0.8  # Differential weight
        CR = 0.9  # Crossover rate

        # Initialize population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.zeros((pop_size, self.n_obj))
        
        for i in range(pop_size):
            F[i] = eval_point(X[i])
            update_archive(X[i], F[i])
        
        # DE Loop
        while eval_count < self.budget:
            for i in range(pop_size):
                if eval_count >= self.budget:
                    break
                
                # Select 3 distinct random individuals r1, r2, r3 different from i
                # We use a simple random selection
                indices = list(range(pop_size))
                indices.remove(i)
                r1, r2, r3 = rng.choice(indices, size=3, replace=False)
                
                # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                v = X[r1] + F_factor * (X[r2] - X[r3])
                
                # Crossover: Binary DE crossover
                # j_rand is a random dimension to guarantee at least one gene from v
                j_rand = rng.integers(self.dim)
                trial = X[i].copy()
                for j in range(self.dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Clip to bounds
                trial = np.clip(trial, lo, hi)
                
                # Evaluate
                f_trial = eval_point(trial)
                
                # Selection: Compare trial with current population member i
                # In multi-objective DE, we usually accept if:
                # 1. trial dominates X[i]
                # 2. X[i] does not dominate trial AND trial is better in terms of some diversity metric or simply random acceptance if non-dominated.
                # A common simple strategy: Accept if trial dominates X[i], OR if X[i] does not dominate trial.
                # This tends to fill the population with non-dominated or mutually non-dominated points.
                
                if dominates(f_trial, F[i]):
                    X[i] = trial
                    F[i] = f_trial
                    update_archive(trial, f_trial)
                elif not dominates(F[i], f_trial):
                    # If they are non-dominated, we might still replace to encourage exploration
                    # Or keep the old one. 
                    # Let's try a 50/50 chance or keep the one with better "rank" if we had ranks.
                    # Simplest: Keep old if it dominates. If non-dominated, replace with probability 0.5?
                    # Actually, a better strategy for DE is to maintain the population and only replace if strictly better or non-dominated.
                    # If we replace non-dominated points, we lose the old one from the population, but it's in the archive.
                    # Let's replace with probability 0.5 to maintain diversity in the population.
                    if rng.random() < 0.5:
                        X[i] = trial
                        F[i] = f_trial
                        update_archive(trial, f_trial)
                else:
                    # X[i] dominates trial, discard trial
                    pass
        
        return all_points
# EVOLVE-BLOCK-END
