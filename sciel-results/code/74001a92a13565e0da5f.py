"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer using Differential Evolution (DE) with
    adaptive mutation and a non-dominated archive.
    
    Strategy:
    - DE is a population-based stochastic global optimization method that uses
      vector differences between population members to guide exploration.
    - For multi-objective problems, we use a non-dominated archive to store
      the best solutions found so far.
    - The mutation strategy uses DE/rand/1: v = x_r1 + F * (x_r2 - x_r3)
    - Crossover is binomial with rate CR.
    - Selection is based on non-domination: the offspring replaces the parent
      only if it dominates the parent, or if the parent is in the archive
      and the offspring is not dominated by the archive.
    - This approach is fundamentally different from MOPSO (which uses velocity
      and position updates) and NSGA-II (which uses non-dominated sorting and
      crowding distance). DE uses vector differences for exploration, which
      can be more effective in high-dimensional spaces with deceptive landscapes
      like WFG4.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Budget allocation
        # 200 evals total.
        # Initialization: 40 random points (population size 40).
        # Remaining: 160 evals for DE iterations.
        # Each generation produces 40 offspring, so 4 generations.
        
        pop_size = 40
        init_pop = 40
        
        # 1. Initialization
        rng = np.random.default_rng(0)
        X_init = lo + range_ * rng.random((init_pop, self.dim))
        F_init = np.array([problem(x) for x in X_init])
        
        # Initialize archive with all non-dominated points from initial population
        archive_X = []
        archive_F = []
        
        def is_dominated(a, b):
            """Check if a dominates b (a is better than b)."""
            return np.all(a <= b) and np.any(a < b)
        
        def add_to_archive(x, f):
            nonlocal archive_X, archive_F
            # Check if dominated by any in archive
            for af in archive_F:
                if is_dominated(af, f):
                    return False
            # Remove any in archive dominated by new point
            new_archive_X = []
            new_archive_F = []
            for ax, af in zip(archive_X, archive_F):
                if not is_dominated(f, af):
                    new_archive_X.append(ax)
                    new_archive_F.append(af)
            new_archive_X.append(x)
            new_archive_F.append(f)
            archive_X = new_archive_X
            archive_F = new_archive_F
            return True

        for x, f in zip(X_init, F_init):
            add_to_archive(x, f)

        # Current population for DE
        X = X_init.copy()
        F = F_init.copy()
        
        used_evals = init_pop
        
        # DE Parameters
        F_mut = 0.8  # Mutation scale factor
        CR = 0.9     # Crossover rate
        
        # 2. Main Loop
        while used_evals < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate offspring for each individual in the population
            offspring_X = np.zeros_like(X)
            offspring_F = np.zeros_like(F)
            
            for i in range(pop_size):
                if used_evals >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                # DE/rand/1 mutation: v = x_r1 + F * (x_r2 - x_r3)
                # Select three distinct random indices different from i
                r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                
                v = X[r1] + F_mut * (X[r2] - X[r3])
                
                # Binomial crossover
                trial = np.zeros(self.dim)
                j_rand = rng.integers(self.dim)
                for j in range(self.dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = X[i][j]
                
                # Clip to bounds
                trial = np.clip(trial, lo, hi)
                
                # Evaluate
                trial_F = problem(trial)
                used_evals += 1
                
                offspring_X[i] = trial
                offspring_F[i] = trial_F
                
                # Selection: offspring replaces parent if it dominates the parent
                # or if the parent is in the archive and the offspring is not dominated
                if is_dominated(trial_F, F[i]):
                    X[i] = trial
                    F[i] = trial_F
                    add_to_archive(trial, trial_F)
                else:
                    # Check if offspring is non-dominated with respect to archive
                    # If so, add to archive but don't replace parent
                    add_to_archive(trial, trial_F)
            
            # Check if we should stop early if no progress
            # For simplicity, we continue until budget is exhausted

        # Final step: Ensure we return something valid (though return value is ignored)
        # The evaluator records all problem(x) calls.
        return X
# EVOLVE-BLOCK-END
