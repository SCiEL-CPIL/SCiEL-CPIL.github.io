import numpy as np
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.archive = []
        self.archive_F = []
        self.evals_used = 0

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # 1. Initial Sampling using Sobol sequence for deterministic spread
        # We need to evaluate 'budget' points total.
        # Let's start with a reasonable initial population size.
        # If budget is 200, let's start with 20 points.
        init_size = 20
        sampler = qmc.Sobol(d=self.dim, scramble=False)
        # Sobol requires power of 2 for best quality, but we can just take first N.
        # To be deterministic, we don't scramble.
        # We need to map [0,1] to [lo, hi].
        
        # Generate initial points
        # qmc.Sobol generates in [0,1].
        # We need init_size points.
        # Note: Sobol generator in scipy might need specific handling.
        # Let's use a simple deterministic grid or Halton if Sobol is tricky.
        # Halton is easier to implement manually or via scipy.stats.qmc.Halton.
        
        sampler = qmc.Halton(d=self.dim, scramble=False)
        points = sampler.random(init_size)
        X_init = lo + points * (hi - lo)
        
        F_init = np.array([problem(x) for x in X_init])
        self.evals_used = init_size
        
        # Store all evaluated points to ensure we don't lose good ones
        # We will maintain an archive of non-dominated points.
        
        # Helper to update archive
        def update_archive(X_new, F_new):
            # X_new: (N, dim), F_new: (N, n_obj)
            # Add to archive and remove dominated points
            X_all = np.vstack([self.archive, X_new]) if self.archive else X_new
            F_all = np.vstack([self.archive_F, F_new]) if self.archive_F else F_new
            
            # Find non-dominated points in F_all
            n = len(F_all)
            is_dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if np.all(F_all[j] <= F_all[i]) and np.any(F_all[j] < F_all[i]):
                        is_dominated[i] = True
                        break
            mask = ~is_dominated
            self.archive = X_all[mask]
            self.archive_F = F_all[mask]
            return self.archive, self.archive_F

        update_archive(X_init, F_init)
        
        # 2. Adaptive Refinement
        # We have budget - init_size evaluations left.
        # We will iteratively pick points from the archive and perturb them.
        # To be deterministic, we use a fixed RNG seed.
        rng = np.random.default_rng(123)
        
        # Directions for local search: Use a set of random directions generated once.
        # Or use axis-aligned moves.
        # Let's use a mix of small and large steps.
        
        while self.evals_used < self.budget:
            # Pick a point to refine. 
            # Strategy: Pick the point that has the largest "gap" to neighbors or just cycle through.
            # For simplicity and determinism, we cycle through the archive.
            # If archive is empty (shouldn't happen), break.
            if len(self.archive) == 0:
                break
                
            idx = self.evals_used % len(self.archive)
            x_current = self.archive[idx]
            f_current = self.archive_F[idx]
            
            # Generate a perturbation
            # We want to explore the front. 
            # A good strategy is to move in a random direction but with a step size that decays or is adaptive.
            # Let's try a fixed set of step sizes and random directions.
            
            # Generate random direction
            direction = rng.normal(size=self.dim)
            direction /= np.linalg.norm(direction)
            
            # Try a few step sizes
            # We have limited budget, so we should try to make each evaluation count.
            # Instead of blind line search, let's just take a step.
            # If the new point is non-dominated, it might improve the set.
            
            # To maximize hypervolume, we want to fill gaps.
            # Let's try a step size proportional to the distance to other points?
            # Or just a fixed small step.
            
            step_size = 0.1 * (hi - lo)
            x_new = x_current + step_size * direction
            x_new = np.clip(x_new, lo, hi)
            
            f_new = problem(x_new)
            self.evals_used += 1
            
            # Check if x_new is dominated by any point in the current archive
            # If not, add it to the archive (and remove any points it dominates)
            # This is a simple greedy non-dominated set update.
            
            # Optimization: Only update if it might be non-dominated.
            # Quick check: is it better than f_current in at least one obj and not worse in others?
            # Actually, it could be non-dominated even if worse than f_current in all objs? No, then it's dominated by f_current.
            # So if f_new is dominated by f_current, skip update.
            if not (np.all(f_new >= f_current) and np.any(f_new > f_current)):
                 # It might be non-dominated
                 update_archive(x_new.reshape(1, -1), f_new.reshape(1, -1))
            
        return self.archive
