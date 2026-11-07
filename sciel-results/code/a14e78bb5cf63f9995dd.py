"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """Multi-objective optimizer using Random Search with Non-Dominated Archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # Archive of non-dominated points: list of (x, f)
        archive_x = []
        archive_f = []
        evals = 0

        def is_dominated(f_new, f_existing):
            """Check if f_new is dominated by f_existing (minimization)."""
            # f_new is dominated if f_existing is <= f_new in all obj and < in at least one
            if np.all(f_existing <= f_new) and np.any(f_existing < f_new):
                return True
            return False

        def add_to_archive(x_new, f_new):
            """Add point to archive if non-dominated, remove dominated points."""
            nonlocal archive_x, archive_f
            
            # Check if new point is dominated by any existing point
            for f_ex in archive_f:
                if is_dominated(f_new, f_ex):
                    return False
            
            # Remove existing points dominated by the new point
            new_archive_x = []
            new_archive_f = []
            for x_ex, f_ex in zip(archive_x, archive_f):
                if not is_dominated(f_ex, f_new):
                    new_archive_x.append(x_ex)
                    new_archive_f.append(f_ex)
            
            new_archive_x.append(x_new)
            new_archive_f.append(f_new)
            
            archive_x = new_archive_x
            archive_f = new_archive_f
            return True

        def evaluate(x):
            nonlocal evals
            if evals >= budget:
                return None
            f = problem(x)
            evals += 1
            return f

        # Phase 1: Initialization with Latin Hypercube Sampling
        n_init = 50
        if n_init > budget:
            n_init = budget
            
        if n_init > 0:
            sampler = qmc.LatinHypercube(d=dim, seed=0)
            points = sampler.random(n_init)
            # Map to bounds
            points = lo + points * (hi - lo)
            
            for x in points:
                f = evaluate(x)
                if f is not None:
                    add_to_archive(x, f)

        # Phase 2: Local Refinement
        if evals < budget:
            # Adaptive step size
            base_sigma = 0.2
            min_sigma = 0.01
            
            # Number of refinement steps
            # We have budget - evals remaining.
            # Each step evaluates 1 point.
            # We iterate through the archive repeatedly.
            
            while evals < budget:
                # If archive is empty (should not happen), break
                if not archive_x:
                    break
                    
                # Select a parent. Random selection to maintain diversity.
                # Could also select based on hypervolume contribution, but random is robust.
                idx = np.random.randint(len(archive_x))
                x_parent = archive_x[idx]
                
                # Generate child
                # Decay sigma based on number of evaluations used
                progress = evals / budget
                sigma = base_sigma * (1 - progress) + min_sigma * progress
                sigma = max(sigma, min_sigma)
                
                x_child = x_parent + np.random.normal(0, sigma, dim)
                x_child = np.clip(x_child, lo, hi)
                
                f_child = evaluate(x_child)
                if f_child is None:
                    break
                
                add_to_archive(x_child, f_child)
                
                # Optional: If we find a new point, we might want to restart the loop 
                # to give it a chance to be a parent, but random selection handles this.
        
        return archive_x
# EVOLVE-BLOCK-END
