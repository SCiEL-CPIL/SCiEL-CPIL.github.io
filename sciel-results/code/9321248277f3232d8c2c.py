"""Deterministic Latin Hypercube Sampling optimizer for WFG9."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-objective optimizer for WFG9 using LHS.

    Strategy: Generate the entire budget of points using deterministic Latin
    Hypercube Sampling (LHS). This ensures uniform coverage of the 12-dimensional
    search space, maximizing the probability of capturing the extent of the
    Pareto front for hypervolume calculation.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 300.0  # safety margin vs 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # Generate deterministic LHS points
        # For each dimension, we create a permutation of [0, N-1] and map it to the interval
        # We use a fixed seed for the permutations to ensure determinism
        rng = np.random.default_rng(42)
        
        # Pre-compute the LHS matrix
        # Shape: (budget, dim)
        # For each dimension d, we have a permutation perm_d of range(budget)
        # The value in cell (i, d) is (perm_d[i] + 0.5) / budget * span[d] + lo[d]
        
        lhs_points = np.zeros((self.budget, self.dim))
        for d in range(self.dim):
            perm = rng.permutation(self.budget)
            # Map permutation indices to the middle of each bin
            # This ensures points are well-distributed within the bins
            lhs_points[:, d] = (perm + 0.5) / self.budget * span[d] + lo[d]

        used = 0
        archive_x = []
        archive_f = []
        
        def is_dominated(f, Fs):
            # Returns True if any row in Fs dominates f
            if len(Fs) == 0:
                return False
            le = np.all(Fs <= f, axis=1)
            lt = np.any(Fs < f, axis=1)
            return bool(np.any(le & lt))

        def dominates(a, b):
            return bool(np.all(a <= b) and np.any(a < b))

        for i in range(self.budget):
            if time.time() > deadline - 10.0:
                break
            x = lhs_points[i]
            f = np.array(problem(x), dtype=float)
            if not np.all(np.isfinite(f)):
                f = np.array([1e3, 1e3, 1e3])
            used += 1
            
            # Maintain a non-dominated archive
            # If the new point is dominated by the current archive, skip it
            if is_dominated(f, np.array(archive_f)) if archive_f else False:
                continue
            
            # Remove points from archive that are dominated by the new point
            new_ax = []
            new_af = []
            for ax, af in zip(archive_x, archive_f):
                if dominates(f, af):
                    continue
                new_ax.append(ax)
                new_af.append(af)
            
            new_ax.append(x)
            new_af.append(f)
            archive_x = new_ax
            archive_f = new_af
            
            # Optional: Cap archive size if it grows too large, though with LHS
            # and 200 points, the non-dominated set is likely manageable.
            # If it does get large, we might need to prune, but for now we keep all
            # non-dominated points as they all contribute to hypervolume.
            
        # Return the archive points (or a dummy if empty)
        if archive_x:
            return np.array(archive_x)
        return np.array([lo + 0.5 * span])

# EVOLVE-BLOCK-END
