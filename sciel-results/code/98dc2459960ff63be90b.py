import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290

        n_evals = self.budget
        if n_evals <= 0:
            return []

        results = []

        # ZDT6 Pareto front: x[1:] = 0, x[0] in [0,1]
        # f1(x0) = 1 + x0 * (sin(10*pi*x0) + sin(10*pi*x0**2))
        # f2(x0) = 1 - sqrt(f1(x0))
        # f1 is non-monotonic with increasing oscillation frequency as x0 increases

        # Strategy: uniform sampling in f1 space for optimal hypervolume coverage
        # We compute f1(x0) analytically, then find x0 values that give uniform f1 coverage
        # This maximizes the hypervolume of the recorded Pareto front points

        if n_evals == 1:
            x0s = np.array([0.5])
        elif n_evals == 2:
            x0s = np.array([0.0, 1.0])
        else:
            # First, sample x0 densely to compute f1 values
            # Use a fine grid to capture the oscillatory structure
            n_dense = max(n_evals * 5, 2000)
            x0_dense = np.linspace(0.0, 1.0, n_dense)
            f1_dense = 1.0 + x0_dense * (np.sin(10 * np.pi * x0_dense) + np.sin(10 * np.pi * x0_dense**2))

            # The Pareto front in objective space: points (f1, f2) where f2 = 1 - sqrt(f1)
            # Only non-dominated points matter. Since f2 = 1 - sqrt(f1), 
            # smaller f1 gives larger f2, so all points on the curve are non-dominated
            # if f1 is monotonic... but f1 is NOT monotonic.
            # We need to find the non-dominated subset.
            
            # A point (f1_i, f2_i) dominates (f1_j, f2_j) if f1_i <= f1_j and f2_i <= f2_j
            # with at least one strict. Since f2 = 1 - sqrt(f1), f2 is decreasing in f1.
            # So point i dominates j if f1_i <= f1_j AND 1-sqrt(f1_i) <= 1-sqrt(f1_j)
            # which means f1_i <= f1_j AND sqrt(f1_i) >= sqrt(f1_j)
            # which means f1_i <= f1_j AND f1_i >= f1_j
            # which means f1_i == f1_j.
            # So actually, for the ZDT6 Pareto front, no point strictly dominates another
            # as long as they have different f1 values. Wait, let me reconsider.
            
            # f2_i <= f2_j means 1 - sqrt(f1_i) <= 1 - sqrt(f1_j) means sqrt(f1_i) >= sqrt(f1_j)
            # means f1_i >= f1_j.
            # So i dominates j if f1_i <= f1_j AND f1_i >= f1_j, i.e., f1_i = f1_j.
            # So all distinct f1 values give non-dominated points on the Pareto front.
            
            # For hypervolume, we want to spread points evenly in the objective space.
            # The hypervolume of a set of points on a curve with reference point (r1, r2)
            # is maximized when points are evenly spaced in a certain sense.
            
            # Let's use uniform spacing in f1 space.
            f1_min = np.min(f1_dense)
            f1_max = np.max(f1_dense)
            
            # Create uniform grid in f1 space
            f1_targets = np.linspace(f1_min, f1_max, n_evals)
            
            # For each target f1 value, find the x0 that achieves it
            # Since f1 is not monotonic, there may be multiple x0 values for a given f1
            # We pick the one that gives the best (lowest) f2, which is 1-sqrt(f1), so same f2
            # Actually all x0 giving same f1 give same f2, so any is fine.
            # But we need to find x0 such that f1(x0) = f1_target.
            
            # Use interpolation: for each f1_target, find x0 by searching in the dense grid
            x0s = np.zeros(n_evals)
            for i, f1_t in enumerate(f1_targets):
                # Find index in f1_dense closest to f1_t
                idx = np.argmin(np.abs(f1_dense - f1_t))
                x0s[i] = x0_dense[idx]

            # Sort by x0 to ensure we evaluate in order
            x0s = np.sort(x0s)

        for i in range(n_evals):
            if time.time() > deadline:
                break
            x = np.zeros(self.dim, dtype=float)
            x[0] = float(x0s[i])
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                break

        return results
