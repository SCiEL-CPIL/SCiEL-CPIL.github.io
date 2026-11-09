"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """
    Multi-objective optimizer using:
    1. Deterministic Sobol quasi-random sequence for uniform initial coverage.
    2. Local Gaussian perturbations around non-dominated points for refinement.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Phase 1: Sobol quasi-random initialization
        # Use a power of 2 for Sobol sampling
        n_sobol = 1
        while n_sobol < self.budget:
            n_sobol *= 2
        # Don't exceed budget
        n_sobol = min(n_sobol, self.budget)

        sobol = qmc.Sobol(d=self.dim, scramble=False)
        # Generate Sobol points in [0,1]^d
        points_unit = sobol.random(n_sobol)
        # Scale to bounds
        X = lo + points_unit * span
        F = np.array([problem(x) for x in X])
        used = n_sobol

        # Helper to check if a point is dominated by any in a list
        def is_dominated_by_any(f, F_list):
            if len(F_list) == 0:
                return False
            F_arr = np.array(F_list)
            better_or_equal = np.all(F_arr <= f, axis=1)
            strictly_better = np.any(F_arr < f, axis=1)
            return np.any(better_or_equal & strictly_better)

        # Helper to get non-dominated points from a list
        def get_nondominated(X_list, F_list):
            if len(F_list) == 0:
                return [], []
            F_arr = np.array(F_list)
            nd_indices = []
            for i, f in enumerate(F_list):
                # Check if f is dominated by any OTHER point
                # Exclude i from the check
                others_F = [F_list[j] for j in range(len(F_list)) if j != i]
                if not is_dominated_by_any(f, others_F):
                    nd_indices.append(i)
            
            nd_X = [X_list[i] for i in nd_indices]
            nd_F = [F_list[i] for i in nd_indices]
            return nd_X, nd_F

        # Initialize non-dominated set
        nd_X, nd_F = get_nondominated(X.tolist(), F.tolist())

        if len(nd_X) == 0:
            # Fallback
            nd_X = X.tolist()
            nd_F = F.tolist()

        # Phase 2: Local refinement
        remaining = self.budget - used

        if remaining > 0:
            # Use global np.random which is seeded by the evaluator
            # This allows different seeds to explore slightly differently
            step_size = 0.1 

            for i in range(remaining):
                # Select a parent from the non-dominated set
                if len(nd_X) == 0:
                    break
                idx = np.random.randint(len(nd_X))
                parent = nd_X[idx]

                # Gaussian perturbation
                perturbation = step_size * span * np.random.randn(self.dim)
                child = parent + perturbation
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1

                # Update non-dominated set
                # 1. Remove points dominated by child
                new_nd_X = []
                new_nd_F = []
                for x_p, f_p in zip(nd_X, nd_F):
                    # Check if f_p is dominated by fc
                    # fc dominates f_p if fc <= f_p and fc < f_p (at least one)
                    if not (np.all(fc <= f_p) and np.any(fc < f_p)):
                        new_nd_X.append(x_p)
                        new_nd_F.append(f_p)
                
                # 2. Check if child is dominated by any remaining point
                child_is_dominated = False
                for f_p in new_nd_F:
                    if np.all(f_p <= fc) and np.any(f_p < fc):
                        child_is_dominated = True
                        break
                
                if not child_is_dominated:
                    new_nd_X.append(child)
                    new_nd_F.append(fc)
                
                nd_X = new_nd_X
                nd_F = new_nd_F

        return X
# EVOLVE-BLOCK-END
