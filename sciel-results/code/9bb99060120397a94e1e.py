# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings("ignore")


class Optimizer:
    """Multi-objective optimizer for WFG7 using a lightweight surrogate-guided approach."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        ref_point = np.array([3.0, 5.0, 7.0])

        # Storage for all evaluated points
        X_data = []
        F_data = []

        def add_point(x, f):
            nonlocal X_data, F_data
            X_data.append(np.array(x))
            F_data.append(np.array(f))

        def get_non_dominated(F):
            """Return indices of non-dominated points."""
            if len(F) == 0:
                return []
            n = len(F)
            nd_idx = []
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i != j:
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            dominated = True
                            break
                if not dominated:
                    nd_idx.append(i)
            return nd_idx

        def is_non_dominated(f, F):
            """Check if f is non-dominated with respect to set F."""
            for fj in F:
                if np.all(fj <= f) and np.any(fj < f):
                    return False
            return True

        # Phase 1: Initial sampling using Latin Hypercube + random
        n_init = min(30, budget)
        for i in range(n_init):
            # LHS-like sampling
            x = np.zeros(dim)
            for d in range(dim):
                u = (i + rng.random()) / n_init
                x[d] = lo[d] + u * (hi[d] - lo[d])
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            add_point(x, f)

        used = n_init

        # Get initial non-dominated set
        F_arr = np.array(F_data)
        nd_idx = get_non_dominated(F_arr)
        X_nd = [X_data[i] for i in nd_idx]
        F_nd = [F_data[i] for i in nd_idx]

        # Phase 2: Iterative improvement using local search around non-dominated points
        # and global exploration
        max_iter = budget - used
        for it in range(max_iter):
            if used >= budget:
                break

            # Strategy: pick a base point (non-dominated) and do local perturbation
            # Also occasionally do global random sampling

            # With some probability, do a global random sample
            if rng.random() < 0.15:
                x = lo + (hi - lo) * rng.random(dim)
                x = np.clip(x, lo, hi)
                f = np.array(problem(x))
                add_point(x, f)
                used += 1
                if is_non_dominated(f, np.array(F_nd)):
                    X_nd.append(x)
                    F_nd.append(f)
                continue

            # Pick a base from non-dominated set (or all points if empty)
            if len(X_nd) > 0:
                # Weight towards points that seem better (lower objectives)
                F_nd_arr = np.array(F_nd)
                # Simple selection: pick randomly but with bias
                base_idx = rng.integers(0, len(X_nd))
                base = X_nd[base_idx]
            else:
                base_idx = rng.integers(0, len(X_data))
                base = X_data[base_idx]

            # Local search: try several perturbations and pick the best
            n_trials = 5
            best_f = None
            best_x = None

            for t in range(n_trials):
                # Adaptive step size based on iteration
                step_scale = 0.05 + 0.1 * (1.0 - it / max(max_iter, 1))
                step = step_scale * (hi - lo)
                x = base + step * rng.standard_normal(dim)
                x = np.clip(x, lo, hi)

                f = np.array(problem(x))
                used += 1
                if used >= budget:
                    add_point(x, f)
                    break

                add_point(x, f)

                # Check if this is better than current best trial
                if best_f is None:
                    best_f = f
                    best_x = x
                else:
                    # Check dominance
                    if np.all(f <= best_f) and np.any(f < best_f):
                        best_f = f
                        best_x = x

            # Update non-dominated set
            if best_x is not None and best_f is not None:
                if is_non_dominated(best_f, np.array(F_nd)):
                    X_nd.append(best_x)
                    F_nd.append(best_f)

            # Prune non-dominated set to keep it manageable
            if len(F_nd) > 30:
                F_nd_arr = np.array(F_nd)
                nd_in_nd = get_non_dominated(F_nd_arr)
                X_nd = [X_nd[i] for i in nd_in_nd]
                F_nd = [F_nd[i] for i in nd_in_nd]

        return X_data

    def _non_dominated_mask(self, F):
        """Return mask of non-dominated points in F."""
        n = len(F)
        mask = np.ones(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i != j:
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        mask[i] = False
                        break
        return mask

    def _is_non_dominated(self, f, F):
        """Check if f is non-dominated with respect to set F."""
        for fj in F:
            if np.all(fj <= f) and np.any(fj < f):
                return False
        return True

    def _dominates(self, f1, f2):
        """Check if f1 dominates f2."""
        return np.all(f1 <= f2) and np.any(f1 < f2)
# EVOLVE-BLOCK-END
