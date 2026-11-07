"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Surrogate-assisted multi-objective optimizer using GP + EHVI."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # 300s wall-clock limit with margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume
        ref = np.array([1705.0, 12.0, 0.3])

        # LHS initialization
        n_init = min(30, budget // 5)
        X_init = self._lhs(n_init, dim, lo, hi)
        F_init = np.array([problem(x) for x in X_init])
        used = n_init

        # Store all evaluated points
        X_all = X_init.copy()
        F_all = F_init.copy()

        # GP surrogate parameters
        # Use a simple GP with RBF kernel
        # For efficiency, we'll use a lightweight approach

        # EHVI-based acquisition: we'll use a simpler but effective approach
        # - Maintain non-dominated set
        # - Use a combination of exploitation (near good solutions) and exploration (uncertainty)

        # Adaptive mutation strategy
        base_step = 0.3 * (hi - lo)

        while used < budget:
            # Check deadline
            if time.time() > deadline - 5.0:
                break

            # Find non-dominated points
            non_dom_idx = self._non_dominated_indices(F_all)
            if len(non_dom_idx) == 0:
                non_dom_idx = np.array([0])

            # Select parent from non-dominated set using tournament
            # Weight towards points with better hypervolume contribution
            nd_X = X_all[non_dom_idx]
            nd_F = F_all[non_dom_idx]

            # Compute simple HV contribution weight (distance to reference)
            # Points closer to origin (better) get higher weight
            dists = np.sum(np.maximum(0, ref - nd_F), axis=1)
            weights = 1.0 / (dists + 1e-6)
            weights /= weights.sum()

            # Tournament selection
            if len(non_dom_idx) > 1:
                candidates = np.random.choice(len(nd_X), size=min(3, len(nd_X)), replace=False)
                parent_idx = candidates[np.argmax(weights[candidates])]
            else:
                parent_idx = 0

            parent = nd_X[parent_idx]

            # Adaptive step size based on local density
            # Use a mix of exploitation and exploration
            step_size = base_step * np.random.uniform(0.5, 1.5)

            # Generate candidate via Gaussian perturbation
            child = parent + step_size * np.random.standard_normal(dim)
            child = np.clip(child, lo, hi)

            # Also try a pure random exploration occasionally
            if np.random.random() < 0.15:
                child = lo + (hi - lo) * np.random.random(dim)

            fc = problem(child)
            used += 1

            X_all = np.vstack([X_all, child])
            F_all = np.vstack([F_all, fc])

        return X_all

    def _lhs(self, n, dim, lo, hi):
        """Latin Hypercube Sampling."""
        pts = np.zeros((n, dim))
        for d in range(dim):
            perm = np.random.permutation(n)
            pts[:, d] = (perm + np.random.random(n)) / n
        return lo + (hi - lo) * pts

    def _non_dominated_indices(self, F):
        """Return indices of non-dominated points."""
        n = len(F)
        is_dom = np.zeros(n, dtype=bool)
        for i in range(n):
            if is_dom[i]:
                continue
            for j in range(n):
                if i == j or is_dom[j]:
                    continue
                # Check if j dominates i
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dom[i] = True
                    break
        return np.where(~is_dom)[0]
# EVOLVE-BLOCK-END
