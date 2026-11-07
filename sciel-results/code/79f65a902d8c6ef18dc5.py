"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG9 using a hybrid LHS + evolutionary approach."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, 10s safety margin

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Phase 1: Latin Hypercube Sampling for broad coverage
        # Use a fraction of the budget for LHS
        n_lhs = min(80, budget // 2)
        n_evo = budget - n_lhs

        # Generate LHS points
        X_lhs = np.zeros((n_lhs, dim))
        for d in range(dim):
            # Latin hypercube: permute [0, 1/n, ..., (n-1)/n] and add uniform jitter
            perm = np.random.permutation(n_lhs)
            X_lhs[:, d] = (perm + np.random.random(n_lhs)) / n_lhs
        X_lhs = lo + (hi - lo) * X_lhs

        # Evaluate LHS points
        F = np.zeros((n_lhs, n_obj))
        for i in range(n_lhs):
            F[i] = problem(X_lhs[i])

        # Track all evaluated points for hypervolume
        all_X = [X_lhs]
        all_F = [F]
        used = n_lhs

        # Phase 2: Evolutionary refinement
        # Initialize population with best non-dominated points from LHS
        pop_size = min(40, n_evo // 2)

        # Find non-dominated points from LHS
        non_dom_mask = self._find_non_dominated(F)
        non_dom_idx = np.where(non_dom_mask)[0]

        if len(non_dom_idx) < pop_size:
            # Pad with random points
            while len(non_dom_idx) < pop_size:
                non_dom_idx = np.append(non_dom_idx, np.random.randint(n_lhs))
        else:
            non_dom_idx = non_dom_idx[:pop_size]

        X_pop = X_lhs[non_dom_idx].copy()
        F_pop = F[non_dom_idx].copy()

        # Evolutionary loop
        while used < budget:
            if time.time() > deadline:
                break

            # Find non-dominated in population
            nd_mask = self._find_non_dominated(F_pop)
            nd_idx = np.where(nd_mask)[0]

            if len(nd_idx) == 0:
                nd_idx = np.arange(len(F_pop))

            # Select parent using tournament or random from non-dominated
            parent_idx = nd_idx[np.random.randint(len(nd_idx))]
            parent = X_pop[parent_idx].copy()

            # Mutation: adaptive step size
            # Use a mix of large and small steps
            if np.random.random() < 0.3:
                # Large jump
                step = 0.3 * (hi - lo)
                child = parent + step * np.random.standard_normal(dim)
            else:
                # Local refinement
                step = 0.05 * (hi - lo)
                child = parent + step * np.random.standard_normal(dim)

            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # Replace worst point in population
            # Use a simple criterion: sum of normalized objectives
            sums = np.sum(F_pop, axis=1)
            worst_idx = np.argmax(sums)

            X_pop[worst_idx] = child
            F_pop[worst_idx] = fc

            # Occasionally add to archive if non-dominated
            if used % 10 == 0:
                # Check if child is non-dominated w.r.t. all_F
                is_nd = True
                for f_arr in all_F:
                    for f in f_arr:
                        if np.all(f <= fc) and np.any(f < fc):
                            is_nd = False
                            break
                    if not is_nd:
                        break
                if is_nd:
                    all_X.append(child.reshape(1, -1))
                    all_F.append(fc.reshape(1, -1))

        # Combine all points for final result
        X_all = np.vstack(all_X)
        F_all = np.vstack(all_F)

        return X_all

    def _find_non_dominated(self, F):
        """Find non-dominated points in a set of objective vectors."""
        n = len(F)
        nd = np.ones(n, dtype=bool)
        for i in range(n):
            if not nd[i]:
                continue
            for j in range(n):
                if i == j or not nd[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    nd[i] = False
                    break
        return nd
# EVOLVE-BLOCK-END
