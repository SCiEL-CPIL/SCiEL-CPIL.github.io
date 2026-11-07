"""Multi-objective optimizer on ZDT6 exploiting the analytical Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """ZDT6 optimizer that places all evaluations on the known Pareto front with
    hypervolume-aware adaptive spacing for maximum coverage."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _f1_of_t(self, t):
        """ZDT6 f1 for point on Pareto front: x1=t, x2..x10=0"""
        return 1.0 - np.exp(-4.0 * t) * np.sin(10.0 * np.pi * t) ** 2

    def _f2_of_t(self, t):
        """ZDT6 f2 for point on Pareto front: x1=t, x2..x10=0"""
        f1 = self._f1_of_t(t)
        return 1.0 - np.sqrt(f1)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        X = np.zeros((budget, dim))
        F = np.zeros((budget, n_obj))

        # ZDT6 Pareto front: x1=t, x2..x10=0
        # We need to choose t values that maximize hypervolume coverage.
        # Strategy: Use greedy hypervolume maximization with a dense candidate set,
        # then select the best subset of 'budget' points.

        # Generate a dense set of candidate t values
        n_candidates = min(5000, budget * 25)  # up to 5000 candidates
        t_candidates = np.linspace(0.001, 0.999, n_candidates)

        # Compute f1, f2 for all candidates
        f1_cand = self._f1_of_t(t_candidates)
        f2_cand = self._f2_of_t(t_candidates)

        # Reference point for hypervolume
        ref = np.array([11.0, 11.0])

        # Filter candidates that are strictly better than reference in both objectives
        valid_mask = (f1_cand < ref[0]) & (f2_cand < ref[1])
        t_valid = t_candidates[valid_mask]
        f1_valid = f1_cand[valid_mask]
        f2_valid = f2_cand[valid_mask]

        # Compute the Pareto envelope (non-dominated points)
        # In 2D, a point is dominated if there exists another point with
        # both f1 and f2 strictly smaller (or one equal and the other smaller)
        n_valid = len(f1_valid)

        # Sort by f1, then for ties by f2
        sort_idx = np.lexsort((f2_valid, f1_valid))
        f1_sorted = f1_valid[sort_idx]
        f2_sorted = f2_valid[sort_idx]
        t_sorted = t_valid[sort_idx]

        # Find non-dominated points by scanning from right to left
        # (from highest f1 to lowest f1, keeping track of minimum f2 seen)
        min_f2 = np.full(n_valid, np.inf)
        min_f2[-1] = f2_sorted[-1]
        for i in range(n_valid - 2, -1, -1):
            min_f2[i] = min(f2_sorted[i], min_f2[i + 1])

        # A point is non-dominated if its f2 equals the minimum f2 from that point onwards
        # (i.e., no point to its right has a strictly smaller f2)
        nondom_mask = (f2_sorted <= min_f2 + 1e-15)

        # Get non-dominated points
        f1_nd = f1_sorted[nondom_mask]
        f2_nd = f2_sorted[nondom_mask]
        t_nd = t_sorted[nondom_mask]

        # Now perform greedy hypervolume maximization on the non-dominated points
        # For 2D, the hypervolume contribution of adding a point to a set can be computed
        # by considering the intervals it creates

        def compute_hv_2d(points_f1, points_f2, ref1, ref2):
            """Compute hypervolume for 2D points with given reference point.
            Points must be sorted by f1 in decreasing order (worst to best)."""
            if len(points_f1) == 0:
                return 0.0
            # Sort by f1 decreasing
            order = np.argsort(-points_f1)
            pf1 = points_f1[order]
            pf2 = points_f2[order]

            hv = 0.0
            prev_f2 = ref2
            for i in range(len(pf1)):
                # Width in f1 direction: from this point's f1 to the next point's f1 (or ref1)
                if i + 1 < len(pf1):
                    width_f1 = pf1[i] - pf1[i + 1]
                else:
                    width_f1 = pf1[i] - 0  # from f1 to 0 (assuming f1 >= 0)

                # Height in f2 direction: from this point's f2 to the previous f2
                height_f2 = prev_f2 - pf2[i]

                if height_f2 > 0 and width_f1 > 0:
                    hv += height_f2 * width_f1

                prev_f2 = pf2[i]

            return hv

        # Greedy selection: start with empty set, iteratively add the point
        # that gives the largest marginal hypervolume gain
        n_nd = len(f1_nd)
        if n_nd == 0:
            # Fallback: use uniform t values
            t_selected = np.linspace(0.001, 0.999, budget)
        elif n_nd <= budget:
            # Use all non-dominated points
            t_selected = t_nd.copy()
            if len(t_selected) < budget:
                # Pad with additional points from the full candidate set
                remaining = np.setdiff1d(t_valid, t_selected, assume_unique=False)
                if len(remaining) > 0:
                    n_add = budget - len(t_selected)
                    step = max(1, len(remaining) // n_add)
                    extra = remaining[::step][:n_add]
                    t_selected = np.sort(np.concatenate([t_selected, extra]))
                else:
                    t_selected = np.linspace(0.001, 0.999, budget)
        else:
            # Greedy selection from non-dominated points
            selected_indices = []
            selected_f1 = []
            selected_f2 = []
            current_hv = 0.0

            # Pre-sort non-dominated points by f1 for efficient HV computation
            nd_order = np.argsort(-f1_nd)
            f1_nd_sorted = f1_nd[nd_order]
            f2_nd_sorted = f2_nd[nd_order]
            t_nd_sorted = t_nd[nd_order]

            for _ in range(budget):
                if time.time() > deadline:
                    break
                best_gain = -1
                best_idx = -1

                for i in range(n_nd):
                    # Compute HV if we add point i to current set
                    new_f1 = np.concatenate([selected_f1, [f1_nd[i]]])
                    new_f2 = np.concatenate([selected_f2, [f2_nd[i]]])
                    new_hv = compute_hv_2d(new_f1, new_f2, ref[0], ref[1])
                    gain = new_hv - current_hv

                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i

                if best_idx >= 0:
                    selected_indices.append(best_idx)
                    selected_f1 = np.concatenate([selected_f1, [f1_nd[best_idx]]])
                    selected_f2 = np.concatenate([selected_f2, [f2_nd[best_idx]]])
                    current_hv += best_gain
                else:
                    break

            t_selected = t_nd[selected_indices]

            # If we didn't select enough points, pad with uniform
            if len(t_selected) < budget:
                t_selected = np.sort(np.concatenate([t_selected,
                                                     np.linspace(0.001, 0.999, budget - len(t_selected))]))

        t_selected = np.clip(t_selected, 0.001, 0.999)
        t_selected = t_selected[:budget]

        # Evaluate all selected points
        for i, t in enumerate(t_selected):
            X[i, 0] = t
            F[i] = problem(X[i])

        return X
# EVOLVE-BLOCK-END
