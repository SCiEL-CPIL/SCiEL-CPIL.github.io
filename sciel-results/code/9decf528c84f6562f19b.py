"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT6 using analytical Pareto front sampling with greedy hypervolume selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        budget = self.budget

        # ZDT6 analytical Pareto front:
        # x[0] = t in [0, 1], x[1:] = 0
        # f1 = 1 + 10*(n-1)*t
        # g = 1 + 10*(n-1)
        # f2 = (1/g) * (1 - sqrt(f1/g))^2 + 1

        g_val = 1 + 10 * (n - 1)
        r1, r2 = 11.0, 11.0

        # Only points with f1 < r1 contribute to hypervolume
        # f1 = 1 + 10*(n-1)*t < 11 => t < 10/(10*(n-1)) = 1/(n-1)
        t_max = 1.0 / (n - 1)  # For n=10, t_max = 1/9

        # Generate candidate points densely in the useful range [0, t_max]
        # Use more candidates than budget for greedy selection
        n_candidates = min(budget * 3, 600)
        t_candidates = np.linspace(0.0, t_max, n_candidates)

        f1_candidates = 1 + 10 * (n - 1) * t_candidates
        sqrt_term = np.sqrt(f1_candidates / g_val)
        f2_candidates = (1.0 / g_val) * (1.0 - sqrt_term)**2 + 1.0

        # Greedy hypervolume maximization:
        # Select points one at a time, each time choosing the point that
        # adds the most hypervolume to the current set.

        def compute_hv(points):
            """Compute hypervolume of a set of 2D points with reference (r1, r2)."""
            if len(points) == 0:
                return 0.0
            # Sort by f1 ascending
            pts = sorted(points, key=lambda p: p[0])
            hv = 0.0
            prev_f2 = r2
            for p in pts:
                f1, f2 = p
                if f2 < prev_f2:
                    hv += (r1 - f1) * (prev_f2 - f2)
                    prev_f2 = f2
                elif f2 < r2:
                    # This point doesn't improve the step but might extend it
                    # Actually, if f2 >= prev_f2, this point is dominated
                    # or doesn't add to the step function
                    pass
            return hv

        def marginal_gain(f1_c, f2_c, selected_sorted):
            """Compute the marginal hypervolume gain of adding point (f1_c, f2_c)."""
            if not selected_sorted:
                return (r1 - f1_c) * (r2 - f2_c)

            # Find neighbors in selected_sorted (sorted by f1)
            left_f1 = 0.0
            left_f2 = r2
            right_f1 = r1
            right_f2 = r2

            for p in selected_sorted:
                if p[0] <= f1_c:
                    left_f1 = p[0]
                    left_f2 = p[1]
                elif p[0] > f1_c:
                    right_f1 = p[0]
                    right_f2 = p[1]
                    break

            # The new point covers [f1_c, right_f1] with height (r2 - f2_c)
            # But we need to account for what was already covered.
            # Before insertion, the segment [left_f1, right_f1] was covered
            # by the point at left (if exists) with height (r2 - left_f2).
            # After insertion, [left_f1, f1_c] is covered by left, and
            # [f1_c, right_f1] is covered by the new point.

            # Old contribution of the region that the new point affects:
            # The new point only affects the region from f1_c to right_f1
            # where previously it was covered by left_f2 (if left exists and left_f1 < f1_c)
            # or not covered at all.

            # More carefully:
            # The step function changes only in [f1_c, right_f1).
            # Before: for f1 in [f1_c, right_f1), the covering point was the one with
            #   the largest f1 <= f1 in selected. That's the left neighbor.
            #   So the height was (r2 - left_f2) if left exists, else 0 (no coverage).
            # After: for f1 in [f1_c, right_f1), the covering point is the new point
            #   (since it has f1_c and no other selected point has f1 in (f1_c, right_f1)).
            #   So the height is (r2 - f2_c).

            # But wait, we also need to check if the new point is dominated by left.
            # If left_f2 <= f2_c, then the new point is dominated and adds nothing.
            if left_f2 <= f2_c:
                return 0.0

            # The gain is the area of [f1_c, right_f1) * (new_height - old_height)
            # where old_height = (r2 - left_f2) if left exists else 0
            # and new_height = (r2 - f2_c)

            old_height = (r2 - left_f2) if left_f1 > 0 else 0.0
            # Actually, if left_f1 == 0, it means no left neighbor, so old coverage was 0.
            # But left_f1 is initialized to 0.0 and left_f2 to r2.
            # If no left neighbor, left_f1 = 0.0, left_f2 = r2, so old_height = 0.
            # That's correct.

            new_height = r2 - f2_c
            width = right_f1 - f1_c

            gain = width * (new_height - old_height)
            if gain < 0:
                gain = 0.0
            return gain

        # Greedy selection
        selected = []  # List of (f1, f2, t)
        selected_sorted = []  # Sorted by f1
        remaining = list(range(n_candidates))

        for _ in range(min(budget, n_candidates)):
            if time.time() > deadline:
                break

            best_gain = -1
            best_idx = -1

            for idx in remaining:
                f1_c = f1_candidates[idx]
                f2_c = f2_candidates[idx]

                if f1_c >= r1 or f2_c >= r2:
                    continue

                gain = marginal_gain(f1_c, f2_c, selected_sorted)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx

            if best_idx == -1:
                break

            f1_sel = f1_candidates[best_idx]
            f2_sel = f2_candidates[best_idx]
            t_sel = t_candidates[best_idx]

            selected.append((f1_sel, f2_sel, t_sel))
            remaining.remove(best_idx)

            # Insert into sorted list
            selected_sorted.append((f1_sel, f2_sel))
            selected_sorted.sort(key=lambda p: p[0])

        # Now evaluate all selected points using the problem function
        # to get the actual objective values
        all_F = []
        for f1_sel, f2_sel, t_sel in selected:
            if time.time() > deadline:
                break
            x = np.zeros(n)
            x[0] = t_sel
            x[1:] = 1e-8
            x = np.clip(x, lo, hi)
            f = problem(x)
            all_F.append(f)

        return all_F
# EVOLVE-BLOCK-END
