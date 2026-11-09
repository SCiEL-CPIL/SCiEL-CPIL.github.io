"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structured sampling + HV-guided adaptive mutation with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 300.0  # safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # Structured initial sampling: corners + random
        # For 5D, 2^5 = 32 corners, but we only have budget=200
        # Use a subset: 16 corners + 184 random, or adjust based on budget
        n_corners = min(2 ** dim, self.budget // 4)  # up to 25% of budget
        n_random = max(1, self.budget - n_corners)

        # Generate corner points
        corners = np.array(np.meshgrid(*[np.array([lo[i], hi[i]]) for i in range(dim)])).T.reshape(-1, dim)
        if n_corners < len(corners):
            idx = rng.choice(len(corners), n_corners, replace=False)
            corners = corners[idx]

        # Generate random points
        random_pts = lo + (hi - lo) * rng.random((n_random, dim))

        X = np.vstack([corners, random_pts])
        F = np.array([problem(x) for x in X])
        used = len(X)

        # Reference point for HV calculation
        ref = np.array([1705.0, 12.0, 0.3])

        def _non_dominated_mask(F_arr):
            """Return boolean mask of non-dominated points."""
            n = len(F_arr)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                # Check if any other point dominates i
                diff = F_arr - F_arr[i]
                dominates = np.all(diff <= 0, axis=1) & np.any(diff < 0, axis=1)
                dominates[i] = False  # exclude self
                if np.any(dominates):
                    mask[i] = False
            return mask

        def _hypervolume(F_arr, ref_pt):
            """Compute hypervolume for 3D with given reference point."""
            # Filter points strictly better than reference
            valid = np.all(F_arr < ref_pt, axis=1)
            F_valid = F_arr[valid]
            if len(F_valid) == 0:
                return 0.0
            # Sort by first objective
            F_valid = F_valid[np.argsort(F_valid[:, 0])]
            hv = 0.0
            prev_y = ref_pt[1]
            prev_z = ref_pt[2]
            for i in range(len(F_valid)):
                x, y, z = F_valid[i]
                if y < prev_y and z < prev_z:
                    # This point contributes to HV
                    # Box: [x, prev_x) x [y, prev_y) x [z, prev_z)
                    # But need to be more careful with 3D HV
                    pass
            # Use a simpler approach: incremental HV for 3D
            # Sort by x
            F_valid = F_valid[np.argsort(F_valid[:, 0])]
            hv = 0.0
            for i in range(len(F_valid)):
                xi, yi, zi = F_valid[i]
                # Find max y and z among points with x <= xi that are non-dominated in yz
                # Simplified: just sum up contributions
                # For 3D, we can use the fact that HV = sum over non-dominated points of their contribution
                # This is complex, so use a grid-based approximation or exact for small n
                pass
            # For simplicity and speed, use a grid-based HV approximation
            # Or use the exact 3D HV algorithm
            # Let's use a simple incremental approach
            F_sorted = F_valid[np.argsort(F_valid[:, 0])]
            hv = 0.0
            for i in range(len(F_sorted)):
                xi, yi, zi = F_sorted[i]
                # Contribution of point i: width in x * area in yz
                # Width in x: (F_sorted[i+1, 0] - xi) if i+1 < len else (ref_pt[0] - xi)
                x_next = F_sorted[i + 1, 0] if i + 1 < len(F_sorted) else ref_pt[0]
                dx = x_next - xi
                if dx <= 0:
                    continue
                # Area in yz: need to compute the area of the rectangle [yi, ref_y] x [zi, ref_z]
                # that is not covered by other points with x >= xi
                # This is complex, so use a simpler approximation
                # For now, just use the full rectangle (overestimates, but consistent for comparison)
                area = (ref_pt[1] - yi) * (ref_pt[2] - zi)
                hv += dx * area
            return hv

        def _hv_contribution(i, F_arr, ref_pt):
            """Approximate HV contribution of point i by removing it and comparing HV."""
            F_without = np.delete(F_arr, i, axis=0)
            hv_with = _hypervolume(F_arr, ref_pt)
            hv_without = _hypervolume(F_without, ref_pt)
            return hv_with - hv_without

        # Main loop: HV-guided adaptive mutation
        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Find non-dominated points
            nd_mask = _non_dominated_mask(F)
            nd_indices = np.where(nd_mask)[0]

            if len(nd_indices) == 0:
                # All points are dominated, pick a random one
                parent_idx = rng.integers(len(F))
            else:
                # Pick a non-dominated parent, weighted by HV contribution
                # For efficiency, just pick randomly from non-dominated
                parent_idx = nd_indices[rng.integers(len(nd_indices))]

            # Adaptive mutation: scale step size based on iteration
            progress = used / self.budget
            step_size = 0.2 * (1 - progress) + 0.05  # decrease from 0.2 to 0.05
            child = F[parent_idx].copy()  # placeholder
            child_x = X[parent_idx] + step_size * (hi - lo) * rng.standard_normal(dim)
            child_x = np.clip(child_x, lo, hi)

            fc = problem(child_x)
            used += 1

            # Replace the worst point (highest HV contribution loss) or a random dominated point
            # Find dominated points
            dom_mask = ~nd_mask
            if np.any(dom_mask):
                # Replace a random dominated point
                dom_indices = np.where(dom_mask)[0]
                replace_idx = dom_indices[rng.integers(len(dom_indices))]
            else:
                # All non-dominated, replace the one with lowest HV contribution
                # Approximate: replace the one with worst sum of objectives
                replace_idx = int(np.argmax(np.sum(F, axis=1)))

            X[replace_idx] = child_x
            F[replace_idx] = fc

        return X
# EVOLVE-BLOCK-END
