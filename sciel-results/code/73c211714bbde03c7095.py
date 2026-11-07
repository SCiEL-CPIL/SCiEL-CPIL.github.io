"""Adaptive coordinate-wise grid refinement for BBOB f23 (Katsuura)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq


class Optimizer:
    """Adaptive coordinate-wise grid refinement for Katsuura."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        B = self.budget
        start_time = time.time()
        deadline = start_time + 270.0

        best_x = None
        best_f = np.inf
        calls = 0

        def safe_call(x):
            nonlocal calls, best_x, best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return x, fx

        def time_left():
            return deadline - time.time()

        # ============================================================
        # Phase 1: Coarse grid search
        # ============================================================
        # In 5D, a 4^5=1024 grid is too large. Use 3^5=243 points.
        # With budget 500, this leaves ~257 calls for refinement.
        n_coarse = 3
        grid_vals = np.linspace(-4.8, 4.8, n_coarse)

        coarse_points = []
        for i1 in range(n_coarse):
            for i2 in range(n_coarse):
                for i3 in range(n_coarse):
                    for i4 in range(n_coarse):
                        for i5 in range(n_coarse):
                            coarse_points.append([grid_vals[i1], grid_vals[i2], grid_vals[i3], grid_vals[i4], grid_vals[i5]])
        coarse_points = np.array(coarse_points, dtype=np.float64)

        # Evaluate all coarse points
        coarse_results = []
        for p in coarse_points:
            if calls >= B - 10 or time_left() < 10:
                break
            xp, fp = safe_call(p)
            coarse_results.append((fp, xp.copy()))

        # Sort by function value
        coarse_results.sort(key=lambda t: t[0])

        # ============================================================
        # Phase 2: Adaptive hypercube subdivision
        # ============================================================
        # Maintain a priority queue of hypercubes to explore.
        # Each entry: (f_min_estimate, hypercube_id, center, half_size)
        # We subdivide the hypercube with the lowest function value at its center.

        # Start with top hypercubes from coarse grid.
        # Each coarse point is the center of a cell of half-size = grid_spacing/2.
        grid_spacing = (4.8 - (-4.8)) / (n_coarse - 1)
        init_half = grid_spacing / 2.0 + 0.1

        # Priority queue: (center_value, counter, center, half_size)
        # We want to explore the best (lowest) centers first.
        counter = 0
        heap = []

        # Add top coarse points as initial hypercubes
        top_k_init = min(10, len(coarse_results))
        for k in range(top_k_init):
            fv, fc = coarse_results[k]
            heap.append((fv, counter, fc, init_half))
            counter += 1
        heapq.heapify(heap)

        # Track which hypercubes we've already refined to avoid duplicates
        # Use a set of rounded centers
        explored = set()
        for k in range(top_k_init):
            fv, fc = coarse_results[k]
            key = tuple(np.round(fc, 4))
            explored.add(key)

        # Subdivision parameters
        subdivide_factor = 2  # Split into 2^d sub-cubes
        min_half_size = 0.01
        max_refinements = 40  # Safety limit on total refinements

        refinement_count = 0
        rng = np.random.default_rng(777)

        while heap and calls < B - 15 and time_left() > 15 and refinement_count < max_refinements:
            # Pop the hypercube with the lowest center value
            fv, _, center, half = heapq.heappop(heap)

            # If half-size is too small, skip subdivision
            if half < min_half_size:
                continue

            # Subdivide: create 2^d sub-cubes
            sub_size = half / subdivide_factor
            sub_centers = []
            for mask in range(2 ** d):
                offset = np.array([
                    (1 if (mask >> i) & 1 else -1) * sub_size
                    for i in range(d)
                ], dtype=np.float64)
                sc = center + offset
                sc = np.clip(sc, -5.0, 5.0)
                sub_centers.append(sc)

            # Evaluate all sub-centers
            sub_results = []
            for sc in sub_centers:
                if calls >= B - 10 or time_left() < 10:
                    break
                key = tuple(np.round(sc, 4))
                if key in explored:
                    continue
                explored.add(key)
                xp, fp = safe_call(sc)
                sub_results.append((fp, xp.copy()))

            # Sort sub-results and push best ones back onto heap
            sub_results.sort(key=lambda t: t[0])
            n_push = min(3, len(sub_results))  # Push top 3 sub-cubes
            for k in range(n_push):
                sfv, sfc = sub_results[k]
                # Check if this sub-cube is promising enough
                if sfv < fv + 1.0:  # Only push if reasonably close to parent
                    heapq.heappush(heap, (sfv, counter, sfc, sub_size))
                    counter += 1

            refinement_count += 1

        # ============================================================
        # Phase 3: Local refinement of the best point found
        # ============================================================
        if best_x is not None and calls < B - 5 and time_left() > 5:
            # Coordinate-wise refinement: for each coordinate, try small perturbations
            x = best_x.copy()
            fx = best_f

            # Adaptive step sizes per coordinate
            steps = [0.1] * d
            remaining = B - calls - 5

            # Use remaining budget for coordinate descent with adaptive steps
            for _ in range(min(remaining, 150)):
                if calls >= B - 3 or time_left() < 3:
                    break

                improved = False
                for i in range(d):
                    if calls >= B - 3 or time_left() < 3:
                        break

                    # Try +step
                    xp = x.copy()
                    xp[i] += steps[i]
                    xp = np.clip(xp, -5.0, 5.0)
                    xp2, fp2 = safe_call(xp)

                    if fp2 < fx:
                        x = xp2.copy()
                        fx = fp2
                        improved = True
                        steps[i] *= 1.3
                    else:
                        # Try -step
                        xp = x.copy()
                        xp[i] -= steps[i]
                        xp = np.clip(xp, -5.0, 5.0)
                        xp2, fp2 = safe_call(xp)
                        if fp2 < fx:
                            x = xp2.copy()
                            fx = fp2
                            improved = True
                            steps[i] *= 1.3
                        else:
                            steps[i] *= 0.6

                    if steps[i] < 1e-6:
                        steps[i] = 1e-6

                if not improved:
                    # Shrink all steps
                    steps = [s * 0.5 for s in steps]
                    if max(steps) < 1e-7:
                        break

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
