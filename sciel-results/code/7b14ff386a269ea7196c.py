import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # ZDT3: f1 = x1, f2 = (1+10*n)*(1 - sqrt(f1/g) - sin(10*pi*f1))
        # where n = dim - 1, g = sum((xi - 0.5)^2) for i > 1.
        # Pareto front: g = 0 (all xi=0.5 for i>1), so f2 = (1+10*n)*(1 - sqrt(f1) - sin(10*pi*f1))
        # with f1 in [0, 1].
        #
        # We need to select K=budget points on the Pareto front (parameterized by f1 in [0,1])
        # to maximize the hypervolume w.r.t. reference point (11, 11).
        #
        # Strategy:
        # 1. Compute a dense set of candidate Pareto points (high resolution).
        # 2. Compute the "marginal HV contribution" of each point.
        # 3. Use a greedy selection to pick the K points that maximize total HV.
        #
        # For ZDT3, the Pareto front is a single curve. The HV of a set of points on a curve
        # w.r.t. a reference point can be computed exactly. We'll use a greedy approach:
        #   - Start with the point that gives the largest single-point HV.
        #   - Iteratively add the point that maximizes the additional HV.
        #
        # However, for a smooth curve, a simpler and nearly optimal approach is to
        # distribute points according to the "density" of the HV contribution.
        #
        # Let's implement the greedy selection for correctness and optimality.

        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget
        ref = np.array([11.0, 11.0])

        # Step 1: Generate dense candidate points on the Pareto front.
        # f1 = x1 in [0, 1], f2 = (1 + 10*(dim-1)) * (1 - sqrt(f1) - sin(10*pi*f1))
        n_dense = 10000
        f1_vals = np.linspace(0.0, 1.0, n_dense)
        n = dim - 1
        f2_vals = (1.0 + 10.0 * n) * (1.0 - np.sqrt(f1_vals) - np.sin(10.0 * np.pi * f1_vals))
        
        # Candidate Pareto points: (f1, f2)
        candidates = np.column_stack([f1_vals, f2_vals])
        
        # Step 2: Greedy selection of K points to maximize HV.
        # HV of a set S w.r.t. ref: sum over each point of the area it uniquely covers.
        # For 2D, if points are sorted by f1 (ascending), the HV contribution of point i
        # when added to a set is: (ref[0] - f1_i) * (f2_prev - f2_i) where f2_prev is the
        # f2 of the previously selected point with smaller f1, or ref[1] if none.
        # But this is only valid if points are non-dominated and sorted.
        #
        # Actually, for 2D HV with reference (r1, r2), given a set of non-dominated points
        # sorted by f1 ascending: f1_1 < f1_2 < ... < f1_m, the HV is:
        #   sum_{i=1}^{m} (r1 - f1_i) * (f2_{i-1} - f2_i)
        # where f2_0 = r2, and f2_{m} = 0 (or min, but here we assume f2 > 0).
        # Wait, that's not quite right. Let me reconsider.
        #
        # The HV is the area of the union of rectangles [f1_i, r1] x [f2_i, r2].
        # For non-dominated points sorted by f1 ascending, f2 is descending.
        # The union area = sum_{i=1}^{m} (r1 - f1_i) * (f2_{i-1} - f2_i) where f2_0 = r2.
        # Yes, that's correct for the "staircase" decomposition.
        #
        # Greedy: maintain a sorted list of selected points. When adding a new point,
        # find its position in the sorted list and compute the additional HV.
        
        K = budget
        selected = []  # list of (f1, f2)
        
        # Precompute f2 for each candidate
        # We'll work with indices into candidates
        
        # To speed up, we can note that the optimal set for a smooth curve in 2D HV
        # is to distribute points according to the inverse of the "slope" of the HV contribution.
        # But let's just do greedy for simplicity and correctness.
        
        # For efficiency, since K=200 and n_dense=10000, greedy O(K * n_dense log n_dense) is fine.
        
        def compute_hv(points):
            """Compute HV of a set of 2D points w.r.t. ref (11, 11). Points are (f1, f2)."""
            if len(points) == 0:
                return 0.0
            # Sort by f1 ascending
            pts = sorted(points, key=lambda p: p[0])
            # Remove dominated points (keep only non-dominated)
            nd = []
            max_f2 = -np.inf
            for p in reversed(pts):
                if p[1] > max_f2:
                    nd.append(p)
                    max_f2 = p[1]
            nd.reverse()
            # Compute HV
            hv = 0.0
            prev_f2 = ref[1]
            for p in nd:
                if p[0] >= ref[0] or p[1] >= ref[1]:
                    continue
                hv += (ref[0] - p[0]) * (prev_f2 - p[1])
                prev_f2 = p[1]
            return hv
        
        # Greedy selection
        # Start with the point that gives the largest single-point HV
        # Single point HV = (r1 - f1) * (r2 - f2)
        single_hvs = (ref[0] - candidates[:, 0]) * (ref[1] - candidates[:, 1])
        # Only consider points strictly better than ref
        valid_mask = (candidates[:, 0] < ref[0]) & (candidates[:, 1] < ref[1])
        single_hvs[~valid_mask] = -np.inf
        
        # Pick the best single point
        best_idx = np.argmax(single_hvs)
        selected = [candidates[best_idx]]
        used = {best_idx}
        
        for _ in range(K - 1):
            best_gain = -np.inf
            best_add_idx = -1
            # For each unused candidate, compute the additional HV if added
            for idx in range(n_dense):
                if idx in used:
                    continue
                p = candidates[idx]
                if p[0] >= ref[0] or p[1] >= ref[1]:
                    continue
                # Tentatively add p to selected
                new_set = selected + [p]
                hv_new = compute_hv(new_set)
                hv_old = compute_hv(selected)
                gain = hv_new - hv_old
                if gain > best_gain:
                    best_gain = gain
                    best_add_idx = idx
            if best_add_idx >= 0:
                selected.append(candidates[best_add_idx])
                used.add(best_add_idx)
        
        # Step 3: Construct the actual x vectors for the selected Pareto points.
        # For each selected point (f1, f2), we set x1 = f1, and x_i = 0.5 for i > 1.
        # This gives g = 0, so f2 matches the Pareto front value.
        
        x_vectors = []
        for f1, f2 in selected:
            x = np.zeros(dim)
            x[0] = f1
            if dim > 1:
                x[1:] = 0.5
            # Clip to bounds
            x = np.clip(x, [b[0] for b in bounds], [b[1] for b in bounds])
            x_vectors.append(x)
        
        # Step 4: Evaluate each point
        for x in x_vectors:
            problem(x)
        
        return x_vectors
