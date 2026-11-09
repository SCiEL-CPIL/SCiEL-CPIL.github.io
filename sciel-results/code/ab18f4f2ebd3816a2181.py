"""Multi-objective optimizer on RE21 using HypE-style marginal hypervolume contribution selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """HypE-style multi-objective search with stratified initialization and adaptive sigma."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # 300s safety margin

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume calculation
        ref = np.array([3000.0, 0.06])

        # Population size
        pop_size = min(30, max(10, budget // 6))

        # Stratified initialization for better spread
        # Use Sobol-like stratification
        def stratified_sample(n, d, lo, hi, rng):
            """Generate stratified random samples in [lo, hi]^d."""
            samples = np.zeros((n, d))
            for i in range(n):
                for j in range(d):
                    # Stratify each dimension
                    stratum = (i * d + j) % n
                    frac = (stratum + rng.random()) / n
                    samples[i, j] = lo[j] + (hi[j] - lo[j]) * frac
            return samples

        X = stratified_sample(pop_size, dim, lo, hi, rng)
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points for hypervolume
        all_X = [x.copy() for x in X]
        all_F = [f.copy() for f in F]

        # Adaptive sigma
        sigma = 0.2  # initial step size relative to bounds
        min_sigma = 0.01
        max_sigma = 0.5
        stagnant_count = 0

        def hypervolume_2d(points, ref):
            """Calculate 2D hypervolume for a set of points with given reference point.
            Only points strictly better than ref in every objective count."""
            if len(points) == 0:
                return 0.0
            # Filter points that are strictly better than ref in every objective
            mask = np.all(points < ref, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0
            # Sort by first objective
            pts = pts[np.argsort(pts[:, 0])]
            hv = 0.0
            prev_y = ref[1]
            for i in range(len(pts)):
                x, y = pts[i]
                if y < prev_y:
                    hv += (ref[0] - x) * (prev_y - y)
                    prev_y = y
            return hv

        def marginal_hypervolume_contributions(F, ref):
            """Calculate marginal hypervolume contribution of each point.
            HV with point - HV without point."""
            n = len(F)
            if n == 0:
                return np.array([])
            contributions = np.zeros(n)
            for i in range(n):
                # HV with all points
                hv_all = hypervolume_2d(F, ref)
                # HV without point i
                F_without = np.delete(F, i, axis=0)
                hv_without = hypervolume_2d(F_without, ref)
                contributions[i] = hv_all - hv_without
            return contributions

        def non_dominated_sort(F):
            """Return indices of non-dominated points."""
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            return np.where(~dominated)[0]

        # Main evolution loop
        while used < budget:
            # Check deadline
            if time.time() > deadline - 5:
                break

            # Calculate marginal hypervolume contributions for selection
            contribs = marginal_hypervolume_contributions(F, ref)

            # Select parent with probability proportional to contribution
            # Add small noise to avoid zero probability for low contrib points
            contribs = np.maximum(contribs, 1e-10)
            probs = contribs / contribs.sum()
            parent_idx = rng.choice(len(F), p=probs)
            parent = X[parent_idx].copy()

            # Adaptive sigma: adjust based on acceptance and stagnation
            # If stagnant, increase sigma to explore
            if stagnant_count > 10:
                sigma = min(max_sigma, sigma * 1.2)
            elif stagnant_count > 5:
                sigma = min(max_sigma, sigma * 1.05)
            
            # Base sigma decreases slightly with progress
            progress = used / budget
            base_sigma = sigma * (1.0 - 0.3 * progress)
            
            # Mutate
            child = parent + base_sigma * (hi - lo) * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)

            # Evaluate
            fc = problem(child)
            used += 1

            # Track in all evaluations
            all_X.append(child.copy())
            all_F.append(fc.copy())

            # Check if child is non-dominated w.r.t current archive
            is_dominated = False
            for i in range(len(F)):
                if np.all(F[i] <= fc) and np.any(F[i] < fc):
                    is_dominated = True
                    break
            
            if not is_dominated:
                # Accept child into archive
                X = np.vstack([X, child])
                F = np.vstack([F, fc])
                stagnant_count = 0
                # Reduce sigma on success
                sigma = max(min_sigma, sigma * 0.9)
                
                # Prune archive if size > pop_size
                if len(F) > pop_size:
                    # Recalculate contributions for the expanded set
                    new_contribs = marginal_hypervolume_contributions(F, ref)
                    # Remove the point with the lowest marginal contribution
                    min_idx = np.argmin(new_contribs)
                    X = np.delete(X, min_idx, axis=0)
                    F = np.delete(F, min_idx, axis=0)
            else:
                # Reject child
                stagnant_count += 1
                # Increase sigma to escape local optima
                sigma = min(max_sigma, sigma * 1.02)

        return X
# EVOLVE-BLOCK-END
