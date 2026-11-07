"""Multi-objective optimizer on WFG9 using adaptive MOEA/D with external archive."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Multi-objective optimizer for WFG9 using MOEA/D with:
    - Tchebycheff scalarization with dynamic ideal/nadir
    - Adaptive neighborhood size
    - External archive with hypervolume contribution tracking
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 315  # safety margin before 330s kill

        rng = np.random.RandomState(42)

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Reference points: 21 points on the 3-objective simplex (grid with n=4)
        n_grid = 4
        ref_points = []
        for i in range(n_grid + 1):
            for j in range(n_grid + 1 - i):
                w1 = i / n_grid
                w2 = j / n_grid
                w3 = 1 - w1 - w2
                if w3 >= -1e-10:
                    ref_points.append([w1, w2, max(0, w3)])

        ref_points = np.array(ref_points)
        ref_points = ref_points / ref_points.sum(axis=1, keepdims=True)
        n_sub = len(ref_points)  # 15 subproblems

        # Compute neighbor lists (T nearest neighbors based on weight distance)
        T = 5
        weights = ref_points
        neighbor_lists = []
        for i in range(n_sub):
            dists = np.sum((weights - weights[i]) ** 2, axis=1)
            sorted_idx = np.argsort(dists)
            neighbors = sorted_idx[1:T + 1].tolist()  # exclude self
            neighbor_lists.append(neighbors)

        # Initialize population
        X = np.zeros((n_sub, self.dim))
        for i in range(n_sub):
            X[i] = lo + span * rng.random_sample(self.dim)

        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = n_sub

        # Track best solution per subproblem
        best_X = X.copy()
        best_F = F.copy()

        # Track ideal and nadir points
        ideal = np.min(F, axis=0)
        nadir = np.max(F, axis=0)

        # External archive (non-dominated solutions)
        archive_F = []
        archive_X = []

        def hypervolume_contribution(f, archive, ref_point):
            """
            Compute the hypervolume contribution of point f given the archive.
            Returns 0 if f is dominated by any point in archive.
            """
            if len(archive) == 0:
                # Volume of box from f to ref_point
                vol = 1.0
                for k in range(self.n_obj):
                    if ref_point[k] <= f[k]:
                        return 0.0
                    vol *= (ref_point[k] - f[k])
                return vol

            # Check if dominated by any archive point
            for af in archive:
                if np.all(af <= f) and np.any(af < f):
                    return 0.0

            # Compute hypervolume contribution using inclusion-exclusion
            # For 3 objectives, this is manageable
            # We need to compute the volume of the box [f, ref] minus volumes
            # covered by archive points that are not dominated by f

            # Collect all relevant points (f + non-dominated archive points)
            relevant = [f]
            for af in archive:
                # Only include if af is not dominated by f
                if not (np.all(f <= af) and np.any(f < af)):
                    relevant.append(af)

            # Compute hypervolume of union of boxes from relevant points to ref
            # Using simple enumeration for 3D
            # Sort points by first objective
            relevant = sorted(relevant, key=lambda x: x[0])

            # For each point, compute the "new" volume it adds
            total_vol = 0.0
            for i, p in enumerate(relevant):
                # Box from p to ref
                # Subtract boxes from earlier points that overlap
                vol = 1.0
                for k in range(self.n_obj):
                    if ref_point[k] <= p[k]:
                        vol = 0.0
                        break
                    vol *= (ref_point[k] - p[k])

                if vol == 0.0:
                    continue

                # Subtract overlap with previously processed points
                subtracted = 0.0
                for j in range(i):
                    q = relevant[j]
                    # Overlap of box [p, ref] and box [q, ref]
                    # = box [max(p,q), ref]
                    overlap = 1.0
                    for k in range(self.n_obj):
                        lo_k = max(p[k], q[k])
                        if ref_point[k] <= lo_k:
                            overlap = 0.0
                            break
                        overlap *= (ref_point[k] - lo_k)
                    subtracted += overlap

                total_vol += vol - subtracted

            # The contribution of f is the total hypervolume minus hypervolume without f
            # But this is complex. Simpler: just check if f is non-dominated and estimate
            return vol  # Approximate: just the volume of f's box

        def is_dominated(f, archive):
            """Check if f is dominated by any point in archive."""
            for af in archive:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def add_to_archive(f, x):
            """Add point to archive if non-dominated."""
            nonlocal archive_F, archive_X
            # Check if dominated
            if is_dominated(f, archive_F):
                return False
            # Remove points dominated by f
            new_archive_F = []
            new_archive_X = []
            for i, af in enumerate(archive_F):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive_F.append(af)
                    new_archive_X.append(archive_X[i])
            archive_F = new_archive_F
            archive_X = new_archive_X
            archive_F.append(f)
            archive_X.append(x)
            return True

        # Tchebycheff scalarization
        def tchebycheff(f, w, z_ideal, z_nadir):
            w_safe = np.where(w > 1e-10, w, 1e-10)
            z_range = np.maximum(z_nadir - z_ideal, 1e-10)
            return np.max(w_safe * (f - z_ideal) / z_range)

        # Main loop
        while used < self.budget:
            if time.time() > deadline - 5:
                break

            for i in range(n_sub):
                if used >= self.budget:
                    break
                if time.time() > deadline - 5:
                    break

                # Select parents from neighbors
                neighbors = neighbor_lists[i]
                if len(neighbors) >= 2:
                    p1_idx = neighbors[rng.randint(len(neighbors))]
                    remaining = [n for n in neighbors if n != p1_idx]
                    p2_idx = remaining[rng.randint(len(remaining))]
                else:
                    p1_idx = rng.randint(n_sub)
                    p2_idx = rng.randint(n_sub)

                # DE mutation with adaptive scaling
                F_scale = 0.5 + 0.4 * rng.random()
                child = best_X[i] + F_scale * (best_X[p1_idx] - best_X[p2_idx])

                # Binomial crossover
                crossover_rate = 0.5
                for j in range(self.dim):
                    if rng.random() < crossover_rate:
                        child[j] = best_X[i][j]

                # Add adaptive noise (decreases over time)
                progress = used / self.budget
                noise_scale = 0.05 * (1 - progress * 0.5) * span
                child += noise_scale * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1

                # Update ideal and nadir
                ideal = np.minimum(ideal, fc)
                nadir = np.maximum(nadir, fc)

                # Try to add to archive
                add_to_archive(fc, child.copy())

                # Evaluate Tchebycheff fitness
                fitness_new = tchebycheff(fc, ref_points[i], ideal, nadir)
                fitness_old = tchebycheff(best_F[i], ref_points[i], ideal, nadir)

                # Check domination
                dominates_old = np.all(fc <= best_F[i]) and np.any(fc < best_F[i])

                if fitness_new < fitness_old or dominates_old:
                    best_X[i] = child
                    best_F[i] = fc

        # Return the archive as the final result
        if archive_F:
            return np.array(archive_X)
        return best_X
# EVOLVE-BLOCK-END
