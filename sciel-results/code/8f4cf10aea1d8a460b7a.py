# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE34 using Sobol exploration + hypervolume-contribution local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        ref = np.array([1705.0, 12.0, 0.3])

        # --- Phase 1: Sobol quasi-random sequence for uniform coverage ---
        from scipy.stats import qmc

        # Generate Sobol sequence with power-of-2 length >= 150
        sobol_n = 128  # power of 2
        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        X_sobol = sobol.random(sobol_n)
        # Map to bounds
        X_sobol = lo + (hi - lo) * X_sobol

        F_sobol = np.empty((sobol_n, n_obj))
        for i in range(sobol_n):
            if time.time() > deadline - 5:
                break
            F_sobol[i] = problem(X_sobol[i])

        used = sobol_n

        # Build initial archive (non-dominated set)
        def is_dominated(point, archive_F):
            """Check if point is dominated by any point in archive_F."""
            return np.any(np.all(archive_F <= point, axis=1) & np.any(archive_F < point, axis=1))

        def prune_archive(archive_F):
            """Remove dominated points from archive_F. Returns indices to keep."""
            n = len(archive_F)
            keep = np.ones(n, dtype=bool)
            for i in range(n):
                if not keep[i]:
                    continue
                for j in range(n):
                    if i == j or not keep[j]:
                        continue
                    if np.all(archive_F[j] <= archive_F[i]) and np.any(archive_F[j] < archive_F[i]):
                        keep[i] = False
                        break
            return keep

        # Build archive from Sobol points
        archive_X = []
        archive_F = []
        for i in range(sobol_n):
            fc = F_sobol[i]
            # Only add if not dominated by current archive
            if len(archive_F) == 0 or not is_dominated(fc, np.array(archive_F)):
                archive_X.append(X_sobol[i])
                archive_F.append(fc)

        archive_X = np.array(archive_X)
        archive_F = np.array(archive_F)

        # Prune the archive
        if len(archive_F) > 1:
            keep = prune_archive(archive_F)
            archive_X = archive_X[keep]
            archive_F = archive_F[keep]

        # --- Phase 2: Hypervolume-contribution-based local search ---
        # Compute hypervolume contribution of each point in the archive
        def compute_hv_contribution(archive_F, ref):
            """Compute the hypervolume contribution of each point in the archive.
            Returns an array of contributions (one per point)."""
            n = len(archive_F)
            if n == 0:
                return np.array([])
            if n == 1:
                diff = np.maximum(ref - archive_F[0], 0)
                return np.array([np.prod(diff)])

            contributions = np.zeros(n)
            for i in range(n):
                # HV without point i
                remaining = [j for j in range(n) if j != i]
                if not remaining:
                    hv_without = 0.0
                else:
                    hv_without = _compute_hv(np.array(archive_F[remaining]), ref)
                # HV with point i
                hv_with = _compute_hv(archive_F, ref)
                contributions[i] = hv_with - hv_without
            return contributions

        def _compute_hv(points, ref):
            """Compute hypervolume of a set of points with reference point.
            Uses a simple recursive approach for small n."""
            n = len(points)
            if n == 0:
                return 0.0
            if n == 1:
                diff = np.maximum(ref - points[0], 0)
                if np.any(diff <= 0):
                    return 0.0
                return float(np.prod(diff))
            if n == 2:
                # For 2 points in 3D, compute union of two boxes
                p1 = points[0]
                p2 = points[1]
                # Box 1: from p1 to ref
                b1_lo = p1
                b1_hi = ref
                # Box 2: from p2 to ref
                b2_lo = p2
                b2_hi = ref

                def box_volume(lo, hi):
                    d = np.maximum(hi - lo, 0)
                    return float(np.prod(d))

                v1 = box_volume(b1_lo, b1_hi)
                v2 = box_volume(b2_lo, b2_hi)

                # Intersection
                inter_lo = np.maximum(b1_lo, b2_lo)
                inter_hi = np.minimum(b1_hi, b2_hi)
                v_inter = box_volume(inter_lo, inter_hi)

                return v1 + v2 - v_inter

            # For larger n, use a simple grid-based approximation or recursive splitting
            # Use a brute-force approach: for each point, compute the volume of the region
            # where it is the "best" in all objectives
            # This is expensive but n is small (archive size is limited)

            # Alternative: use the inclusion-exclusion or a simple Monte Carlo
            # For efficiency, let's use a different approach:
            # Sort by first objective, then compute contributions

            # Actually, for small archives (n < 50), let's use a simple approach:
            # Compute HV by iterating over all subsets... no, that's too expensive.
            # Let's use a grid-based approximation for the local search phase.
            # Actually, for the local search we just need relative contributions,
            # so we can use a simpler metric: distance to reference point in a normalized sense.

            # For now, use a simple proxy: the volume of the box from point to ref,
            # adjusted for dominance. This is approximate but fast.
            total = 0.0
            for i in range(n):
                diff = np.maximum(ref - points[i], 0)
                if np.all(diff > 0):
                    total += np.prod(diff)
            return total / n  # approximate

        # For the local search, we'll use a simpler contribution metric:
        # The volume of the box from each point to the reference point
        # (points that are closer to the reference in all objectives contribute more)
        def compute_simple_contribution(archive_F, ref):
            """Compute a simple contribution metric for each point."""
            n = len(archive_F)
            if n == 0:
                return np.array([])
            contributions = np.zeros(n)
            for i in range(n):
                diff = np.maximum(ref - archive_F[i], 0)
                contributions[i] = np.prod(diff)
            return contributions

        remaining = budget - used
        if remaining > 0:
            # Local search: perturb points with lowest contribution
            for iteration in range(remaining):
                if time.time() > deadline - 3:
                    break
                if used >= budget:
                    break

                if len(archive_F) == 0:
                    # No archive, just do random search
                    x = np.random.uniform(lo, hi)
                    fc = problem(x)
                    used += 1
                    archive_X = np.array([x])
                    archive_F = np.array([fc])
                    continue

                # Compute contributions
                contribs = compute_simple_contribution(archive_F, ref)

                # Select the point with the lowest contribution (or a random one among the lowest)
                # Use a softmax-like selection to balance exploration and exploitation
                # Simply pick the point with the lowest contribution
                idx = np.argmin(contribs)
                parent = archive_X[idx]

                # Gaussian perturbation with adaptive step size
                # Step size decreases over time
                progress = iteration / max(1, remaining - 1)
                step = 0.20 * (1 - progress) + 0.03 * progress

                # Generate a perturbation
                noise = np.random.standard_normal(dim) * step * (hi - lo)
                child = parent + noise
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1

                # Check if the child is non-dominated
                if not is_dominated(fc, archive_F):
                    # Add to archive
                    archive_X = np.vstack([archive_X, child.reshape(1, -1)])
                    archive_F = np.vstack([archive_F, fc.reshape(1, -1)])

                    # Prune
                    if len(archive_F) > 1:
                        keep = prune_archive(archive_F)
                        archive_X = archive_X[keep]
                        archive_F = archive_F[keep]

                # Occasionally do a larger jump for diversity
                if iteration % 10 == 0 and used < budget:
                    jump_size = 0.3 * (1 - progress) + 0.1 * progress
                    noise_large = np.random.standard_normal(dim) * jump_size * (hi - lo)
                    child2 = parent + noise_large
                    child2 = np.clip(child2, lo, hi)
                    fc2 = problem(child2)
                    used += 1
                    if not is_dominated(fc2, archive_F):
                        archive_X = np.vstack([archive_X, child2.reshape(1, -1)])
                        archive_F = np.vstack([archive_F, fc2.reshape(1, -1)])
                        if len(archive_F) > 1:
                            keep = prune_archive(archive_F)
                            archive_X = archive_X[keep]
                            archive_F = archive_F[keep]

        return archive_X

# EVOLVE-BLOCK-END
