"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer with deadline guard and hypervolume-based selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 285.0
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Initial population: Latin Hypercube + boundary points
        pop_size = min(24, budget // 4)
        X = np.empty((pop_size, dim))
        
        # LHS for first half
        n_lhs = pop_size // 2
        for d in range(dim):
            perm = rng.permutation(n_lhs)
            X[perm, d] = lo[d] + span[d] * (np.arange(n_lhs) + 0.5) / n_lhs
        
        # Boundary/corner points for second half
        n_boundary = pop_size - n_lhs
        for i in range(n_boundary):
            # Mix of random boundary points and corners
            if i % 3 == 0:
                # Random point on boundary
                face = rng.integers(dim)
                X[n_lhs + i] = lo + span * rng.random(dim)
                X[n_lhs + i, face] = rng.choice([lo[face], hi[face]])
            elif i % 3 == 1:
                # Corner
                X[n_lhs + i] = np.where(rng.random(dim) < 0.5, lo, hi)
            else:
                # Random interior
                X[n_lhs + i] = lo + span * rng.random(dim)

        F = np.array([problem(x) for x in X])
        used = pop_size

        def hypervolume_2d(points, ref):
            """Compute 2D hypervolume contribution."""
            if len(points) == 0:
                return 0.0
            # Sort by first objective
            idx = np.argsort(points[:, 0])
            pts = points[idx]
            # Keep only points better than reference
            mask = np.all(pts < ref, axis=1)
            pts = pts[mask]
            if len(pts) == 0:
                return 0.0
            # Remove dominated points
            keep = np.ones(len(pts), dtype=bool)
            for i in range(len(pts)):
                if not keep[i]:
                    continue
                for j in range(i + 1, len(pts)):
                    if keep[j] and pts[i, 0] <= pts[j, 0] and pts[i, 1] <= pts[j, 1]:
                        keep[j] = False
            pts = pts[keep]
            if len(pts) == 0:
                return 0.0
            # Sort by first obj ascending
            idx = np.argsort(pts[:, 0])
            pts = pts[idx]
            hv = 0.0
            prev_x = ref[0]
            for i in range(len(pts)):
                width = prev_x - pts[i, 0]
                if width <= 0:
                    continue
                height = ref[1] - pts[i, 1]
                if height <= 0:
                    continue
                hv += width * height
                prev_x = pts[i, 0]
            return hv

        while used < budget:
            if time.time() > deadline:
                break

            # Find non-dominated points
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                diff = F[i] - F
                mask = np.all(diff <= 0, axis=1) & np.any(diff < 0, axis=1)
                mask[i] = False
                dominated[mask] = True

            non_dom_idx = np.where(~dominated)[0]
            n_nd = len(non_dom_idx)

            # Select parent: hypervolume-based tournament
            if n_nd > 0:
                if n_nd <= 3:
                    parent_idx = non_dom_idx[rng.integers(n_nd)]
                else:
                    # Tournament of 3 based on hypervolume contribution
                    candidates = non_dom_idx[rng.integers(n_nd, size=min(3, n_nd))]
                    # Compute simple quality metric: sum of objectives
                    sums = np.sum(F[candidates], axis=1)
                    parent_idx = candidates[np.argmin(sums)]
            else:
                parent_idx = rng.integers(n)

            # Adaptive mutation
            progress = used / budget
            base_step = 0.15 * (1.0 - 0.5 * progress)

            if n_nd >= 2:
                nd_points = F[non_dom_idx]
                local_spread = np.std(nd_points, axis=0)
                local_range = np.ptp(nd_points, axis=0)
                local_range[local_range < 1e-10] = 1e-10
                adaptive_factor = np.clip(np.mean(local_spread / local_range), 0.2, 1.0)
            else:
                adaptive_factor = 1.0

            r = rng.random()
            if r < 0.20 and n_nd >= 2:
                # Crossover
                other_idx = non_dom_idx[rng.integers(n_nd)]
                alpha = rng.random()
                child = alpha * X[parent_idx] + (1 - alpha) * X[other_idx]
                child += 0.05 * span * rng.standard_normal(dim)
            elif r < 0.60:
                # Small mutation
                step = base_step * adaptive_factor
                child = X[parent_idx] + step * span * rng.standard_normal(dim)
            elif r < 0.80:
                # Medium mutation
                child = X[parent_idx] + 0.25 * span * rng.standard_normal(dim)
            else:
                # Large exploration mutation
                child = X[parent_idx] + 0.5 * span * rng.standard_normal(dim)

            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # Replace worst point based on hypervolume contribution
            # Compute simple quality score
            worst_score = np.sum(F, axis=1).copy()
            for i in range(n):
                if dominated[i]:
                    worst_score[i] += 1e10
            worst_idx = int(np.argmax(worst_score))
            X[worst_idx] = child
            F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END
