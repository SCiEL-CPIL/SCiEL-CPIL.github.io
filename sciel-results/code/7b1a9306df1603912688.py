# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Elitist non-dominated archive (NSGA-II style) with bounded DE mutation.

    Maintains the full Pareto front (not a scalarized subset) so that
    hypervolume is directly optimized, and uses bounded DE mutation with
    adaptive step size to explore the front efficiently.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        # Hard wall-clock deadline with a safety margin (evaluator kills at 330s).
        deadline = start + 250.0

        dim = self.dim
        n_obj = self.n_obj

        lo = np.empty(dim)
        hi = np.empty(dim)
        for i, (l, h) in enumerate(bounds):
            lo[i] = l
            hi[i] = h
        span = hi - lo

        rng = np.random.default_rng(0)

        # Population size: keep it modest so each generation is cheap and we
        # can fit many generations within the budget.
        pop = max(8, min(40, self.budget // 4))

        # --- Initial random population ---
        X = lo + span * rng.random((pop, dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop

        # --- Elitist non-dominated archive (Pareto front) ---
        # archive_F: (m, n_obj) objective vectors of the current Pareto front.
        archive_F = F.copy()

        def non_dominated_idx(A):
            """Return boolean mask of rows of A that are non-dominated."""
            n = A.shape[0]
            keep = np.ones(n, dtype=bool)
            # A row i dominates row j iff A_i <= A_j elementwise and A_i < A_j somewhere.
            # Vectorized dominance test: for each pair (i, j).
            # le[i, j] = all(A_i <= A_j); lt[i, j] = any(A_i < A_j)
            le = np.all(A[:, None, :] <= A[None, :, :], axis=2)  # (n, n)
            lt = np.any(A[:, None, :] < A[None, :, :], axis=2)   # (n, n)
            dom = le & lt  # dom[i, j] = i dominates j
            dominated = np.any(dom, axis=0)  # j is dominated by some i
            return ~dominated

        def update_archive(Fnew, Xnew):
            """Add new points to the archive, dropping any dominated ones.

            Fnew: (k, n_obj); Xnew: (k, dim). Returns nothing; mutates archive.
            """
            nonlocal archive_F
            # Combine and keep non-dominated.
            combined = np.vstack([archive_F, Fnew])
            mask = non_dominated_idx(combined)
            archive_F = combined[mask]

        def nearest_neighbor_dist(f):
            """Distance from f to the nearest archive point (for crowding)."""
            if archive_F.shape[0] == 0:
                return 0.0
            d = np.linalg.norm(archive_F - f, axis=1)
            return float(np.min(d))

        # --- Main evolution loop ---
        while used < self.budget:
            # Safety margin: stop if we are getting close to the deadline.
            if time.time() > deadline - 5.0:
                break

            # Select parents: prefer non-dominated (archive) and crowded (isolated) points.
            # Build a candidate pool from the current archive (Pareto front).
            m = archive_F.shape[0]
            if m == 0:
                # Fallback: use the whole population.
                pool_F = F
                pool_X = X
                pm = pop
            else:
                pool_F = archive_F
                pool_X = None  # archive only stores F; reconstruct X from population by matching
                pm = m

            # We need X for DE mutation. Use the population X matched to F where possible.
            # Simplest robust approach: keep a parallel structure of (X, F) for the archive.
            # Rebuild archive_X by matching archive_F rows to population rows.
            # For efficiency, instead we maintain archive over the population indices.
            # We'll just draw parents from the full population but bias toward archive membership.
            # Determine which population rows are in the archive (non-dominated).
            nd = non_dominated_idx(F)
            # Crowding distance within the population (non-dominated preferred).
            if np.any(nd):
                # Weight: archive members get priority; among them prefer isolated (far from others).
                # Compute pairwise distance to nearest other population point.
                nd_idx = np.where(nd)[0]
                # For each nd point, distance to nearest other nd point.
                nnd = len(nd_idx)
                if nnd > 1:
                    subF = F[nd_idx]
                    # nearest neighbor distance
                    nn = np.empty(nnd)
                    for a in range(nnd):
                        d = np.linalg.norm(subF - subF[a], axis=1)
                        d[a] = np.inf
                        nn[a] = np.min(d)
                else:
                    nn = np.zeros(nnd)
                # Probability proportional to crowding (larger nn -> more likely).
                w = nn + 1e-9
                w = w / w.sum()
                p1 = nd_idx[rng.choice(nnd, p=w)]
                p2 = nd_idx[rng.choice(nnd, p=w)]
                p3 = nd_idx[rng.choice(nnd, p=w)]
            else:
                p1 = int(rng.integers(pop))
                p2 = int(rng.integers(pop))
                p3 = int(rng.integers(pop))
                # Ensure distinctness loosely.
                if p1 == p2:
                    p2 = (p2 + 1) % pop
                if p1 == p3:
                    p3 = (p3 + 1) % pop

            # --- Bounded DE mutation with adaptive step size ---
            # F_factor adapts: start larger, shrink as budget is consumed.
            progress = used / max(1, self.budget)
            f_factor = 0.9 * (1.0 - 0.5 * progress)  # 0.9 -> 0.45
            # Occasionally use a larger jump for exploration.
            if rng.random() < 0.15:
                f_factor = rng.uniform(0.9, 1.4)

            child = X[p1] + f_factor * (X[p2] - X[p3])
            # Occasionally add a small Gaussian perturbation.
            if rng.random() < 0.5:
                child = child + 0.05 * span * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # --- Elitist replacement: add to archive, drop dominated ---
            update_archive(fc.reshape(1, -1), child.reshape(1, -1))

            # --- Update population: keep it diverse; replace the most "redundant" point.
            # Replace a random non-archive (dominated) point, or the point closest to the child.
            # Find the population point closest to the child in objective space.
            d = np.linalg.norm(F - fc, axis=1)
            worst = int(np.argmin(d))  # most redundant (closest to new child)
            # Only replace if the child is not dominated by that point (elitism).
            # If child is better than F[worst], replace; otherwise keep archive-driven.
            if np.all(fc <= F[worst]) and np.any(fc < F[worst]):
                X[worst] = child
                F[worst] = fc
            else:
                # Replace a dominated (non-archive) point to keep population diverse.
                dom_mask = ~nd
                if np.any(dom_mask):
                    cand = np.where(dom_mask)[0]
                    idx = int(cand[rng.integers(len(cand))])
                    X[idx] = child
                    F[idx] = fc

        # Return the archive (return value is ignored by evaluator).
        return archive_F
# EVOLVE-BLOCK-END
