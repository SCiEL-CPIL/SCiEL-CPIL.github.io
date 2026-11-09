# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Deadline-guarded NSGA-II-style search with hypervolume-guided selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        deadline = time.time() + 280.0  # safety margin before 330s wall-clock
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        def dominates(a, b):
            # a dominates b if a <= b in all and a < b in at least one
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_indices(F):
            n = len(F)
            nd = list(range(n))
            for i in range(n):
                if i in nd:
                    for j in nd:
                        if i != j and dominates(F[i], F[j]):
                            nd.remove(i)
                            break
            return nd

        def hv_contrib(F, ref):
            # contribution of each point to hypervolume (3D, exact-ish via inclusion)
            n = len(F)
            valid = [i for i in range(n) if np.all(F[i] < ref)]
            contrib = np.zeros(n)
            if not valid:
                return contrib
            # For 3D, compute contribution of each point as the box it uniquely covers
            # Use a simple O(n^3) exact method for small n
            for i in valid:
                f = F[i]
                # box from f to ref, but subtract parts covered by other valid points
                vol = (ref[0] - f[0]) * (ref[1] - f[1]) * (ref[2] - f[2])
                # subtract overlaps with other points that dominate in at least the relevant dims
                for j in valid:
                    if i == j:
                        continue
                    fj = F[j]
                    if not (fj[0] <= f[0] and fj[1] <= f[1] and fj[2] <= f[2]):
                        continue
                    # overlap box
                    ox = max(f[0], fj[0])
                    oy = max(f[1], fj[1])
                    oz = max(f[2], fj[2])
                    if ox < ref[0] and oy < ref[1] and oz < ref[2]:
                        vol -= (ref[0] - ox) * (ref[1] - oy) * (ref[2] - oz)
                # Inclusion-exclusion for 3D is complex; approximate by just this
                # This is a lower bound; good enough for selection
                contrib[i] = max(0.0, vol)
            return contrib

        # Initialize population
        rng = np.random.default_rng(42)
        pop_size = min(30, budget // 3)
        X = lo + span * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        ref = np.array([3.0, 5.0, 7.0])

        # Main evolutionary loop
        while used < budget:
            if time.time() > deadline:
                break

            # Find non-dominated set
            nd_idx = non_dominated_indices(F)
            nd_set = set(nd_idx)

            # Compute hypervolume contribution for selection
            contrib = hv_contrib(F, ref)
            # Weight: non-dominated get bonus
            weights = contrib.copy()
            for i in nd_set:
                weights[i] += 1.0
            weights = np.maximum(weights, 1e-9)
            weights = weights / weights.sum()

            # Select parents
            p1 = np.random.choice(len(F), p=weights)
            p2 = np.random.choice(len(F), p=weights)
            if p1 == p2:
                p2 = (p2 + 1) % len(F)

            # SBX-like crossover + mutation
            child = (X[p1] + X[p2]) / 2.0
            # Adaptive mutation
            mut_rate = 0.1 + 0.2 * (1.0 - used / max(1, budget))
            child += mut_rate * span * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace worst point (by hypervolume contribution, or dominated)
            # Prefer replacing a dominated point with low contribution
            dominated_mask = np.array([i not in nd_set for i in range(len(F))])
            if np.any(dominated_mask):
                cand = np.where(dominated_mask)[0]
                # pick the one with lowest contribution
                worst = cand[np.argmin(contrib[cand])]
            else:
                worst = int(np.argmin(contrib))
            X[worst] = child
            F[worst] = fc

            # Local refinement: if we have budget and time, do a local search around best point
            if used < budget - 5 and time.time() < deadline - 10:
                # Find the point with best hypervolume contribution
                best_idx = int(np.argmax(contrib))
                for _ in range(3):
                    if used >= budget or time.time() > deadline:
                        break
                    # Small local step
                    step = 0.02 * span
                    local = X[best_idx] + step * rng.standard_normal(dim)
                    local = np.clip(local, lo, hi)
                    fl = problem(local)
                    used += 1
                    if np.all(fl < F[best_idx]):
                        X[best_idx] = local
                        F[best_idx] = fl
                        best_idx = int(np.argmax(hv_contrib(F, ref)))

        # Final: ensure we have a good set; do a final non-dominated cleanup
        # Return the archive of non-dominated points
        nd_idx = non_dominated_indices(F)
        archive = F[nd_idx]
        return archive
# EVOLVE-BLOCK-END
