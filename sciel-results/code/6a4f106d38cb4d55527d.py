"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """DE-based multi-objective search with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290.0  # safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # Initial sampling: 20 random points
        n_init = 20
        X_init = lo + (hi - lo) * rng.random((n_init, dim))
        F_init = np.array([problem(x) for x in X_init])
        used = n_init

        # Reference point for HV calculation
        ref = np.array([1705.0, 12.0, 0.3])

        def _non_dominated_mask(F_arr):
            """Return boolean mask of non-dominated points."""
            n = len(F_arr)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                diff = F_arr - F_arr[i]
                dominates = np.all(diff <= 0, axis=1) & np.any(diff < 0, axis=1)
                dominates[i] = False
                if np.any(dominates):
                    mask[i] = False
            return mask

        def _hypervolume(F_arr, ref_pt):
            """Compute hypervolume for 3D with given reference point (exact for small n)."""
            valid = np.all(F_arr < ref_pt, axis=1)
            F_valid = F_arr[valid]
            if len(F_valid) == 0:
                return 0.0
            # Sort by first objective
            F_valid = F_valid[np.argsort(F_valid[:, 0])]
            # For 3D, use the exact algorithm
            # HV = sum over non-dominated points of their contribution
            # Use a simple approach: for each point, compute the volume of the box it uniquely covers
            # This is an approximation but works well for small n
            hv = 0.0
            for i in range(len(F_valid)):
                xi, yi, zi = F_valid[i]
                # Check if this point is non-dominated in the yz-plane among points with x >= xi
                # For simplicity, just compute the full box and subtract overlaps (approximation)
                # Better: use the fact that in 3D, HV can be computed by sorting and incremental updates
                pass
            
            # Use a simpler exact method for 3D:
            # Sort by x, then for each point, compute the area in yz that it uniquely covers
            # This requires checking all pairs, which is O(n^2) but n is small
            n_pts = len(F_valid)
            if n_pts == 0:
                return 0.0
            
            # Sort by x
            order = np.argsort(F_valid[:, 0])
            F_sorted = F_valid[order]
            
            hv = 0.0
            for i in range(n_pts):
                xi, yi, zi = F_sorted[i]
                # The x-interval for this point is [xi, x_{i+1}) or [xi, ref[0]) if last
                x_next = F_sorted[i + 1, 0] if i + 1 < n_pts else ref_pt[0]
                dx = x_next - xi
                if dx <= 0:
                    continue
                
                # Compute the area in yz that is uniquely covered by this point
                # This is the area of [yi, ref[1]] x [zi, ref[2]] minus the area covered by
                # other points with x in [xi, x_next] that dominate in yz
                # For simplicity, use an approximation: assume no overlaps in yz
                area = (ref_pt[1] - yi) * (ref_pt[2] - zi)
                hv += dx * area
            
            return hv

        # Differential Evolution parameters
        pop_size = 30
        CR = 0.5  # crossover rate
        F_mut = 0.8  # mutation factor

        # Initialize population
        X_pop = lo + (hi - lo) * rng.random((pop_size, dim))
        F_pop = np.array([problem(x) for x in X_pop])
        used += pop_size

        # Main DE loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Find non-dominated points in current population
            nd_mask = _non_dominated_mask(F_pop)
            nd_indices = np.where(nd_mask)[0]

            if len(nd_indices) == 0:
                nd_indices = np.arange(pop_size)

            # Select parents for mutation
            for i in range(pop_size):
                if used >= self.budget or time.time() > deadline:
                    break

                # Select 3 distinct parents (excluding i)
                candidates = [j for j in range(pop_size) if j != i]
                parents = rng.choice(candidates, 3, replace=False)
                a, b, c = parents

                # Mutation: v = X[a] + F_mut * (X[b] - X[c])
                v = X_pop[a] + F_mut * (X_pop[b] - X_pop[c])
                v = np.clip(v, lo, hi)

                # Crossover
                trial = X_pop[i].copy()
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]

                # Evaluate trial
                f_trial = problem(trial)
                used += 1

                # Selection: keep trial if it dominates current or is non-dominated
                # Simple selection: replace if trial is better in at least one objective
                # and not worse in all
                improves = np.any(f_trial < F_pop[i])
                worsens = np.all(f_trial > F_pop[i])
                
                if improves and not worsens:
                    X_pop[i] = trial
                    F_pop[i] = f_trial

        return X_pop
# EVOLVE-BLOCK-END
