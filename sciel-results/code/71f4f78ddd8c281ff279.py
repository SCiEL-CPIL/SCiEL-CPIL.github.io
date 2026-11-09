"""Multi-objective optimizer on WFG4 with adaptive hypervolume-based search."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Hypervolume-guided evolutionary optimizer for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def __call__(self, problem, bounds):
        deadline = time.time() + 280.0
        dim, n_obj = self.dim, self.n_obj
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        rng = np.random

        ref = np.array([3.0, 5.0, 7.0])

        # Population size
        pop_size = 30
        evals = 0

        # ---- LHS initialization ----
        X = np.zeros((pop_size, dim))
        F = np.zeros((pop_size, n_obj))

        for d in range(dim):
            bins = np.linspace(0, 1, pop_size + 1)
            pts = rng.uniform(bins[:-1], bins[1:])
            rng.shuffle(pts)
            X[:, d] = pts

        X = lo + X * span
        for i in range(pop_size):
            F[i] = problem(X[i])
            evals += 1

        # ---- Helper: check dominance ----
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        # ---- Helper: non-dominated sort ----
        def non_dominated_sort(F_mat):
            n = F_mat.shape[0]
            fronts = []
            remaining = list(range(n))
            while remaining:
                front = []
                for i in remaining:
                    is_dom = False
                    for j in remaining:
                        if i != j and dominates(F_mat[j], F_mat[i]):
                            is_dom = True
                            break
                    if not is_dom:
                        front.append(i)
                fronts.append(front)
                remaining = [x for x in remaining if x not in front]
            return fronts

        # ---- Helper: compute simple HV contribution approximation ----
        def compute_hv(points, ref_point):
            """Compute hypervolume for 3D points using a simple method."""
            # Filter points that are better than reference in all objectives
            mask = np.all(points < ref_point, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0
            # Sort by first objective
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            hv = 0.0
            prev_z1 = ref_point[0]
            for i in range(len(pts)):
                z1, z2, z3 = pts[i]
                if z2 < ref_point[1] and z3 < ref_point[2]:
                    width = prev_z1 - z1
                    height = (ref_point[1] - z2) * (ref_point[2] - z3)
                    hv += width * height
                prev_z1 = z1
            return hv

        # ---- Main evolution loop ----
        # Use a combination of:
        # 1. Hypervolume-based selection
        # 2. Adaptive mutation (DE-like + Gaussian)
        # 3. Elitism with non-dominated sorting

        while evals < self.budget:
            if time.time() > deadline:
                break

            # Generate offspring
            for _ in range(pop_size):
                if evals >= self.budget:
                    break
                if time.time() > deadline:
                    break

                # Select parents using tournament on hypervolume contribution
                # Simple approach: pick random parents, use DE-style mutation
                idx1, idx2, idx3 = rng.choice(pop_size, 3, replace=False)

                # DE/rand/1 mutation
                v = X[idx1] + 0.8 * (X[idx2] - X[idx3])
                v = np.clip(v, lo, hi)

                # Add small Gaussian perturbation
                v = v + 0.05 * span * rng.standard_normal(dim)
                v = np.clip(v, lo, hi)

                fv = problem(v)
                evals += 1

                # Update population using non-dominated sorting
                # Combine current pop and new point
                X_comb = np.vstack([X, v.reshape(1, -1)])
                F_comb = np.vstack([F, fv.reshape(1, -1)])

                # Non-dominated sort
                fronts = non_dominated_sort(F_comb)

                # Keep first pop_size from fronts
                new_X = []
                new_F = []
                for front in fronts:
                    for idx in front:
                        if len(new_X) < pop_size:
                            new_X.append(X_comb[idx])
                            new_F.append(F_comb[idx])
                    if len(new_X) >= pop_size:
                        break

                if len(new_X) < pop_size:
                    # Fill remaining with random points from current pop
                    remaining = list(range(pop_size))
                    for idx in remaining:
                        if len(new_X) < pop_size:
                            new_X.append(X[idx])
                            new_F.append(F[idx])

                X = np.array(new_X[:pop_size])
                F = np.array(new_F[:pop_size])

            # Periodic diversity injection
            if evals % (pop_size * 2) < pop_size:
                # Replace worst point with a random LHS point
                # Find point with worst HV contribution (simplest: largest sum of objectives)
                worst_idx = np.argmax(np.sum(F, axis=1))
                if time.time() < deadline and evals < self.budget:
                    new_x = lo + rng.uniform(0, 1, dim) * span
                    new_f = problem(new_x)
                    evals += 1
                    X[worst_idx] = new_x
                    F[worst_idx] = new_f

        return X
# EVOLVE-BLOCK-END
