"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead optimizer for RE21 multi-objective problem."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        ref = np.array([3000.0, 0.06])
        rng = np.random

        used = 0
        all_X = []
        all_F = []

        def evaluate(x):
            nonlocal used
            x_clipped = np.clip(x, lo, hi)
            fc = problem(x_clipped)
            used += 1
            all_X.append(x_clipped.copy())
            all_F.append(np.array(fc, dtype=float))
            return fc

        def nelder_mead(start_x, weights, max_iter=15):
            """Run Nelder-Mead on weighted sum of objectives."""
            nonlocal used
            n = self.dim
            # Build initial simplex
            simplex = [start_x.copy()]
            for i in range(n):
                dx = start_x.copy()
                dx[i] += 0.1 * (hi[i] - lo[i])
                simplex.append(np.clip(dx, lo, hi))

            # Evaluate initial simplex
            fvals = []
            for i in range(n + 1):
                if used >= self.budget or time.time() > deadline:
                    return
                f = evaluate(simplex[i])
                fvals.append(np.dot(weights, np.array(f, dtype=float)))

            # Nelder-Mead iterations
            for it in range(max_iter):
                if used >= self.budget or time.time() > deadline:
                    break

                # Sort simplex by objective value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, lo, hi)
                if used >= self.budget or time.time() > deadline:
                    break
                fr = np.dot(weights, np.array(evaluate(xr), dtype=float))

                if fr < fvals[0]:
                    # Expansion
                    gamma = 2.0
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, lo, hi)
                    if used >= self.budget or time.time() > deadline:
                        break
                    fe = np.dot(weights, np.array(evaluate(xe), dtype=float))
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    rho = 0.5
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, lo, hi)
                    if used >= self.budget or time.time() > deadline:
                        break
                    fc = np.dot(weights, np.array(evaluate(xc), dtype=float))
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lo, hi)
                            if used >= self.budget or time.time() > deadline:
                                return
                            fvals[i] = np.dot(weights, np.array(evaluate(simplex[i]), dtype=float))

        # Generate stratified starting points and weight vectors
        # Use different weight vectors to explore different parts of Pareto front
        n_starts = 12
        # Stratified random starting points in [0,1]^dim
        start_points = []
        for i in range(n_starts):
            # Use Latin hypercube-like stratification
            fracs = []
            for d in range(self.dim):
                # Stratified sampling
                idx = (i * self.dim + d) % n_starts
                frac = (idx + rng.random()) / n_starts
                fracs.append(frac)
            start_points.append(lo + np.array(fracs) * (hi - lo))

        # Weight vectors: uniform on simplex for 2 objectives
        # Plus some extreme weights to find endpoints
        weight_vectors = []
        # Endpoint weights
        weight_vectors.append(np.array([1.0, 0.0]))
        weight_vectors.append(np.array([0.0, 1.0]))
        # Intermediate weights
        for w in [0.1, 0.3, 0.5, 0.7, 0.9]:
            weight_vectors.append(np.array([w, 1.0 - w]))
        # More diverse weights
        for w in [0.2, 0.4, 0.6, 0.8]:
            weight_vectors.append(np.array([w, 1.0 - w]))

        # Shuffle weight vectors
        rng.shuffle(weight_vectors)

        # Phase 1: Quick random initialization (5 points)
        for i in range(5):
            if time.time() > deadline or used >= self.budget:
                break
            x = lo + (hi - lo) * rng.random(self.dim)
            evaluate(x)

        # Phase 2: Multi-start Nelder-Mead
        for i in range(n_starts):
            if time.time() > deadline or used >= self.budget:
                break

            # Choose weight vector (cycle through them)
            w = weight_vectors[i % len(weight_vectors)]

            # Choose starting point
            if i < len(start_points):
                x0 = start_points[i]
            else:
                x0 = lo + (hi - lo) * rng.random(self.dim)

            nelder_mead(x0, w, max_iter=12)

        # Phase 3: Fill remaining budget with random exploration
        while used < self.budget:
            if time.time() > deadline:
                break
            x = lo + (hi - lo) * rng.random(self.dim)
            evaluate(x)

        return all_X
# EVOLVE-BLOCK-END
