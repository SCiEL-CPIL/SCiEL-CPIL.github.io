"""MOEA/D-style multi-objective optimizer for WFG9."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """MOEA/D-style decomposition-based multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo

        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        # Population size
        pop_size = 90

        # Generate reference points on simplex (for Tchebycheff decomposition)
        # Number of reference points = pop_size
        ref_points = self._generate_reference_points(n_obj, pop_size)

        # Ideal point (lower bound of objectives, initialized to 0)
        ideal = np.zeros(n_obj)

        # Initialize population
        X = lo + span * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])

        # Update ideal point
        ideal = np.min(F, axis=0)

        # Track evaluations used
        used = pop_size

        # Weights for Tchebycheff
        weights = ref_points

        # Neighborhood size
        T = 5

        # Build neighborhood based on reference points
        # For each weight, find T nearest neighbors
        # Compute pairwise distances between weights
        w_diff = weights[:, None, :] - weights[None, :, :]
        w_dist = np.sqrt(np.sum(w_diff ** 2, axis=2))
        # For each i, get T nearest (excluding self)
        neighborhood = np.zeros((pop_size, T), dtype=int)
        for i in range(pop_size):
            d = w_dist[i].copy()
            d[i] = np.inf
            neighborhood[i] = np.argsort(d)[:T]

        # Adaptive mutation rate
        mutation_scale = 0.15

        # Main loop
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Select parent using binary tournament on Tchebycheff indicator
            # For each subproblem, use neighborhood mating
            # Pick a random subproblem index
            idx = np.random.randint(pop_size)

            # Choose parent from neighborhood of idx
            # 80% from neighborhood, 20% from entire population
            if np.random.random() < 0.8:
                neigh = neighborhood[idx]
                # Binary tournament among neighborhood
                p1 = neigh[np.random.randint(T)]
                p2 = neigh[np.random.randint(T)]
                # Compare using Tchebycheff indicator with weight[idx]
                w = weights[idx]
                # Normalize F
                F_max = np.max(F, axis=0)
                F_min = np.min(F, axis=0)
                F_range = F_max - F_min
                F_range[F_range < 1e-10] = 1.0
                F_norm = (F - F_min) / F_range
                tch1 = np.max(w * F_norm[p1])
                tch2 = np.max(w * F_norm[p2])
                p1 = p1 if tch1 <= tch2 else p2
            else:
                p1 = np.random.randint(pop_size)

            # SBX crossover
            child = self._sbx_crossover(X[p1], X[p1], lo, hi, 20.0)

            # Mutation
            if np.random.random() < 0.9:
                # Polynomial-like mutation
                mut_mask = np.random.random(dim) < 0.3
                if np.any(mut_mask):
                    child[mut_mask] += mutation_scale * span[mut_mask] * np.random.standard_normal(np.sum(mut_mask))
                    child = np.clip(child, lo, hi)

            # Evaluate
            fc = problem(child)
            used += 1

            # Update ideal point
            ideal = np.minimum(ideal, fc)

            # Decide whether to replace based on Tchebycheff indicator
            # Compare child with X[idx] using weight[idx]
            w = weights[idx]
            F_max = np.max(F, axis=0)
            F_min = np.min(F, axis=0)
            F_range = F_max - F_min
            F_range[F_range < 1e-10] = 1.0

            # Tchebycheff for child
            child_norm = (fc - F_min) / F_range
            tch_child = np.max(w * child_norm)

            # Tchebycheff for current
            curr_norm = (F[idx] - F_min) / F_range
            tch_curr = np.max(w * curr_norm)

            if tch_child <= tch_curr:
                X[idx] = child
                F[idx] = fc

        return X

    def _generate_reference_points(self, n_obj, n_points):
        """Generate reference points on the simplex using systematic sampling."""
        if n_obj == 2:
            # For 2 objectives, evenly spaced
            points = np.zeros((n_points, n_obj))
            for i in range(n_points):
                points[i, 0] = i / (n_points - 1) if n_points > 1 else 0.5
                points[i, 1] = 1.0 - points[i, 0]
            return points
        elif n_obj == 3:
            # For 3 objectives, use a grid on the simplex
            # Use Delaunay-style or systematic approach
            # Simple approach: generate points and keep those on simplex
            points = []
            # Use a resolution-based approach
            # For n_obj=3, we need points where w1+w2+w3=1, wi>=0
            # Use a triangular grid
            # Number of divisions per edge
            # Total points on grid = (m+1)(m+2)/2 where m is divisions
            # Find m such that total >= n_points
            m = 1
            while (m + 1) * (m + 2) // 2 < n_points:
                m += 1
            for i in range(m + 1):
                for j in range(m + 1 - i):
                    k = m - i - j
                    points.append([i / m, j / m, k / m])
            points = np.array(points)
            # If we have more than n_points, sample uniformly
            if len(points) > n_points:
                idx = np.random.choice(len(points), n_points, replace=False)
                points = points[idx]
            elif len(points) < n_points:
                # Pad with random points on simplex
                extra = n_points - len(points)
                random_pts = np.random.random((extra, n_obj))
                random_pts /= random_pts.sum(axis=1, keepdims=True)
                points = np.vstack([points, random_pts])
            return points
        else:
            # General case: random points on simplex
            points = np.random.random((n_points, n_obj))
            points /= points.sum(axis=1, keepdims=True)
            return points

    def _sbx_crossover(self, x1, x2, lo, hi, eta):
        """Simulated Binary Crossover."""
        child = np.zeros_like(x1)
        for i in range(len(x1)):
            if np.random.random() < 0.5:
                if abs(x1[i] - x2[i]) < 1e-14:
                    child[i] = x1[i]
                else:
                    u = np.random.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                    child[i] = 0.5 * ((x1[i] + x2[i]) - beta * (x1[i] - x2[i]))
                child[i] = np.clip(child[i], lo[i], hi[i])
            else:
                if abs(x1[i] - x2[i]) < 1e-14:
                    child[i] = x1[i]
                else:
                    u = np.random.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                    child[i] = 0.5 * ((x1[i] + x2[i]) + beta * (x1[i] - x2[i]))
                child[i] = np.clip(child[i], lo[i], hi[i])
        return child
# EVOLVE-BLOCK-END
