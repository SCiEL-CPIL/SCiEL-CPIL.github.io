"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """MOEA/D with Tchebycheff decomposition and DE-based local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # Generate reference points on the (n_obj-1)-simplex
        n_ref = 25
        ref_points = self._generate_reference_points(n_ref, n_obj)
        M = len(ref_points)

        # Initialize population
        rng = np.random.default_rng(42)
        X = np.zeros((M, dim))
        for i in range(M):
            X[i] = lo + (hi - lo) * rng.random(dim)

        F = np.array([problem(x) for x in X])
        evals = M

        # Track best solution per subproblem
        best_X = X.copy()
        best_F = F.copy()
        best_z = np.min(F, axis=0)  # ideal point estimate

        # Neighbor list: for each subproblem, find M-1 nearest neighbors
        neighbor_list = self._build_neighbor_list(ref_points, M)

        # DE parameters
        cr = 0.5
        F_de = 0.8
        max_age = 100

        age = np.zeros(M, dtype=int)
        used = evals

        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Select a subproblem to evolve (random)
            idx = rng.integers(M)

            # Find neighbors
            neighbors = neighbor_list[idx]

            # DE-based variation
            if len(neighbors) >= 2:
                n1, n2 = rng.choice(neighbors, size=2, replace=False)
                y = best_X[idx] + F_de * (best_X[n1] - best_X[n2]) + 0.1 * (hi - lo) * rng.standard_normal(dim)
            elif len(neighbors) == 1:
                n1 = neighbors[0]
                y = best_X[idx] + 0.5 * (best_X[n1] - best_X[idx]) + 0.1 * (hi - lo) * rng.standard_normal(dim)
            else:
                y = best_X[idx] + 0.1 * (hi - lo) * rng.standard_normal(dim)

            y = np.clip(y, lo, hi)

            # Evaluate
            fy = problem(y)
            used += 1

            # Update best_z
            best_z = np.minimum(best_z, fy)

            # Tchebycheff fitness
            w = ref_points[idx]
            w_safe = np.where(w > 1e-9, w, 1e-9)
            best_norm = best_z - np.min(best_F, axis=0)
            # Use a large constant for normalization to avoid division issues
            norm = np.max(best_norm) if np.max(best_norm) > 1e-9 else 1.0
            norm = max(norm, 1e-9)

            # Compute Tchebycheff value for current best
            best_t = np.max(w_safe * np.abs(best_F[idx] - best_z) / norm)

            # Compute Tchebycheff value for new solution
            new_t = np.max(w_safe * np.abs(fy - best_z) / norm)

            if new_t < best_t:
                best_X[idx] = y
                best_F[idx] = fy
                age[idx] = 0
            else:
                age[idx] += 1

            # If aged out, try a random restart
            if age[idx] > max_age:
                y_rand = lo + (hi - lo) * rng.random(dim)
                if used < self.budget:
                    fy_rand = problem(y_rand)
                    used += 1
                    best_z = np.minimum(best_z, fy_rand)
                    norm = np.max(best_z - np.min(best_F, axis=0))
                    norm = max(norm, 1e-9)
                    new_t_rand = np.max(w_safe * np.abs(fy_rand - best_z) / norm)
                    if new_t_rand < best_t:
                        best_X[idx] = y_rand
                        best_F[idx] = fy_rand
                        age[idx] = 0

        return best_X

    def _generate_reference_points(self, n, n_obj):
        """Generate reference points on the (n_obj-1)-simplex."""
        if n_obj == 3:
            points = []
            # Use a grid-based approach for 3 objectives
            # Sum of weights = 1, all weights >= 0
            # Use a triangular grid
            n_steps = 5
            for i in range(n_steps + 1):
                for j in range(n_steps + 1 - i):
                    k = n_steps - i - j
                    w = np.array([i / n_steps, j / n_steps, k / n_steps])
                    points.append(w)
            # If we have more than n, subsample; if fewer, add random
            if len(points) > n:
                idx = np.random.choice(len(points), size=n, replace=False)
                points = [points[i] for i in idx]
            elif len(points) < n:
                while len(points) < n:
                    w = np.random.dirichlet(np.ones(n_obj))
                    points.append(w)
            return np.array(points[:n])
        else:
            # General case: random Dirichlet
            points = [np.random.dirichlet(np.ones(n_obj)) for _ in range(n)]
            return np.array(points)

    def _build_neighbor_list(self, ref_points, M):
        """Build neighbor list for each reference point based on distance."""
        neighbor_list = []
        for i in range(M):
            dists = np.linalg.norm(ref_points - ref_points[i], axis=1)
            dists[i] = np.inf  # Exclude self
            neighbors = np.argsort(dists)[:M - 1]
            neighbor_list.append(neighbors)
        return neighbor_list
# EVOLVE-BLOCK-END
