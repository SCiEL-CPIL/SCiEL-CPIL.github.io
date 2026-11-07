"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """MOEA/D-style multi-objective optimizer with decomposition and external archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        N = 30  # number of subproblems
        archive = []  # list of (x, f) for non-dominated solutions

        # Generate weight vectors using simplex lattice for 3 objectives
        weights = []
        for i in range(N + 1):
            for j in range(N - i + 1):
                k = N - i - j
                if i + j + k == N:
                    weights.append(np.array([i / N, j / N, k / N]))
        weights = np.array(weights[:N])  # take first N
        if len(weights) < N:
            # Pad with random weights if needed
            extra = rng.random((N - len(weights), self.n_obj))
            extra /= extra.sum(axis=1, keepdims=True)
            weights = np.vstack([weights, extra])

        # Initialize population
        X = lo + range_ * rng.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        used = N

        # Initialize external archive
        for i in range(N):
            self._update_archive(archive, X[i], F[i])

        # Neighbor list for each subproblem
        # For simplicity, use a fixed neighborhood radius
        neighbor_radius = 5
        neighbors = []
        for i in range(N):
            dists = np.linalg.norm(weights - weights[i], axis=1)
            idx = np.argsort(dists)[1:neighbor_radius + 1]
            neighbors.append(idx)

        # DE-style mutation parameters
        CR = 0.5  # crossover rate
        F_mut = 0.8  # mutation factor

        while used < self.budget:
            if time.time() > deadline:
                break

            # Select a subproblem to evolve
            i = rng.integers(N)
            neigh = neighbors[i]

            # Select parents from neighborhood
            p1_idx, p2_idx = rng.choice(neigh, size=2, replace=False)
            p1 = X[p1_idx]
            p2 = X[p2_idx]

            # DE//rand/1/bin mutation
            child = p1 + F_mut * (p2 - X[i])
            # Crossover
            mask = rng.random(self.dim) < CR
            child = np.where(mask, child, X[i])
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Update external archive
            self._update_archive(archive, child, fc)

            # Check if child is better than current solution for subproblem i
            # using the Tchebycheff scalarization
            def tcheb(f, w):
                return np.max(w * np.abs(f - f.min()))

            # Actually, use normalized Tchebycheff
            f_min = F.min(axis=0)
            f_max = F.max(axis=0)
            f_range = f_max - f_min
            f_range[f_range == 0] = 1

            def tcheb_norm(f, w, f_min, f_range):
                return np.max(w * np.abs((f - f_min) / f_range))

            cur_score = tcheb_norm(F[i], weights[i], f_min, f_range)
            child_score = tcheb_norm(fc, weights[i], f_min, f_range)

            # Update if child is better or non-dominated
            if child_score < cur_score:
                X[i] = child
                F[i] = fc
            else:
                # Check if child dominates current or is non-dominated
                if np.all(fc <= F[i]) and np.any(fc < F[i]):
                    X[i] = child
                    F[i] = fc
                elif not (np.all(F[i] <= fc) and np.any(F[i] < fc)):
                    # Non-dominated, keep with some probability
                    if rng.random() < 0.1:
                        X[i] = child
                        F[i] = fc

        return X

    def _update_archive(self, archive, x, f):
        """Update external archive with non-dominated solutions."""
        ref = np.array([3.0, 5.0, 7.0])
        if not np.all(f < ref):
            return

        # Check if dominated by any in archive
        for ax, af in archive:
            if np.all(af <= f) and np.any(af < f):
                return

        # Remove dominated solutions from archive
        new_archive = []
        for ax, af in archive:
            if not (np.all(f <= af) and np.any(f < af)):
                new_archive.append((ax, af))
        new_archive.append((x, f))
        archive.clear()
        archive.extend(new_archive)
# EVOLVE-BLOCK-END
