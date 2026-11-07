"""Multi-objective optimizer on RE37 with evolutionary search and local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective evolutionary search with adaptive mutation and local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Initial population: mix of random and grid points
        n_init = min(40, budget // 5)
        X = np.empty((n_init, dim))
        # First half: uniform random
        n_rand = n_init // 2
        X[:n_rand] = lo + span * rng.random((n_rand, dim))
        # Second half: structured grid-like points
        n_grid = n_init - n_rand
        for i in range(n_grid):
            # Spread points across the hypercube
            X[n_rand + i] = lo + span * ((i + 0.5) / n_grid * 0.8 + 0.1)
            # Add some variation
            X[n_rand + i] += span * (rng.random(dim) - 0.5) * 0.2
        X = np.clip(X, lo, hi)
        F = np.array([problem(x) for x in X])
        used = n_init

        # Archive of non-dominated points
        archive_X = [x.copy() for x in X]
        archive_F = [f.copy() for f in F]

        def dominates(a, b):
            """Returns True if a dominates b (all <= and at least one <)."""
            return np.all(a <= b) and np.any(a < b)

        def is_dominated(f, archive_F_list):
            """Returns True if f is dominated by any point in archive_F_list."""
            for af in archive_F_list:
                if dominates(af, f):
                    return True
            return False

        def update_archive(x, f):
            """Update the archive with a new point if it's non-dominated."""
            # Check if new point is dominated
            if is_dominated(f, archive_F):
                return False
            # Remove points dominated by the new point
            new_archive_X = []
            new_archive_F = []
            for ax, af in zip(archive_X, archive_F):
                if not dominates(f, af):
                    new_archive_X.append(ax)
                    new_archive_F.append(af)
            new_archive_X.append(x)
            new_archive_F.append(f)
            archive_X.clear()
            archive_F.clear()
            archive_X.extend(new_archive_X)
            archive_F.extend(new_archive_F)
            return True

        # Main evolutionary loop
        mutation_scale = 0.15
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Select parents: pick from non-dominated set
            non_dom_mask = np.array([
                not is_dominated(f, archive_F) for f in F
            ])
            if np.any(non_dom_mask):
                parent_indices = np.where(non_dom_mask)[0]
            else:
                parent_indices = np.arange(len(F))

            parent_idx = parent_indices[rng.integers(len(parent_indices))]
            parent = X[parent_idx]

            # Adaptive mutation
            if rng.random() < 0.3:
                # Large mutation (exploration)
                child = parent + mutation_scale * (hi - lo) * rng.standard_normal(dim)
            else:
                # Small mutation (exploitation)
                child = parent + 0.05 * (hi - lo) * rng.standard_normal(dim)

            # Occasionally do a crossover-like operation
            if rng.random() < 0.2 and len(parent_indices) >= 2:
                parent2_idx = parent_indices[rng.integers(len(parent_indices))]
                if parent2_idx != parent_idx:
                    parent2 = X[parent2_idx]
                    alpha = rng.random()
                    child = alpha * parent + (1 - alpha) * parent2
                    child += 0.05 * (hi - lo) * rng.standard_normal(dim)

            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # Update population: replace worst point
            # Use a simple metric: sum of normalized objectives
            F_min = F.min(axis=0)
            F_max = F.max(axis=0)
            F_range = F_max - F_min
            F_range[F_range == 0] = 1
            F_norm = (F - F_min) / F_range
            scores = F_norm.sum(axis=1)
            worst = int(np.argmax(scores))

            X[worst] = child
            F[worst] = fc

            # Update archive
            update_archive(child, fc)

            # Local refinement around archive points periodically
            if used % 50 == 0 and len(archive_X) > 0:
                # Pick a random archive point and do local search around it
                if time.time() > deadline - 10:
                    break
                arch_idx = rng.integers(len(archive_X))
                center = archive_X[arch_idx].copy()
                n_local = 5
                for _ in range(n_local):
                    if used >= budget or time.time() > deadline:
                        break
                    local_child = center + 0.03 * (hi - lo) * rng.standard_normal(dim)
                    local_child = np.clip(local_child, lo, hi)
                    local_fc = problem(local_child)
                    used += 1
                    update_archive(local_child, local_fc)

        # Final local refinement pass around all archive points
        if time.time() < deadline - 5:
            for ax in archive_X[:]:
                if time.time() > deadline:
                    break
                center = ax.copy()
                n_local = 3
                for _ in range(n_local):
                    if used >= budget or time.time() > deadline:
                        break
                    local_child = center + 0.02 * (hi - lo) * rng.standard_normal(dim)
                    local_child = np.clip(local_child, lo, hi)
                    local_fc = problem(local_child)
                    used += 1
                    update_archive(local_child, local_fc)

        return X
# EVOLVE-BLOCK-END
