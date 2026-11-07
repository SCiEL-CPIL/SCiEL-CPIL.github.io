"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Two-phase multi-objective optimizer: LHS exploration + archive-based DE refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 290.0  # Safety margin from 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(0)

        # Phase 1: LHS for broad coverage
        lhs_size = min(100, self.budget // 2)
        sampler = qmc.LatinHypercube(d=self.dim, seed=0)
        lhs_samples = sampler.random(n=lhs_size)
        X_lhs = lo + (hi - lo) * lhs_samples
        F_lhs = np.array([problem(x) for x in X_lhs])
        used = lhs_size

        # Build initial archive from non-dominated points
        archive_X, archive_F = self._build_archive(X_lhs, F_lhs)

        # Phase 2: Archive-based DE refinement
        pop_size = min(30, max(10, (self.budget - used) // 5))
        # Initialize population from archive + random
        if len(archive_X) < pop_size:
            n_rand = pop_size - len(archive_X)
            X_rand = lo + (hi - lo) * rng.random((n_rand, self.dim))
            F_rand = np.array([problem(x) for x in X_rand])
            used += n_rand
            X_init = np.vstack([archive_X, X_rand])
            F_init = np.vstack([archive_F, F_rand])
        else:
            X_init = archive_X[:pop_size]
            F_init = archive_F[:pop_size]

        # DE parameters
        CR = 0.9
        F_scale = 0.8

        while used < self.budget:
            if time.time() - start_time > deadline:
                break

            # Select parents from archive using binary tournament
            if len(archive_X) < 2:
                # Not enough in archive, pick from population
                i1, i2 = rng.integers(0, len(X_init), size=2)
                if np.sum(F_init[i1]) <= np.sum(F_init[i2]):
                    parent1 = X_init[i1]
                else:
                    parent1 = X_init[i2]
                i1, i2 = rng.integers(0, len(X_init), size=2)
                if np.sum(F_init[i1]) <= np.sum(F_init[i2]):
                    parent2 = X_init[i1]
                else:
                    parent2 = X_init[i2]
                i1, i2 = rng.integers(0, len(X_init), size=2)
                if np.sum(F_init[i1]) <= np.sum(F_init[i2]):
                    parent3 = X_init[i1]
                else:
                    parent3 = X_init[i2]
            else:
                parent1 = self._tournament_select(archive_X, archive_F, rng)
                parent2 = self._tournament_select(archive_X, archive_F, rng)
                parent3 = self._tournament_select(archive_X, archive_F, rng)

            # DE/BI (binary) crossover
            v = parent1 + F_scale * (parent2 - parent3)
            j_rand = rng.integers(0, self.dim)
            mask = rng.random(self.dim) < CR
            mask[j_rand] = True
            child = np.where(mask, v, parent1)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Update archive
            dominated_idx = np.array([
                np.any(np.all(archive_F <= fc, axis=1) & np.any(archive_F < fc, axis=1))
                for _ in range(len(archive_F))
            ]) if len(archive_F) > 0 else np.array([], dtype=bool)
            
            # Check if child is dominated by any archive member
            child_dominated = np.any(np.all(archive_F <= fc, axis=1) & np.any(archive_F < fc, axis=1)) if len(archive_F) > 0 else False

            if not child_dominated:
                # Remove dominated archive members
                if len(archive_F) > 0:
                    dominated = np.array([
                        np.all(fc <= f, axis=0) and np.any(fc < f)
                        for f in archive_F
                    ])
                    archive_X = np.delete(archive_X, np.where(dominated)[0], axis=0)
                    archive_F = np.delete(archive_F, np.where(dominated)[0], axis=0)
                
                archive_X = np.vstack([archive_X, child.reshape(1, -1)])
                archive_F = np.vstack([archive_F, fc.reshape(1, -1)])

                # Limit archive size
                if len(archive_X) > 200:
                    archive_X, archive_F = self._reduce_archive(archive_X, archive_F)

            # Update population
            # Replace worst individual in population
            if len(X_init) > 0:
                worst = int(np.argmax(np.sum(F_init, axis=1)))
                X_init[worst] = child
                F_init[worst] = fc

        return X_init

    def _build_archive(self, X, F):
        """Build initial archive from non-dominated points."""
        n = len(X)
        if n == 0:
            return np.array([]).reshape(0, self.dim), np.array([]).reshape(0, self.n_obj)
        
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        
        mask = ~dominated
        return X[mask], F[mask]

    def _tournament_select(self, X, F, rng, k=2):
        """Binary tournament selection based on domination and crowding."""
        idx = rng.integers(0, len(X), size=k)
        # Simple: pick the one with lower sum of objectives
        sums = np.sum(F[idx], axis=1)
        return X[idx[np.argmin(sums)]]

    def _reduce_archive(self, X, F, max_size=200):
        """Reduce archive by removing least crowded points."""
        if len(X) <= max_size:
            return X, F
        
        # Compute crowding distance
        n = len(F)
        crowding = np.zeros(n)
        for m in range(self.n_obj):
            order = np.argsort(F[:, m])
            crowding[order[0]] = np.inf
            crowding[order[-1]] = np.inf
            for i in range(1, n - 1):
                if F[order[i+1], m] - F[order[i-1], m] > 1e-10:
                    crowding[order[i]] += (F[order[i+1], m] - F[order[i-1], m])
        
        # Remove points with lowest crowding distance
        keep = np.argsort(crowding)[-max_size:]
        keep = np.sort(keep)
        return X[keep], F[keep]


# EVOLVE-BLOCK-END
