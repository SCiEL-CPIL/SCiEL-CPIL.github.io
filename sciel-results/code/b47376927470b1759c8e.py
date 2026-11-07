"""Multi-objective optimizer for RE37: NSGA-II with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective optimizer with deadline safety."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.pop_size = 16

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        d = self.dim
        pop_size = self.pop_size

        rng = np.random.default_rng(0)

        # Initialization
        X = lo + (hi - lo) * rng.random((pop_size, d))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Main NSGA-II loop
        while used + pop_size <= self.budget:
            if time.time() >= deadline:
                break

            # Calculate ranks and crowding distance
            ranks, cd = self._calculate_rank_cd(F)

            # Create offspring
            Q = np.zeros((pop_size, d))
            FQ = np.zeros((pop_size, self.n_obj))

            for i in range(pop_size):
                p1 = self._tournament(ranks, cd, rng)
                p2 = self._tournament(ranks, cd, rng)

                x1 = X[p1]
                x2 = X[p2]

                # BLX-Alpha crossover
                alpha = 0.5
                low = np.minimum(x1, x2) - alpha * np.abs(x1 - x2)
                high = np.maximum(x1, x2) + alpha * np.abs(x1 - x2)
                child = rng.uniform(low, high, size=d)
                child = np.clip(child, lo, hi)

                # Gaussian mutation
                for k in range(d):
                    if rng.random() < 1.0 / d:
                        child[k] += 0.1 * (hi[k] - lo[k]) * rng.standard_normal()
                        child[k] = np.clip(child[k], lo[k], hi[k])

                Q[i] = child
                FQ[i] = problem(child)

            used += pop_size

            # Combine and select
            R_X = np.vstack([X, Q])
            R_F = np.vstack([F, FQ])

            X, F = self._nsga2_select(R_X, R_F)

        # Final polish: adaptive step-size local search on non-dominated points
        # Reserve remaining budget for high-precision refinement
        remaining = self.budget - used
        if remaining > 0:
            # Get non-dominated front indices
            dominated_mask = np.array([
                np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1)) for f in F
            ])
            front_idx = np.where(~dominated_mask)[0]
            if len(front_idx) == 0:
                front_idx = np.array([np.argmin(np.sum(F, axis=1))])

            # Adaptive step-size hill climbing
            n_polish = min(remaining, 50)
            n_evals_used = 0
            step_scale = 0.05  # initial step size as fraction of range

            for _ in range(n_polish):
                if time.time() >= deadline or n_evals_used >= n_polish:
                    break

                # Pick a point from the front (cycle through)
                pi = front_idx[n_evals_used % len(front_idx)]
                parent = X[pi].copy()

                # Try multiple perturbations with adaptive step
                best_child = None
                best_fc = None
                n_trials = min(4, remaining - n_evals_used)

                for t in range(n_trials):
                    if time.time() >= deadline:
                        break
                    # Gaussian perturbation with adaptive step
                    step = step_scale * (hi - lo)
                    child = parent + step * rng.standard_normal(d)
                    child = np.clip(child, lo, hi)
                    fc = problem(child)
                    n_evals_used += 1

                    # Check if this child improves upon parent or is non-dominated
                    # Simple acceptance: better in at least one obj and no worse in others
                    if np.all(fc <= F[pi]) and np.any(fc < F[pi]):
                        best_child = child
                        best_fc = fc
                        break
                    # Or better than the worst point
                    worst_idx = int(np.argmax(np.sum(F, axis=1)))
                    if np.all(fc <= F[worst_idx]) and np.any(fc < F[worst_idx]):
                        if best_child is None or np.sum(fc) < np.sum(best_fc):
                            best_child = child
                            best_fc = fc

                if best_child is not None:
                    # Replace the worse of parent and worst
                    worst_idx = int(np.argmax(np.sum(F, axis=1)))
                    if np.sum(best_fc) < np.sum(F[worst_idx]):
                        X[worst_idx] = best_child
                        F[worst_idx] = best_fc
                    elif np.sum(best_fc) < np.sum(F[pi]):
                        X[pi] = best_child
                        F[pi] = best_fc

                # Shrink step size adaptively
                step_scale *= 0.85

            # Use any leftover budget with simple mutation
            while used + n_evals_used < self.budget:
                if time.time() >= deadline:
                    break
                pi = int(rng.integers(len(X)))
                child = X[pi] + 0.02 * (hi - lo) * rng.standard_normal(d)
                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1
                worst_idx = int(np.argmax(np.sum(F, axis=1)))
                if np.sum(fc) < np.sum(F[worst_idx]):
                    X[worst_idx] = child
                    F[worst_idx] = fc

        return X

    def _calculate_rank_cd(self, F):
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        cd = np.zeros(n)

        # Non-dominated sorting
        remaining = set(range(n))
        current_rank = 0

        while remaining:
            front_indices = []
            for i in list(remaining):
                dominated = False
                for j in remaining:
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    front_indices.append(i)
                    remaining.remove(i)

            if not front_indices:
                break

            for i in front_indices:
                ranks[i] = current_rank

            if len(front_indices) > 1:
                for m in range(self.n_obj):
                    order = np.argsort(F[front_indices, m])
                    sorted_indices = [front_indices[k] for k in order]

                    cd[sorted_indices[0]] = np.inf
                    cd[sorted_indices[-1]] = np.inf

                    span = F[sorted_indices[-1], m] - F[sorted_indices[0], m]
                    if span > 0:
                        for k in range(1, len(sorted_indices) - 1):
                            idx_curr = sorted_indices[k]
                            idx_prev = sorted_indices[k - 1]
                            idx_next = sorted_indices[k + 1]
                            cd[idx_curr] += (F[idx_next, m] - F[idx_prev, m]) / span
            else:
                for i in front_indices:
                    cd[i] = np.inf

            current_rank += 1

        return ranks, cd

    def _tournament(self, ranks, cd, rng):
        i1 = rng.integers(len(ranks))
        i2 = rng.integers(len(ranks))

        if ranks[i1] < ranks[i2]:
            return i1
        elif ranks[i2] < ranks[i1]:
            return i2
        else:
            if cd[i1] > cd[i2]:
                return i1
            elif cd[i2] > cd[i1]:
                return i2
            else:
                return i1

    def _nsga2_select(self, R_X, R_F):
        ranks, cd = self._calculate_rank_cd(R_F)

        # Sort by rank asc, then cd desc
        sorted_indices = np.lexsort((-cd, ranks))

        selected_idx = sorted_indices[:self.pop_size]

        return R_X[selected_idx], R_F[selected_idx]
# EVOLVE-BLOCK-END
