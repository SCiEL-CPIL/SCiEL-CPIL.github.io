"""Multi-objective optimizer for RE37 using NSGA-II with adaptive mutation and deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective optimizer for RE37 (4 vars, 3 objectives)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """Fast non-dominated sorting. Returns list of fronts (each a list of indices)."""
        n = len(F)
        S = [[] for _ in range(n)]
        ndom = np.zeros(n, dtype=int)
        rank = np.full(n, -1, dtype=int)
        fronts = []

        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    ndom[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    ndom[i] += 1

        current_front = [i for i in range(n) if ndom[i] == 0]
        if not current_front:
            return [[i for i in range(n)]]

        front_idx = 0
        while current_front:
            rank[current_front] = front_idx
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in S[i]:
                    ndom[j] -= 1
                    if ndom[j] == 0:
                        next_front.append(j)
            current_front = next_front
            front_idx += 1

        # Safety: if any points weren't assigned (shouldn't happen), add them
        unassigned = [i for i in range(n) if rank[i] < 0]
        if unassigned:
            fronts.append(unassigned)

        return fronts

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for a set of indices."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)

        cd = np.zeros(n)
        for m in range(self.n_obj):
            vals = F[indices, m]
            sorted_idx = np.argsort(vals)
            cd[sorted_idx[0]] = np.inf
            cd[sorted_idx[-1]] = np.inf
            val_range = vals[sorted_idx[-1]] - vals[sorted_idx[0]]
            if val_range > 0:
                for k in range(1, n - 1):
                    cd[sorted_idx[k]] += (vals[sorted_idx[k + 1]] - vals[sorted_idx[k - 1]]) / val_range
        return cd

    def _compute_cd_for_fronts(self, F, fronts):
        """Compute crowding distance for all fronts, returns array of same length as F."""
        n = len(F)
        cd = np.zeros(n)
        for front in fronts:
            if len(front) > 0:
                cd[front] = self._crowding_distance(F, front)
        return cd

    def _tournament_select(self, F, front_ranks, cd, n_select):
        """Binary tournament selection based on front rank and crowding distance."""
        n = len(F)
        selected = []
        for _ in range(n_select):
            i1 = np.random.randint(n)
            i2 = np.random.randint(n)
            if front_ranks[i1] < front_ranks[i2]:
                selected.append(i1)
            elif front_ranks[i2] < front_ranks[i1]:
                selected.append(i2)
            else:
                if cd[i1] > cd[i2]:
                    selected.append(i1)
                else:
                    selected.append(i2)
        return selected

    def _local_polish(self, problem, bounds, X, F, budget_remaining, deadline):
        """Hypervolume-driven local polish: refine non-dominated points toward gaps."""
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        ref = np.array([1.1, 1.3, 1.1])

        # Get non-dominated indices
        fronts = self._fast_non_dominated_sort(F)
        if not fronts:
            return X, F, budget_remaining

        nd_indices = list(fronts[0])
        if not nd_indices:
            return X, F, budget_remaining

        # Compute marginal hypervolume contribution for each ND point
        # A point contributes more HV if it's far from other ND points in objective space
        nd_F = F[nd_indices]
        n_nd = len(nd_indices)
        
        # Simple heuristic: contribution ~ distance to nearest neighbor in obj space
        # Points with larger "gap" to neighbors are more valuable to refine
        contributions = np.zeros(n_nd)
        for i in range(n_nd):
            min_dist = np.inf
            for j in range(n_nd):
                if i == j:
                    continue
                d = np.max(np.abs(nd_F[i] - nd_F[j]))
                min_dist = min(min_dist, d)
            contributions[i] = min_dist

        # Process points in order of contribution (most isolated first)
        order = np.argsort(-contributions)

        for oi in order:
            if budget_remaining <= 0 or time.time() > deadline:
                break

            idx = nd_indices[oi]
            base_x = X[idx].copy()
            base_f = F[idx].copy()
            best_x = base_x.copy()
            best_f = base_f.copy()

            # Multi-scale local search: try multiple step sizes
            step_sizes = [0.05, 0.02, 0.01, 0.005]
            for step_frac in step_sizes:
                if budget_remaining <= 0 or time.time() > deadline:
                    break
                for d in range(dim):
                    if budget_remaining <= 0 or time.time() > deadline:
                        break
                    for direction in [-1, 1]:
                        if budget_remaining <= 0 or time.time() > deadline:
                            break
                        step = step_frac * (hi[d] - lo[d])
                        trial_x = base_x.copy()
                        trial_x[d] += direction * step
                        trial_x = np.clip(trial_x, lo, hi)
                        trial_f = np.array(problem(trial_x))
                        budget_remaining -= 1

                        # Check if trial dominates best
                        if np.all(trial_f <= best_f) and np.any(trial_f < best_f):
                            best_x = trial_x.copy()
                            best_f = trial_f.copy()
                            base_x = trial_x.copy()
                            base_f = trial_f.copy()

            if np.all(best_f < F[idx]):
                X[idx] = best_x
                F[idx] = best_f

        return X, F, budget_remaining

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # 300s limit, 15s safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # DE Parameters
        # Larger population for better diversity, fewer generations
        pop_size = 50 
        F = 0.7  # Scaling factor
        CR = 0.9 # Crossover rate

        # Initialize population
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F_vals = np.array([problem(x) for x in X])
        used = pop_size

        # Main DE Loop
        while used < budget:
            if time.time() > deadline:
                break
            
            for i in range(pop_size):
                if used >= budget or time.time() > deadline:
                    break
                
                # Select 3 distinct random indices
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                
                # DE/rand/1/bin
                v = X[r1] + F * (X[r2] - X[r3])
                
                # Crossover
                j_rand = np.random.randint(dim)
                trial_x = X[i].copy()
                for j in range(dim):
                    if np.random.random() < CR or j == j_rand:
                        trial_x[j] = v[j]
                
                # Clip
                trial_x = np.clip(trial_x, lo, hi)
                
                # Evaluate
                f_trial = np.array(problem(trial_x))
                used += 1
                
                # Selection: If trial dominates current, replace
                if np.all(f_trial <= F_vals[i]) and np.any(f_trial < F_vals[i]):
                    X[i] = trial_x
                    F_vals[i] = f_trial
                # If non-dominated, we keep the current one.

        # Local polish on non-dominated points if budget remains
        if used < budget and time.time() < deadline:
            budget_remaining = budget - used
            X, F_vals, _ = self._local_polish(problem, bounds, X, F_vals, budget_remaining, deadline)

        return X
# EVOLVE-BLOCK-END
