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
        nd_F = F[nd_indices]
        n_nd = len(nd_indices)
        
        # Simple heuristic: contribution ~ distance to nearest neighbor in obj space
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
            # Use a while loop to consume budget as long as we can improve or just explore
            # We will do a limited number of iterations per point to ensure coverage
            max_iters_per_point = 10
            iters = 0
            while budget_remaining > 0 and iters < max_iters_per_point and time.time() < deadline:
                iters += 1
                # Randomly pick a dimension and a step size
                d = np.random.randint(dim)
                step_frac = np.random.choice([0.05, 0.02, 0.01, 0.005])
                direction = np.random.choice([-1, 1])
                
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
                else:
                    # If no improvement, try a smaller step or different direction next time
                    # For now, just continue
                    pass

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

        # Increase population size for better coverage of the 3D objective space
        pop_size = 40
        # Reserve budget for local refinement to push the front outward
        reserve = 30
        effective_budget = budget - reserve

        # Initialize population using global numpy random (respects evaluator seed)
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        gen_count = 0

        while used < effective_budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(F)
            front_ranks = np.zeros(len(F), dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    front_ranks[idx] = rank

            # Compute crowding distance once for selection
            cd = self._compute_cd_for_fronts(F, fronts)

            gen_count += 1
            gen_fraction = used / budget

            # Higher base mutation scale (0.2) for broader exploration;
            # occasional large jumps (20% chance) to escape local basins
            base_mut_scale = 0.2
            jump_prob = 0.20
            jump_scale = 0.4
            mut_rate = 1.0 / dim * (1.5 + 0.5 * gen_fraction)

            # Periodic random injection every 15 generations to maintain diversity
            inject_random = (gen_count % 15 == 0) and (used + pop_size <= budget)

            # Generate offspring
            n_offspring = pop_size
            children_X = np.zeros((n_offspring, dim))
            children_F = np.zeros((n_offspring, n_obj))

            for k in range(n_offspring):
                if time.time() > deadline or used + k >= effective_budget:
                    break

                # Random injection: replace one slot with a fresh random point
                if inject_random and k == 0:
                    child = lo + (hi - lo) * np.random.random(dim)
                else:
                    # Select parents via tournament
                    parents = self._tournament_select(F, front_ranks, cd, 2)
                    p1, p2 = X[parents[0]], X[parents[1]]

                    # SBX crossover
                    if np.random.random() < 0.9:
                        eta_c = 15.0
                        child = np.zeros(dim)
                        for d in range(dim):
                            if np.random.random() < 0.5:
                                u = np.random.random()
                                if u <= 0.5:
                                    beta = (2 * u) ** (1.0 / (eta_c + 1))
                                else:
                                    beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1))
                                child[d] = 0.5 * ((1 + beta) * p1[d] + (1 - beta) * p2[d])
                            else:
                                u = np.random.random()
                                if u <= 0.5:
                                    beta = (2 * u) ** (1.0 / (eta_c + 1))
                                else:
                                    beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1))
                                child[d] = 0.5 * ((1 + beta) * p2[d] + (1 - beta) * p1[d])
                    else:
                        child = p1.copy()

                    # Polynomial mutation with adaptive rate and scale
                    eta_m = 20.0
                    for d in range(dim):
                        if np.random.random() < mut_rate:
                            u = np.random.random()
                            range_d = hi[d] - lo[d]
                            if u < 0.5:
                                delta1 = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                            else:
                                delta1 = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                            # Apply base scale; occasional large jump
                            scale = jump_scale if np.random.random() < jump_prob else base_mut_scale
                            child[d] += delta1 * scale * range_d

                child = np.clip(child, lo, hi)
                children_X[k] = child
                children_F[k] = problem(child)
                used += 1

            # Combine parent and offspring
            X_all = np.vstack([X, children_X[:used - pop_size] if used > pop_size else children_X])
            F_all = np.vstack([F, children_F[:used - pop_size] if used > pop_size else children_F])

            # Non-dominated sort on combined
            fronts_all = self._fast_non_dominated_sort(F_all)

            # Select next population
            new_X = np.zeros((pop_size, dim))
            new_F = np.zeros((pop_size, n_obj))
            count = 0

            for front in fronts_all:
                if count + len(front) <= pop_size:
                    for idx in front:
                        new_X[count] = X_all[idx]
                        new_F[count] = F_all[idx]
                        count += 1
                else:
                    remaining = pop_size - count
                    cd_front = self._crowding_distance(F_all, front)
                    sorted_front = np.argsort(-cd_front)
                    for i in range(remaining):
                        idx = front[sorted_front[i]]
                        new_X[count] = X_all[idx]
                        new_F[count] = F_all[idx]
                        count += 1
                    break

            X, F = new_X, new_F

        # Local polish on non-dominated points using the reserved budget
        if used < budget and time.time() < deadline:
            budget_remaining = budget - used
            X, F, _ = self._local_polish(problem, bounds, X, F, budget_remaining, deadline)

        return X
# EVOLVE-BLOCK-END
