"""Rewritten optimizer for WFG9 multi-objective task."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj

        # Population size: balance exploration and exploitation
        # With 200 budget, use a moderate population
        pop_size = min(30, self.budget // 5)
        if pop_size < 10:
            pop_size = 10

        # Random initialization
        rng = np.random.default_rng(42)
        X = lo + span * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        # Track all evaluated points for hypervolume (we keep the archive of non-dominated)
        # Actually, the evaluator records ALL returned objective vectors from problem() calls
        # So we just need to make good evaluations. The return value is ignored.

        # NSGA-II components
        def fast_non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]  # domination sets
            n_dominates = [0] * n       # number of solutions dominated by i
            fronts = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dominates[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dominates[i] += 1
            # First front
            front = []
            for i in range(n):
                if n_dominates[i] == 0:
                    front.append(i)
            fronts.append(front)
            while front:
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_dominates[j] -= 1
                        if n_dominates[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                front = next_front
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            n = len(indices)
            if n <= 2:
                return [float('inf')] * n
            dist = np.zeros(n)
            for m in range(n_obj):
                vals = F[indices, m]
                min_v, max_v = vals.min(), vals.max()
                if max_v - min_v < 1e-15:
                    dist[:] = float('inf')
                    continue
                # Sort by objective m
                order = np.argsort(vals)
                dist[order[0]] = float('inf')
                dist[order[-1]] = float('inf')
                for k in range(1, n - 1):
                    dist[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / (max_v - min_v)
            return dist

        def binary_tournament(pop_indices, fronts, crowding):
            """Select one parent via binary tournament on (front, crowding)."""
            i, j = rng.choice(len(pop_indices), size=2, replace=False)
            fi, fj = pop_indices[i], pop_indices[j]
            # Front rank: lower is better
            front_i = next(k for k, f in enumerate(fronts) if fi in f)
            front_j = next(k for k, f in enumerate(fronts) if fj in f)
            if front_i != front_j:
                return i if front_i < front_j else j
            # Same front: higher crowding distance is better
            cd_i = crowding[i]
            cd_j = crowding[j]
            return i if cd_i >= cd_j else j

        # Main evolutionary loop
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break

            # Compute fronts and crowding
            fronts = fast_non_dominated_sort(F)
            
            # Compute crowding distance per front
            front_crowding = {}
            for front in fronts:
                cd = crowding_distance(F, front)
                for idx, d in zip(front, cd):
                    front_crowding[idx] = d

            # Build population index list
            pop_indices = list(range(pop_size))
            
            # Create offspring
            n_offspring = min(pop_size, self.budget - used)
            offspring_X = []
            offspring_F = []
            
            for _ in range(n_offspring):
                if time.time() > deadline:
                    break
                if used + len(offspring_F) >= self.budget:
                    break
                    
                # Select parents
                pi = binary_tournament(pop_indices, fronts, front_crowding)
                pj = binary_tournament(pop_indices, fronts, front_crowding)
                parent_i = X[pi]
                parent_j = X[pj]
                
                # SBX crossover
                if rng.random() < 0.9:
                    eta = 20.0
                    child1 = np.zeros(dim)
                    child2 = np.zeros(dim)
                    for d in range(dim):
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta + 1))
                        else:
                            beta = (2 * (1 - u)) ** (1.0 / (eta + 1))
                        child1[d] = 0.5 * ((1 + beta) * parent_i[d] + (1 - beta) * parent_j[d])
                        child2[d] = 0.5 * ((1 - beta) * parent_i[d] + (1 + beta) * parent_j[d])
                else:
                    child1 = parent_i.copy()
                    child2 = parent_j.copy()
                
                # Polynomial mutation
                eta_m = 20.0
                for d in range(dim):
                    if rng.random() < 1.0 / dim:
                        u = rng.random()
                        if u < 0.5:
                            delta = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                        else:
                            delta = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                        child1[d] += delta * (hi[d] - lo[d])
                    if rng.random() < 1.0 / dim:
                        u = rng.random()
                        if u < 0.5:
                            delta = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                        else:
                            delta = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                        child2[d] += delta * (hi[d] - lo[d])
                
                # Clip to bounds
                child1 = np.clip(child1, lo, hi)
                child2 = np.clip(child2, lo, hi)
                
                # Evaluate
                f1 = problem(child1)
                used += 1
                offspring_X.append(child1)
                offspring_F.append(f1)
                
                if used + 1 < self.budget and time.time() < deadline:
                    f2 = problem(child2)
                    used += 1
                    offspring_X.append(child2)
                    offspring_F.append(f2)
                else:
                    break
            
            if not offspring_X:
                break
            
            offspring_F = np.array(offspring_F)
            
            # Combine parent and offspring
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, offspring_F])
            
            # Non-dominated sort
            all_fronts = fast_non_dominated_sort(combined_F)
            
            # Select next generation
            next_X = []
            next_F = []
            for front in all_fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - len(next_X)
                    cd = crowding_distance(combined_F, front)
                    # Sort by crowding distance descending
                    order = np.argsort(-cd)
                    for k in range(remaining):
                        idx = front[order[k]]
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    break
            
            if not next_X:
                break
            
            X = np.array(next_X)
            F = np.array(next_F)
        
        return X
# EVOLVE-BLOCK-END
