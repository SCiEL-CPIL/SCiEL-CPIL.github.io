"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # 300s wall-clock limit
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # Population size: use a good fraction of budget
        pop_size = min(50, max(20, self.budget // 4))
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F_arr):
            """Fast non-dominated sort returning list of fronts and dominance counts."""
            n = len(F_arr)
            S = [[] for _ in range(n)]
            n_dom = [0] * n
            fronts = [[]]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        S[i].append(j)
                        n_dom[i] += 1
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        n_dom[j] += 1
                if n_dom[i] == 0:
                    fronts[0].append(i)
                    fronts[0] = sorted(set(fronts[0]))
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                k += 1
                fronts.append(next_front)
            return fronts[:-1] if fronts[-1] else fronts

        def crowding_distance(F_arr, indices):
            """Compute crowding distance for given indices."""
            if len(indices) <= 2:
                return {i: float('inf') for i in indices}
            m = len(indices)
            cd = {i: 0.0 for i in indices}
            for k in range(n_obj):
                sorted_idx = sorted(indices, key=lambda i: F_arr[i][k])
                cd[sorted_idx[0]] = float('inf')
                cd[sorted_idx[-1]] = float('inf')
                f_range = F_arr[sorted_idx[-1]][k] - F_arr[sorted_idx[0]][k]
                if f_range == 0:
                    continue
                for i in range(1, m - 1):
                    cd[sorted_idx[i]] += (F_arr[sorted_idx[i+1]][k] - F_arr[sorted_idx[i-1]][k]) / f_range
            return cd

        def tournament_select(X_arr, F_arr, fronts, cd_cache, n_select):
            """Tournament selection based on rank and crowding distance."""
            selected = []
            for _ in range(n_select):
                i1 = rng.integers(len(X_arr))
                i2 = rng.integers(len(X_arr))
                # Compare ranks
                rank1 = next((r for r, front in enumerate(fronts) if i1 in front), len(fronts))
                rank2 = next((r for r, front in enumerate(fronts) if i2 in front), len(fronts))
                if rank1 < rank2:
                    winner = i1
                elif rank2 < rank1:
                    winner = i2
                else:
                    # Same rank, use crowding distance
                    cd1 = cd_cache.get(i1, 0)
                    cd2 = cd_cache.get(i2, 0)
                    winner = i1 if cd1 >= cd2 else i2
                selected.append(winner)
            return selected

        while used < self.budget:
            # Check deadline with safety margin
            if time.time() > deadline - 5.0:
                break

            # Fast non-dominated sort
            fronts = fast_non_dominated_sort(F)
            
            # Compute crowding distance for the last front (or all if needed)
            cd_cache = {}
            for front in fronts:
                cd_cache.update(crowding_distance(F, front))

            # Create offspring population
            n_offspring = pop_size
            offspring_X = []
            offspring_F = []
            
            for _ in range(n_offspring):
                # Select parents via tournament
                p1_idx = tournament_select(X, F, fronts, cd_cache, 1)[0]
                p2_idx = tournament_select(X, F, fronts, cd_cache, 1)[0]
                
                # Crossover (SBX-like or simple blend)
                if rng.random() < 0.9:
                    alpha = rng.random()
                    child = alpha * X[p1_idx] + (1 - alpha) * X[p2_idx]
                else:
                    child = X[p1_idx].copy()
                
                # Mutation (polynomial-like)
                if rng.random() < 0.9:
                    # Adaptive mutation rate
                    mut_rate = 0.1 * (1 + 0.5 * rng.random())
                    mask = rng.random(dim) < mut_rate
                    child[mask] += mut_rate * (hi - lo)[mask] * rng.standard_normal(np.sum(mask))
                
                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1
                offspring_X.append(child)
                offspring_F.append(fc)

                if used >= self.budget or time.time() > deadline - 5.0:
                    break

            if not offspring_X:
                break

            # Combine parent and offspring
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, np.array(offspring_F)])
            
            # Non-dominated sort on combined
            combined_fronts = fast_non_dominated_sort(combined_F)
            
            # Select next generation
            new_X = []
            new_F = []
            for front in combined_fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to trim this front using crowding distance
                    cd = crowding_distance(combined_F, front)
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    needed = pop_size - len(new_X)
                    for idx in sorted_front[:needed]:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break

            if new_X:
                X = np.array(new_X)
                F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END
