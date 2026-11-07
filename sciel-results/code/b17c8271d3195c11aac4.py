"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref_point = np.array([3.0, 5.0, 7.0])

        # Initialize population
        pop_size = min(50, self.budget)
        X = np.random.uniform(lo, hi, size=(pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points
        all_X = X.copy()
        all_F = F.copy()

        def fast_non_dominated_sort(F):
            """Fast non-dominated sort for 3 objectives."""
            n = len(F)
            if n == 0:
                return []
            
            S = [[] for _ in range(n)]  # Dominated sets
            n_dom = [0] * n  # Number of dominators
            fronts = [[]]
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
                
                if n_dom[i] == 0:
                    fronts[0].append(i)
            
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
            
            return fronts[:-1]  # Remove last empty front if any

        def crowding_distance(F, front_indices):
            """Calculate crowding distance for a front."""
            n = len(front_indices)
            if n <= 2:
                return [np.inf] * n
            
            distances = [0.0] * n
            for m in range(self.n_obj):
                obj_vals = [F[idx, m] for idx in front_indices]
                sorted_idx = np.argsort(obj_vals)
                sorted_front = [front_indices[i] for i in sorted_idx]
                
                # Boundary points get infinite distance
                distances[front_indices.index(sorted_front[0])] = np.inf
                distances[front_indices.index(sorted_front[-1])] = np.inf
                
                obj_range = obj_vals[sorted_idx[-1]] - obj_vals[sorted_idx[0]]
                if obj_range == 0:
                    continue
                
                for i in range(1, n - 1):
                    pos = front_indices.index(sorted_front[i])
                    distances[pos] += (obj_vals[sorted_idx[i + 1]] - obj_vals[sorted_idx[i - 1]]) / obj_range
            
            return distances

        def select_parents(F, pop_size):
            """Binary tournament selection based on rank and crowding distance."""
            fronts = fast_non_dominated_sort(F)
            n = len(F)
            
            # Assign ranks
            ranks = np.zeros(n, dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank
            
            # Calculate crowding distances
            crowd_dist = np.zeros(n)
            for front in fronts:
                if len(front) > 0:
                    cd = crowding_distance(F, front)
                    for i, idx in enumerate(front):
                        crowd_dist[idx] = cd[i]
            
            # Binary tournament
            i1, i2 = np.random.randint(0, n, size=2)
            if ranks[i1] < ranks[i2]:
                return i1
            elif ranks[i2] < ranks[i1]:
                return i2
            else:
                return i1 if crowd_dist[i1] > crowd_dist[i2] else i2

        def crossover(x1, x2):
            """SBX crossover."""
            eta = 20
            if np.random.rand() < 0.5:
                beta = 1 / (2 * (1 - np.random.rand()))
            else:
                beta = 1 / (2 * np.random.rand())
            beta_c = beta ** (1 / (eta + 1))
            
            c1 = 0.5 * ((1 + beta_c) * x1 + (1 - beta_c) * x2)
            c2 = 0.5 * ((1 - beta_c) * x1 + (1 + beta_c) * x2)
            
            # Clip to bounds
            c1 = np.clip(c1, lo, hi)
            c2 = np.clip(c2, lo, hi)
            
            return c1, c2

        def mutation(x):
            """Polynomial mutation."""
            eta = 20
            x_mut = x.copy()
            for i in range(self.dim):
                if np.random.rand() < 1.0 / self.dim:
                    if np.random.rand() < 0.5:
                        delta = (x_mut[i] - lo[i]) / (hi[i] - lo[i])
                        delta_q = 1 / (1 + 2 * (1 - delta))
                        r = np.random.rand()
                        xq = (2 * r + (1 - 2 * r) * delta_q ** (eta + 1)) ** (1 / (eta + 1))
                        delta_new = xq - 1
                    else:
                        delta = (hi[i] - x_mut[i]) / (hi[i] - lo[i])
                        delta_q = 1 / (1 + 2 * delta)
                        r = np.random.rand()
                        xq = (2 * (1 - r) + 2 * (r - 1) * delta_q ** (eta + 1)) ** (1 / (eta + 1))
                        delta_new = 1 - xq
                    
                    x_mut[i] = x_mut[i] + delta_new * (hi[i] - lo[i])
                    x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
            
            return x_mut

        # Main optimization loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Generate offspring
            offspring_X = []
            offspring_F = []
            
            while len(offspring_X) < pop_size and used + len(offspring_X) < self.budget:
                # Select parents
                p1 = select_parents(F, pop_size)
                p2 = select_parents(F, pop_size)
                
                # Crossover
                c1, c2 = crossover(X[p1], X[p2])
                
                # Mutation
                c1 = mutation(c1)
                c2 = mutation(c2)
                
                # Evaluate
                f1 = problem(c1)
                f2 = problem(c2)
                
                offspring_X.extend([c1, c2])
                offspring_F.extend([f1, f2])
                used += 2
                
                # Update all records
                all_X = np.vstack([all_X, c1, c2])
                all_F = np.vstack([all_F, f1, f2])
            
            # Combine parents and offspring
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, np.array(offspring_F)])
            
            # Select next generation
            fronts = fast_non_dominated_sort(combined_F)
            new_X = []
            new_F = []
            
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - len(new_X)
                    cd = crowding_distance(combined_F, front)
                    sorted_front = np.argsort(-cd)  # Sort by descending crowding distance
                    for i in range(remaining):
                        idx = front[sorted_front[i]]
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)

        return all_X
# EVOLVE-BLOCK-END
