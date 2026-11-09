"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid Multi-Objective Optimizer.
    Combines the fast, simple 'replace worst' loop structure of the top scorer 
    with the intelligent selection (Tournament + Crowding) and variation 
    (SBX + Poly Mutation) of NSGA-II.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        # Safety margin: Evaluator kills at 330s. We stop at 310s to be safe.
        deadline = t_start + 310.0 

        rng = np.random.default_rng(0)
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Population size: 
        # Inspiration 1 used 20. Current used 50.
        # 30 allows for ~5-6 generations if we did full NSGA-II, 
        # but here we do 1 child per generation, so 170 generations.
        # Larger pop means better diversity in selection, but slower sort.
        # 30 is a good balance.
        pop_size = 30
        if pop_size > self.budget:
            pop_size = max(10, self.budget // 2)

        # Initialize population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Helper: Fast Non-Dominated Sort
        def fast_non_dominated_sort(F):
            n = len(F)
            if n == 0:
                return []
            
            S = [[] for _ in range(n)]
            n_count = np.zeros(n, dtype=int)
            
            # Compute domination
            # Using broadcasting for small N is fast, but N=31 is small enough for loops too.
            # Let's use the vectorized check for speed.
            # F[i] <= F[j] and F[i] != F[j]
            
            # To avoid O(N^2) memory if N gets large, we stick to loops for clarity and safety, 
            # but N is small (31).
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_count[j] += 1
            
            fronts = []
            front = []
            for i in range(n):
                if n_count[i] == 0:
                    front.append(i)
            
            while front:
                fronts.append(front)
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            next_front.append(j)
                front = next_front
            
            return fronts

        # Helper: Crowding Distance
        def crowding_distance(F, indices):
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)

            dist = np.zeros(n)
            F_sub = F[indices]

            for m in range(self.n_obj):
                sorted_idx = np.argsort(F_sub[:, m])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                f_range = F_sub[sorted_idx[-1], m] - F_sub[sorted_idx[0], m]
                if f_range > 0:
                    for i in range(1, n - 1):
                        dist[sorted_idx[i]] += (F_sub[sorted_idx[i + 1], m] - F_sub[sorted_idx[i - 1], m]) / f_range

            return dist

        # Helper: Select Parents via Tournament
        def select_parents(F, pop_size):
            # We select 2 parents for 1 child. 
            # We need rank and crowding distance for the current population.
            fronts = fast_non_dominated_sort(F)
            
            rank = np.zeros(len(F), dtype=int)
            for front_idx, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = front_idx

            cd = np.zeros(len(F))
            for front in fronts:
                cd_sub = crowding_distance(F, front)
                for k, idx in enumerate(front):
                    cd[idx] = cd_sub[k]

            parents = []
            for _ in range(2):
                i = rng.integers(len(F))
                j = rng.integers(len(F))
                if rank[i] < rank[j]:
                    parents.append(i)
                elif rank[j] < rank[i]:
                    parents.append(j)
                else:
                    if cd[i] > cd[j]:
                        parents.append(i)
                    else:
                        parents.append(j)
            return parents

        # Helper: SBX Crossover
        def sbx_crossover(p1, p2):
            eta = 15.0
            c1 = np.zeros(self.dim)
            c2 = np.zeros(self.dim)
            for k in range(self.dim):
                if rng.random() < 0.5:
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (eta + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                    c1[k] = 0.5 * ((1 + beta) * p1[k] + (1 - beta) * p2[k])
                    c2[k] = 0.5 * ((1 - beta) * p1[k] + (1 + beta) * p2[k])
                else:
                    c1[k] = p2[k]
                    c2[k] = p1[k]
            return c1, c2

        # Helper: Polynomial Mutation
        def polynomial_mutation(x):
            eta = 20.0
            x_mut = x.copy()
            for k in range(self.dim):
                if rng.random() < 1.0 / self.dim:
                    u = rng.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1 / (eta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                    x_mut[k] += delta * span[k]
            return np.clip(x_mut, lo, hi)

        # Main Loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # 1. Select Parents
            p1_idx, p2_idx = select_parents(F, pop_size)
            
            # 2. Generate Offspring
            c1, c2 = sbx_crossover(X[p1_idx], X[p2_idx])
            # We only need 1 child to maintain population size N
            child_X = polynomial_mutation(c1)
            
            # 3. Evaluate
            child_F = problem(child_X)
            used += 1
            
            # 4. Update Population (NSGA-II Truncation)
            # Combine current population and child
            X_combined = np.vstack([X, [child_X]])
            F_combined = np.vstack([F, [child_F]])
            
            # Sort into fronts
            fronts = fast_non_dominated_sort(F_combined)
            
            # Select best pop_size individuals
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    # Take all from this front
                    for idx in front:
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - len(new_X)
                    cd = crowding_distance(F_combined, front)
                    # Sort front indices by crowding distance descending
                    sorted_front = sorted(range(len(front)), key=lambda k: cd[k], reverse=True)
                    for k in range(remaining):
                        idx = front[sorted_front[k]]
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END
