# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Track evaluations
        eval_count = 0
        all_X = []
        all_F = []
        
        def evaluate(x):
            nonlocal eval_count
            x_clipped = np.clip(np.asarray(x, dtype=float), lo, hi)
            f = np.asarray(problem(x_clipped), dtype=float)
            all_X.append(x_clipped.copy())
            all_F.append(f.copy())
            eval_count += 1
            return f

        # Population size
        # We need to balance exploration and exploitation.
        # With budget 200, N=20 allows ~10 generations.
        N = 20
        
        # Phase 1: Initialization
        X = np.array([lo + (hi - lo) * rng.random(self.dim) for _ in range(N)])
        F = np.array([evaluate(x) for x in X])
        
        # Helper: Fast Non-Dominated Sorting
        def fast_non_dominated_sort(F_arr):
            n = len(F_arr)
            fronts = [[]]
            S = [set() for _ in range(n)]
            n_doms = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    # If i dominates j
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        S[i].add(j)
                    # If j dominates i
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        n_doms[i] += 1
                
                if n_doms[i] == 0:
                    fronts[0].append(i)
            
            k = 0
            while len(fronts[k]) > 0:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_doms[j] -= 1
                        if n_doms[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
                else:
                    break
            return fronts

        # Helper: Crowding Distance
        def crowding_distance(F_arr, indices):
            n = len(indices)
            if n <= 2:
                return [float('inf')] * n
            
            cd = [0.0] * n
            m = self.n_obj
            
            for k in range(m):
                # Sort indices by objective k
                sorted_indices = sorted(indices, key=lambda i: F_arr[i][k])
                # Set boundaries to infinity
                cd[sorted_indices[0]] = float('inf')
                cd[sorted_indices[-1]] = float('inf')
                
                f_min = F_arr[sorted_indices[0]][k]
                f_max = F_arr[sorted_indices[-1]][k]
                range_k = f_max - f_min
                if range_k == 0:
                    continue
                    
                for i in range(1, n - 1):
                    cd[sorted_indices[i]] += (F_arr[sorted_indices[i+1]][k] - F_arr[sorted_indices[i-1]][k]) / range_k
            
            return cd

        # Helper: SBX Crossover
        def sbx_crossover(x1, x2, eta=15.0):
            x_child = np.zeros(self.dim)
            for i in range(self.dim):
                if rng.random() < 0.5:
                    # Crossover on dimension i
                    if abs(x1[i] - x2[i]) < 1e-14:
                        x_child[i] = x1[i]
                    else:
                        u = rng.random()
                        if u <= 0.5:
                            alpha = (2 * u) ** (1 / (eta + 1))
                        else:
                            alpha = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                        
                        c1 = 0.5 * ((x1[i] + x2[i]) - alpha * (x1[i] - x2[i]))
                        c2 = 0.5 * ((x1[i] + x2[i]) + alpha * (x1[i] - x2[i]))
                        
                        # Clip to bounds
                        c1 = np.clip(c1, lo[i], hi[i])
                        c2 = np.clip(c2, lo[i], hi[i])
                        
                        if rng.random() < 0.5:
                            x_child[i] = c1
                        else:
                            x_child[i] = c2
                else:
                    x_child[i] = x1[i]
            return x_child

        # Helper: Polynomial Mutation
        def polynomial_mutation(x, eta=20.0):
            x_mut = x.copy()
            for i in range(self.dim):
                if rng.random() < 1.0 / self.dim:
                    u = rng.random()
                    delta1 = (1 + 2 * (x_mut[i] - lo[i]) / (hi[i] - lo[i])) ** (1 / (eta + 1))
                    delta2 = (1 + 2 * (hi[i] - x_mut[i]) / (hi[i] - lo[i])) ** (1 / (eta + 1))
                    
                    if u < 0.5:
                        deltaq = (2 * u) ** (1 / (eta + 1)) - 1
                    else:
                        deltaq = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                    
                    x_mut[i] += deltaq * (hi[i] - lo[i])
                    x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
            return x_mut

        # Main NSGA-II Loop
        while eval_count + N <= self.budget:
            # Generate Offspring
            X_off = []
            for _ in range(N):
                # Tournament Selection
                idx1 = self.tournament_select(F, N)
                idx2 = self.tournament_select(F, N)
                
                # Crossover
                child = sbx_crossover(X[idx1], X[idx2])
                # Mutation
                child = polynomial_mutation(child)
                X_off.append(child)
            
            # Evaluate Offspring
            F_off = np.array([evaluate(x) for x in X_off])
            
            # Combine Populations
            X_comb = np.vstack((X, X_off))
            F_comb = np.vstack((F, F_off))
            
            # Non-Dominated Sorting
            fronts = fast_non_dominated_sort(F_comb)
            
            # Select Next Generation
            X_new = []
            F_new = []
            count = 0
            for front in fronts:
                if count + len(front) <= N:
                    # Take all
                    for idx in front:
                        X_new.append(X_comb[idx])
                        F_new.append(F_comb[idx])
                    count += len(front)
                else:
                    # Need to select from this front based on crowding distance
                    needed = N - count
                    cd = crowding_distance(F_comb, front)
                    # Sort by crowding distance (descending)
                    sorted_front = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    for idx in sorted_front[:needed]:
                        X_new.append(X_comb[idx])
                        F_new.append(F_comb[idx])
                    count = N
                    break
            
            X = np.array(X_new)
            F = np.array(F_new)

        return all_X[-1] if all_X else lo.copy()

    def tournament_select(self, F, N, k=2):
        # Select k random indices, return the best one
        # "Best" is defined by non-dominated sorting rank + crowding distance
        # For simplicity in tournament, we can just use rank if we had it, 
        # but here we just pick random and compare dominance or sum.
        # Better: Pre-compute ranks? No, F changes.
        # Simple tournament: Pick k random, evaluate their "quality".
        # Quality: 1 if dominates all others in tournament, else 0.
        # If tie, use sum of objectives.
        
        indices = rng.choice(N, size=k, replace=False)
        best_idx = indices[0]
        best_f = F[best_idx]
        
        for idx in indices[1:]:
            f = F[idx]
            # Check dominance
            if np.all(f <= best_f) and np.any(f < best_f):
                best_idx = idx
                best_f = f
            elif np.all(best_f <= f) and np.any(best_f < f):
                # Current best dominates candidate
                pass
            else:
                # Incomparable: use sum
                if np.sum(f) < np.sum(best_f):
                    best_idx = idx
                    best_f = f
        return best_idx
# EVOLVE-BLOCK-END
