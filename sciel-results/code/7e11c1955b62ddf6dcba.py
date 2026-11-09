import numpy as np
import time

class Optimizer:
        def __init__(self, budget, dim, n_obj):
            self.budget = budget
            self.dim = dim
            self.n_obj = n_obj

        def __call__(self, problem, bounds):
            start_time = time.time()
            deadline = start_time + 290.0
            
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            
            N = 20 # Population size
            budget_used = 0
            max_evals = self.budget
            
            # Halton Sequence Generator
            def halton(index, base):
                result = 0.0
                f = 1.0 / base
                i = index
                while i > 0:
                    result += f * (i % base)
                    i = i // base
                    f /= base
                return result

            # Generate N Halton points in [0, 1]^dim
            X = np.zeros((N, self.dim))
            for i in range(N):
                for d in range(self.dim):
                    # Use a different prime base for each dimension
                    # Primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
                    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
                    base = primes[d % len(primes)]
                    X[i, d] = halton(i + 1, base) # 1-based index for Halton
            
            # Scale to bounds
            X = lo + X * (hi - lo)
            
            # Evaluate
            F = np.array([problem(x) for x in X])
            budget_used += N
            
            # Helper: Non-dominated sort
            def non_dominated_sort(F):
                n = len(F)
                S = [[] for _ in range(n)]
                nd = np.zeros(n, dtype=int)
                fronts = []
                for i in range(n):
                    for j in range(n):
                        if i == j: continue
                        if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                            S[i].append(j)
                            nd[i] += 1 # Wait, nd counts dominators. 
                            # If i dominates j, then j has a dominator. 
                            # So nd[j] should increase.
                            # Let's rewrite carefully.
                # Correct Non-dominated sort
                S = [[] for _ in range(n)]
                nd = np.zeros(n, dtype=int)
                fronts = []
                for i in range(n):
                    for j in range(n):
                        if i == j: continue
                        # If i dominates j
                        if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                            S[i].append(j)
                            nd[j] += 1 # j is dominated by i
                current_front = [i for i in range(n) if nd[i] == 0]
                if current_front:
                    fronts.append(current_front)
                k = 0
                while current_front:
                    next_front = []
                    for i in current_front:
                        for j in S[i]:
                            nd[j] -= 1
                            if nd[j] == 0:
                                next_front.append(j)
                    if next_front:
                        fronts.append(next_front)
                    current_front = next_front
                    k += 1
                return fronts

            def crowding_distance(F, indices):
                if len(indices) <= 2:
                    return np.full(len(indices), np.inf)
                cd = np.zeros(len(indices))
                n_obj = F.shape[1]
                for m in range(n_obj):
                    vals = F[indices, m]
                    order = np.argsort(vals)
                    cd[order[0]] = np.inf
                    cd[order[-1]] = np.inf
                    rng_val = vals[order[-1]] - vals[order[0]]
                    if rng_val > 1e-15:
                        for k in range(1, len(order) - 1):
                            cd[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / rng_val
                return cd

            def tournament_select(F, X, n_sel):
                fronts = non_dominated_sort(F)
                selected = []
                for front in fronts:
                    if len(selected) + len(front) <= n_sel:
                        selected.extend(front)
                    else:
                        remaining = n_sel - len(selected)
                        cd = crowding_distance(F, front)
                        order = np.argsort(-cd)
                        selected.extend(front[order[:remaining]])
                        break
                return selected

            # Main Loop
            CR = 0.9
            F_mut = 0.8
            rng = np.random.default_rng(42)

            while budget_used < max_evals:
                if time.time() > deadline - 5.0:
                    break
                
                # Select parents
                parent_indices = tournament_select(F, X, N)
                
                for i in range(N):
                    if budget_used >= max_evals or time.time() > deadline - 5.0:
                        break
                    
                    # Find distinct parents
                    candidates = [p for p in parent_indices if p != i]
                    if len(candidates) < 3:
                        candidates = [p for p in range(N) if p != i]
                    if len(candidates) < 3:
                        continue
                    
                    r1, r2, r3 = rng.choice(candidates, size=3, replace=False)
                    
                    # DE Mutation
                    v = X[r1] + F_mut * (X[r2] - X[r3])
                    v = np.clip(v, lo, hi)
                    
                    # Crossover
                    trial = X[i].copy()
                    j_rand = rng.integers(self.dim)
                    for j in range(self.dim):
                        if rng.random() < CR or j == j_rand:
                            trial[j] = v[j]
                    trial = np.clip(trial, lo, hi)
                    
                    # Evaluate
                    f_trial = problem(trial)
                    budget_used += 1
                    
                    # Selection
                    if np.all(f_trial <= F[i]) and np.any(f_trial < F[i]):
                        X[i] = trial
                        F[i] = f_trial
                    elif not (np.all(F[i] <= f_trial) and np.any(F[i] < F[i])):
                        # Incomparable
                        # Use crowding distance heuristic? 
                        # Or just accept with probability 0.5?
                        # Better: Check front rank.
                        # For speed, just random or keep old?
                        # Let's keep old if we can't easily determine rank.
                        # Actually, standard NSGA-II replaces if non-dominated.
                        # But we need to maintain population size.
                        # If incomparable, we can use a simple heuristic:
                        # If trial is in a better front (lower index), replace.
                        # Calculating front for single point is expensive.
                        # Let's just keep the original to avoid noise, or random.
                        # Let's try: if random < 0.5, replace.
                        if rng.random() < 0.5:
                            X[i] = trial
                            F[i] = f_trial

            return X
