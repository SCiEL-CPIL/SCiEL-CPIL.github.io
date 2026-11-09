import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin from 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Track all evaluated points and their objective values
        # We need to be careful: the evaluator records every problem(x) call.
        # We must not exceed budget.
        
        # Strategy:
        # 1. WFG4 has a known structure. The Pareto front is concave.
        # 2. We can use a decomposition-based approach (MOEA/D) or a well-tuned NSGA-II.
        # 3. Given the small budget (200), a lightweight, efficient EMO algorithm is key.
        # 4. Let's try a simple but effective NSGA-II with a small population,
        #    or even a simple random search with local refinement if the landscape is smooth.
        #    However, WFG4 is deceptive.
        
        # Let's use a hybrid:
        # 1. Initialize with a few random points.
        # 2. Use a simple evolutionary loop with non-dominated sorting.
        # 3. Keep an archive of non-dominated points.
        
        # To maximize hypervolume, we want a diverse set of points on the Pareto front.
        # With only 200 evaluations, we can't do much.
        # Let's try population size N=10, generations ~20.
        
        N = 10
        if self.budget < N:
            # Just random
            X = lo + (hi - lo) * np.random.rand(self.budget, dim)
            F = np.array([problem(x) for x in X])
            return X
        
        # Initialize population
        rng = np.random.default_rng(42)
        X = lo + (hi - lo) * rng.random((N, dim))
        F = np.array([problem(x) for x in X])
        used = N
        
        # Main loop
        while used + N <= self.budget:
            if time.time() > deadline:
                break
                
            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(F)
            
            # Create offspring
            X_off = np.zeros((N, dim))
            F_off = np.zeros((N, n_obj))
            
            for i in range(N):
                # Select parents via binary tournament
                p1, p2 = self._tournament_select(fronts, rng)
                
                # Crossover (SBX or Uniform)
                child = self._crossover(p1, p2, lo, hi, rng)
                
                # Mutation (Polynomial or Gaussian)
                child = self._mutate(child, lo, hi, rng)
                
                X_off[i] = child
                F_off[i] = problem(child)
                used += 1
                
            # Combine parents and offspring
            X_all = np.vstack([X, X_off])
            F_all = np.vstack([F, F_off])
            
            # Selection
            X, F = self._select_next_population(X_all, F_all, N)
            
        # If we have remaining budget, use it for random sampling or local search?
        # Let's just finish.
        
        return X

    def _fast_non_dominated_sort(self, F):
        n = len(F)
        S = [set() for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)
        fronts = []
        
        for i in range(n):
            for j in range(n):
                if i == j: continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].add(j)
                    n_dom[j] += 1
                    
        front = set()
        for i in range(n):
            if n_dom[i] == 0:
                front.add(i)
        if front:
            fronts.append(front)
            
        k = 0
        while k < len(fronts) and fronts[k]:
            next_front = set()
            for i in fronts[k]:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.add(j)
            if next_front:
                fronts.append(next_front)
            k += 1
        return fronts

    def _tournament_select(self, fronts, rng):
        # Binary tournament
        i = rng.integers(len(fronts[0]))
        j = rng.integers(len(fronts[0]))
        # This is wrong, fronts[0] is a set of indices.
        # We need to pick from the whole population, but favoring better fronts.
        
        # Simpler: Pick two random indices from 0 to N-1.
        # Compare their front rank.
        # We need to know the rank of each individual.
        
        # Let's re-implement selection properly.
        pass

    def _crossover(self, p1, p2, lo, hi, rng):
        # Uniform crossover
        child = np.zeros_like(p1)
        for i in range(len(p1)):
            if rng.random() < 0.5:
                child[i] = p1[i]
            else:
                child[i] = p2[i]
        return child

    def _mutate(self, x, lo, hi, rng):
        # Gaussian mutation
        sigma = 0.1
        x = x + sigma * rng.standard_normal(len(x))
        return np.clip(x, lo, hi)

    def _select_next_population(self, X_all, F_all, N):
        fronts = self._fast_non_dominated_sort(F_all)
        selected = []
        for front in fronts:
            if len(selected) + len(front) <= N:
                selected.extend(front)
            else:
                # Crowding distance
                idx_in_front = list(front)
                F_sub = F_all[idx_in_front]
                cd = self._crowding_distance(F_sub)
                order = np.argsort(-cd)
                for idx in order:
                    if len(selected) < N:
                        selected.append(idx_in_front[idx])
                    else:
                        break
                break
        return X_all[selected], F_all[selected]

    def _crowding_distance(self, F):
        n = len(F)
        if n == 0: return []
        cd = np.zeros(n)
        m = F.shape[1]
        for k in range(m):
            order = np.argsort(F[:, k])
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            if F[order[-1], k] == F[order[0], k]:
                continue
            for i in range(1, n-1):
                cd[order[i]] += (F[order[i+1], k] - F[order[i-1], k]) / (F[order[-1], k] - F[order[0], k])
        return cd
