import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """Multi-objective optimizer using Differential Evolution."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """
        Perform fast non-dominated sorting.
        Returns a list of fronts, where each front is a list of indices.
        """
        n = len(F)
        if n == 0:
            return []

        S = [[] for _ in range(n)]
        n_dom = [0] * n
        fronts = [[]]

        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                # Check if p dominates q
                if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                    S[p].append(q)
                    n_dom[p] += 1
                # Check if q dominates p
                elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                    n_dom[q] += 1

        # First front
        for i in range(n):
            if n_dom[i] == 0:
                fronts[0].append(i)

        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    n_dom[q] -= 1
                    if n_dom[q] == 0:
                        next_front.append(q)
            i += 1
            if next_front:
                fronts.append(next_front)
            else:
                break
        
        return fronts[:-1] if len(fronts) > 1 else fronts

    def _crowding_distance(self, F, front_indices):
        """
        Calculate crowding distance for a given front.
        """
        n = len(front_indices)
        if n <= 2:
            return [float('inf')] * n

        dist = [0.0] * n
        m = self.n_obj

        for k in range(m):
            # Sort indices by k-th objective
            sorted_indices = sorted(front_indices, key=lambda idx: F[idx][k])
            
            # Assign infinite distance to boundary points
            dist[sorted_indices[0]] = float('inf')
            dist[sorted_indices[-1]] = float('inf')
            
            f_min = F[sorted_indices[0]][k]
            f_max = F[sorted_indices[-1]][k]
            range_val = f_max - f_min
            if range_val == 0:
                continue

            for i in range(1, n - 1):
                dist[sorted_indices[i]] += (F[sorted_indices[i + 1]][k] - F[sorted_indices[i - 1]][k]) / range_val

        return dist

    def _tournament_select(self, ranks, crowding, size=2):
        """
        Tournament selection based on rank and crowding distance.
        """
        indices = np.random.randint(0, len(ranks), size)
        best_idx = indices[0]
        for i in indices[1:]:
            if ranks[i] < ranks[best_idx]:
                best_idx = i
            elif ranks[i] == ranks[best_idx]:
                if crowding[i] > crowding[best_idx]:
                    best_idx = i
        return best_idx

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        # With budget 200, N=40 allows for 1 init (40) + 4 gens (160) = 200 evals.
        # DE works well with larger populations.
        N = 40
        
        # Initial Population
        # Use Sobol for better spread than random
        sobol = qmc.Sobol(d=self.dim, scramble=True)
        X = sobol.random(N)
        X = lo + X * (hi - lo)
        
        all_X = []
        all_F = []
        
        # Evaluate initial population
        for x in X:
            f = problem(x)
            all_X.append(x)
            all_F.append(f)
        
        X = np.array(all_X)
        F = np.array(all_F)
        
        # DE Parameters
        F_factor = 0.8
        CR = 0.9
        
        # Main Evolutionary Loop
        while True:
            if time.time() > deadline - 5.0:
                break
            
            # Check budget
            if len(all_X) + N > self.budget:
                # If we can't fit a full generation, stop
                break
                
            # 1. Non-dominated sorting and crowding distance for current population
            fronts = self._fast_non_dominated_sort(F)
            ranks = np.zeros(N, dtype=int)
            crowding = np.zeros(N)
            
            for i, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = i
                dists = self._crowding_distance(F, front)
                for j, idx in enumerate(front):
                    crowding[idx] = dists[j]
            
            # 2. Generate Offspring using Differential Evolution
            offspring_X = []
            offspring_F = []
            
            for i in range(N):
                if len(all_X) + len(offspring_X) + 1 > self.budget:
                    break
                
                # Select 3 distinct parents a, b, c
                # Ensure they are distinct from each other and preferably from i, though DE allows i
                # Standard DE/rand/1: v = a + F*(b-c)
                # We pick random distinct indices
                a_idx = np.random.randint(0, N)
                b_idx = np.random.randint(0, N)
                c_idx = np.random.randint(0, N)
                
                # Ensure distinctness
                while b_idx == a_idx:
                    b_idx = np.random.randint(0, N)
                while c_idx == a_idx or c_idx == b_idx:
                    c_idx = np.random.randint(0, N)
                
                # Differential Evolution Vector
                v = X[a_idx] + F_factor * (X[b_idx] - X[c_idx])
                
                # Clip to bounds
                v = np.clip(v, lo, hi)
                
                # Crossover
                # j_rand ensures at least one gene is taken from v
                j_rand = np.random.randint(0, self.dim)
                u = np.random.rand(self.dim) < CR
                u[j_rand] = True
                
                # Offspring
                o = np.where(u, v, X[i])
                
                # Evaluate
                f_o = problem(o)
                offspring_X.append(o)
                offspring_F.append(f_o)
                
                # Record for global archive
                all_X.append(o)
                all_F.append(f_o)
            
            if not offspring_X:
                break
                
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # 3. Combine Parent and Offspring
            X_combined = np.vstack((X, offspring_X))
            F_combined = np.vstack((F, offspring_F))
            
            # 4. Selection (Elitism)
            fronts = self._fast_non_dominated_sort(F_combined)
            
            new_X = []
            new_F = []
            n_selected = 0
            for front in fronts:
                if n_selected + len(front) <= N:
                    # Take all
                    for idx in front:
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])
                    n_selected += len(front)
                else:
                    # Need to select from this front based on crowding
                    needed = N - n_selected
                    dists = self._crowding_distance(F_combined, front)
                    
                    # Sort by crowding distance descending
                    sorted_pairs = sorted(zip(front, dists), key=lambda x: x[1], reverse=True)
                    for idx, _ in sorted_pairs[:needed]:
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])
                    n_selected += needed
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)
            
            if time.time() > deadline - 10.0:
                break

        return np.array(all_X)
