import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 295.0  # 5s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Population size
        # With budget 200, a pop size of 20 allows ~10 generations if we generate 1 child per iter
        # Or we can generate a full new population each generation?
        # Standard NSGA-II: Pop N, generate N offspring, select N from 2N.
        # Budget 200. If N=20, we can do 200/20 = 10 generations.
        # This is reasonable.
        pop_size = 20
        
        # Initialize
        X = np.random.uniform(lo, hi, size=(pop_size, n_dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        def non_dominated_sort(F):
            """
            Returns list of lists of indices, sorted by non-dominated front.
            """
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dom = [0] * n             # n_dom[i] = number of individuals dominating i
            rank = [0] * n
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1
            
            fronts = []
            current_front = [i for i in range(n) if n_dom[i] == 0]
            rank = [0] * n
            for i in current_front:
                rank[i] = 0
            
            k = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                current_front = next_front
                k += 1
            return fronts, rank

        def crowding_distance(F, indices):
            """
            Calculate crowding distance for individuals in 'indices' based on F.
            Returns dict {idx: distance}
            """
            if len(indices) <= 2:
                return {i: float('inf') for i in indices}
            
            m = len(indices)
            dist = {i: 0.0 for i in indices}
            
            for k in range(n_obj):
                # Sort indices by objective k
                sorted_indices = sorted(indices, key=lambda i: F[i][k])
                
                # End points get infinity
                dist[sorted_indices[0]] = float('inf')
                dist[sorted_indices[-1]] = float('inf')
                
                f_min = F[sorted_indices[0]][k]
                f_max = F[sorted_indices[-1]][k]
                if f_max == f_min:
                    continue
                    
                for i in range(1, m - 1):
                    idx = sorted_indices[i]
                    dist[idx] += (F[sorted_indices[i+1]][k] - F[sorted_indices[i-1]][k]) / (f_max - f_min)
            
            return dist

        # Main Loop
        while used < budget:
            if time.time() > deadline:
                break
            
            # Generate offspring
            # We generate 1 offspring at a time to save time and allow early termination if needed, 
            # but NSGA-II logic usually works with full generations. 
            # Given budget 200 and pop 20, we can do "generations" of 20 offspring.
            # Let's do one offspring per iteration for flexibility, but use NSGA-II selection on the combined pool?
            # No, standard NSGA-II: 
            # 1. Select parents from X
            # 2. Generate N offspring
            # 3. Combine X and Offspring
            # 4. Select N best
            
            # To strictly adhere to budget, we can generate 1 child, combine, select. 
            # This is "steady-state" NSGA-II.
            
            # Select 2 parents for DE mutation
            r1 = np.random.randint(0, pop_size)
            r2 = np.random.randint(0, pop_size)
            r3 = np.random.randint(0, pop_size)
            
            # Ensure r1, r2, r3 are distinct if possible? Not strictly required for DE, but good.
            # Simple DE: v = x[r1] + 0.5 * (x[r2] - x[r3])
            v = X[r1] + 0.5 * (X[r2] - X[r3])
            
            # Add some random noise to escape local optima?
            # v += 0.05 * (hi - lo) * np.random.normal(size=n_dim)
            
            v = np.clip(v, lo, hi)
            
            fv = problem(v)
            used += 1
            
            # Combine
            X_new = np.vstack([X, v])
            F_new = np.vstack([F, fv])
            
            # Non-dominated sort
            fronts, ranks = non_dominated_sort(F_new)
            
            # Selection
            selected_indices = []
            for front in fronts:
                if len(selected_indices) + len(front) <= pop_size:
                    selected_indices.extend(front)
                else:
                    # Need to pick from this front based on crowding distance
                    remaining = pop_size - len(selected_indices)
                    cd = crowding_distance(F_new, front)
                    # Sort by crowding distance descending
                    front_sorted = sorted(front, key=lambda i: cd[i], reverse=True)
                    selected_indices.extend(front_sorted[:remaining])
                    break
            
            X = X_new[selected_indices]
            F = F_new[selected_indices]
            
        return X
