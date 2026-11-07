import numpy as np
from scipy.optimize import differential_evolution # Not needed if implementing manually

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Determine population size and generations
        # Budget is 200. Let's use Pop=20, Gen=10.
        pop_size = 20
        n_gen = self.budget // pop_size
        if n_gen < 1:
            pop_size = self.budget
            n_gen = 1
            
        # Initialize population
        X = lo + (hi - lo) * np.random.rand(pop_size, self.dim)
        F = np.array([problem(x) for x in X])
        
        # Helper: Non-dominated sort
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)] # S[i] = list of indices dominated by i
            n_d = np.zeros(n, dtype=int) # n_d[i] = count of individuals dominating i
            rank = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_d[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_d[i] += 1
            
            current_front = []
            k = 0
            for i in range(n):
                if n_d[i] == 0:
                    rank[i] = 0
                    current_front.append(i)
                    
            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_d[j] -= 1
                        if n_d[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                k += 1
                current_front = next_front
            return rank

        # Helper: Crowding distance
        def crowding_distance(F, rank, indices_in_front):
            if len(indices_in_front) <= 2:
                return np.full(len(indices_in_front), np.inf)
            
            dist = np.zeros(len(F))
            for m in range(self.n_obj):
                f_vals = F[indices_in_front, m]
                min_val = np.min(f_vals)
                max_val = np.max(f_vals)
                if max_val == min_val:
                    continue
                # Sort indices by this objective
                sorted_idx = np.argsort(f_vals)
                dist[indices_in_front[sorted_idx[0]]] = np.inf
                dist[indices_in_front[sorted_idx[-1]]] = np.inf
                for i in range(1, len(sorted_idx) - 1):
                    idx = indices_in_front[sorted_idx[i]]
                    prev_idx = indices_in_front[sorted_idx[i-1]]
                    next_idx = indices_in_front[sorted_idx[i+1]]
                    dist[idx] += (F[next_idx, m] - F[prev_idx, m]) / (max_val - min_val)
            return dist

        # Helper: Tournament selection
        def tournament_select(X, F, rank, cd, k=2):
            idxs = np.random.randint(0, len(X), k)
            best = 0
            for i in range(1, k):
                if rank[idxs[i]] < rank[idxs[best]]:
                    best = i
                elif rank[idxs[i]] == rank[idxs[best]] and cd[idxs[i]] > cd[idxs[best]]:
                    best = i
            return X[idxs[best]].copy()

        # Main loop
        for g in range(n_gen):
            # Create offspring
            Q = []
            FQ = []
            for _ in range(pop_size):
                p1 = tournament_select(X, F, rank, cd)
                p2 = tournament_select(X, F, rank, cd)
                
                # SBX Crossover
                # Simple blending crossover for simplicity and robustness
                alpha = np.random.rand(self.dim)
                child = alpha * p1 + (1 - alpha) * p2
                
                # Mutation
                # Gaussian mutation
                sigma = 0.1
                child = child + sigma * (hi - lo) * np.random.randn(self.dim)
                child = np.clip(child, lo, hi)
                
                Q.append(child)
                FQ.append(problem(child))
            
            Q = np.array(Q)
            FQ = np.array(FQ)
            
            # Combine
            R_X = np.vstack([X, Q])
            R_F = np.vstack([F, FQ])
            
            # Sort
            rank = non_dominated_sort(R_F)
            
            # Select next generation
            P_next = []
            F_next = []
            remaining = pop_size
            k = 0
            while remaining > 0:
                front_indices = np.where(rank == k)[0]
                if len(front_indices) <= remaining:
                    P_next.extend(R_X[front_indices])
                    F_next.extend(R_F[front_indices])
                    remaining -= len(front_indices)
                    k += 1
                else:
                    # Need to select subset of this front
                    cd = crowding_distance(R_F, rank, front_indices)
                    # Sort by crowding distance descending
                    sorted_front = front_indices[np.argsort(-cd[front_indices])]
                    selected = sorted_front[:remaining]
                    P_next.extend(R_X[selected])
                    F_next.extend(R_F[selected])
                    remaining = 0
                    break
            
            X = np.array(P_next)
            F = np.array(F_next)
            # Recalculate rank and cd for next iteration's selection
            rank = non_dominated_sort(F)
            # Calculate cd for the whole population? No, only needed for selection.
            # For tournament, we need cd for all individuals in X.
            # Let's compute cd for all individuals in X based on their front.
            cd = np.zeros(len(X))
            for r in np.unique(rank):
                front_idx = np.where(rank == r)[0]
                cd[front_idx] = crowding_distance(F, rank, front_idx)
                
        return X
