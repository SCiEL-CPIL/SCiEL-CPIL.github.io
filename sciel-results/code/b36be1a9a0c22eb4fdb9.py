"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style evolutionary multi-objective search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # Safety margin before 330s limit
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        # Initial population
        pop_size = 40
        if self.budget < pop_size:
            pop_size = max(1, self.budget)
            
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track all evaluated points for final archive
        all_X = [X.copy()]
        all_F = [F.copy()]
        
        def fast_non_dominated_sort(F_arr):
            """Returns list of fronts (indices) and rank array."""
            n = len(F_arr)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            rank = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        S[i].append(j)
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    rank[i] = 0
            
            fronts = []
            current_front = [i for i in range(n) if rank[i] == 0]
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            rank[j] = len(fronts)
                            next_front.append(j)
                current_front = next_front
            return fronts, rank

        def crowding_distance(F_arr, indices):
            """Returns crowding distances for the given indices in F_arr."""
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            
            m = len(indices)
            dist = np.zeros(m)
            k = F_arr.shape[1]
            
            for obj in range(k):
                vals = F_arr[indices, obj]
                sorted_idx = np.argsort(vals)
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                min_val = vals[sorted_idx[0]]
                max_val = vals[sorted_idx[-1]]
                if max_val - min_val < 1e-12:
                    continue
                for i in range(1, m - 1):
                    dist[sorted_idx[i]] += (vals[sorted_idx[i+1]] - vals[sorted_idx[i-1]]) / (max_val - min_val)
            
            return dist

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Combine current population with new offspring (we generate pop_size offspring)
            # For simplicity and speed, let's do a standard NSGA-II step
            # Generate offspring
            offspring_X = np.zeros((pop_size, self.dim))
            
            for i in range(pop_size):
                # Tournament selection
                parents = []
                for _ in range(2):
                    t1 = rng.integers(pop_size)
                    t2 = rng.integers(pop_size)
                    f1 = F[t1]
                    f2 = F[t2]
                    # Simple dominance check
                    if np.all(f1 <= f2) and np.any(f1 < f2):
                        parents.append(t1)
                    elif np.all(f2 <= f1) and np.any(f2 < f1):
                        parents.append(t2)
                    else:
                        # Non-dominated, pick randomly
                        if rng.random() > 0.5:
                            parents.append(t1)
                        else:
                            parents.append(t2)
                
                p1 = X[parents[0]]
                p2 = X[parents[1]]
                
                # Simulated Binary Crossover
                if rng.random() < 0.9:
                    child = np.zeros(self.dim)
                    for d in range(self.dim):
                        if rng.random() < 0.5:
                            beta = 1.0 / (1.0 + 2.0 * rng.random())
                            alpha = 2.0
                            beta_q = (beta * (1.0 + (alpha + 1.0) * abs(p1[d] - p2[d]) / (hi[d] - lo[d]))) ** (1.0 / (alpha + 1.0))
                            beta_q = min(1.0, max(0.0, beta_q))
                            child[d] = 0.5 * ((1.0 + beta_q) * p1[d] + (1.0 - beta_q) * p2[d])
                        else:
                            beta = 1.0 / (1.0 + 2.0 * rng.random())
                            alpha = 2.0
                            beta_q = (beta * (1.0 + (alpha + 1.0) * abs(p2[d] - p1[d]) / (hi[d] - lo[d]))) ** (1.0 / (alpha + 1.0))
                            beta_q = min(1.0, max(0.0, beta_q))
                            child[d] = 0.5 * ((1.0 + beta_q) * p2[d] + (1.0 - beta_q) * p1[d])
                else:
                    child = p1.copy()
                
                # Polynomial Mutation
                if rng.random() < 1.0 / self.dim:
                    delta = rng.random()
                    if delta < 0.5:
                        mut_pow = 2.0 * (delta + 1.0) ** (1.0 / (20.0 + 1.0))
                    else:
                        mut_pow = 2.0 * (1.0 - delta + 1.0) ** (1.0 / (20.0 + 1.0))
                    mut_pow = min(1.0, max(0.0, mut_pow))
                    child[d] += (hi[d] - lo[d]) * mut_pow * (rng.random() * 2 - 1)
                
                child = np.clip(child, lo, hi)
                offspring_X[i] = child
            
            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += pop_size
            
            # Combine
            X_comb = np.vstack([X, offspring_X])
            F_comb = np.vstack([F, offspring_F])
            
            # Non-dominated sort
            fronts, rank = fast_non_dominated_sort(F_comb)
            
            # Select next population
            X_new = []
            F_new = []
            for front in fronts:
                if len(X_new) + len(front) <= pop_size:
                    X_new.extend(X_comb[front])
                    F_new.extend(F_comb[front])
                else:
                    # Need crowding distance
                    remaining = pop_size - len(X_new)
                    cd = crowding_distance(F_comb, front)
                    # Sort by crowding distance (descending)
                    sorted_front = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    X_new.extend(X_comb[sorted_front[:remaining]])
                    F_new.extend(F_comb[sorted_front[:remaining]])
                    break
            
            X = np.array(X_new)
            F = np.array(F_new)
            
            # Update archive
            all_X.append(X)
            all_F.append(F)

        # Return the best non-dominated points found so far
        # We need to return X, but the task says "Every returned objective vector is recorded".
        # The return value is ignored. But we should return something valid.
        # Let's return the current population X.
        return X
# EVOLVE-BLOCK-END
