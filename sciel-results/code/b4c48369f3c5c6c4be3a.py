"""Multi-objective optimizer for RE34 using NSGA-II with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using NSGA-II style approach."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 325.0  # safety margin before 330s kill
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        pop_size = min(50, self.budget // 2)
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)
        
        def non_dominated_sort(F):
            """Perform non-dominated sorting, return list of fronts."""
            n = len(F)
            fronts = []
            S = [[] for _ in range(n)]  # Dominated by
            n_dom = np.zeros(n, dtype=int)  # Number dominated
            front = []
            
            for i in range(n):
                for j in range(i + 1, n):
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif dominates(F[j], F[i]):
                        S[j].append(i)
                        n_dom[i] += 1
            
            for i in range(n):
                if n_dom[i] == 0:
                    front.append(i)
            
            while front:
                fronts.append(front)
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                front = next_front
            
            # Assign front indices
            front_idx = np.full(n, -1, dtype=int)
            for fi, f in enumerate(fronts):
                for i in f:
                    front_idx[i] = fi
            return fronts, front_idx
        
        def crowding_distance(F, front_idx):
            """Calculate crowding distance for each point."""
            n = len(F)
            cd = np.zeros(n)
            
            for fi in range(np.max(front_idx) + 1):
                idx = np.where(front_idx == fi)[0]
                if len(idx) <= 2:
                    cd[idx] = np.inf
                    continue
                
                for m in range(self.n_obj):
                    vals = F[idx, m]
                    order = np.argsort(vals)
                    cd[idx[order[0]]] = np.inf
                    cd[idx[order[-1]]] = np.inf
                    val_range = vals[order[-1]] - vals[order[0]]
                    if val_range > 1e-12:
                        for k in range(1, len(idx) - 1):
                            cd[idx[order[k]]] += (vals[order[k + 1]] - vals[order[k - 1]]) / val_range
            
            return cd
        
        while used < self.budget:
            # Check time
            if time.time() > deadline - 5:
                break
            
            # Create offspring
            offspring_X = []
            offspring_F = []
            
            for _ in range(pop_size):
                # Tournament selection
                parents = []
                for _ in range(2):
                    i = rng.integers(pop_size)
                    j = rng.integers(pop_size)
                    # Simple selection based on front and crowding
                    fronts_i, front_idx_i = non_dominated_sort(F)
                    fronts_j, front_idx_j = non_dominated_sort(F)
                    if front_idx_i[i] < front_idx_j[j]:
                        parents.append(X[i])
                    elif front_idx_j[j] < front_idx_i[i]:
                        parents.append(X[j])
                    else:
                        # Use crowding distance
                        cd_i = crowding_distance(F, front_idx_i)
                        cd_j = crowding_distance(F, front_idx_j)
                        if cd_i[i] > cd_j[j]:
                            parents.append(X[i])
                        else:
                            parents.append(X[j])
                
                # Crossover (SBX)
                if rng.random() < 0.9:
                    child = np.zeros(self.dim)
                    for d in range(self.dim):
                        if rng.random() < 0.5:
                            beta = 1 + 2 * rng.random()
                            alpha = 15 + 2
                            if rng.random() < 0.5:
                                child[d] = 0.5 * ((1 + beta) * parents[0][d] + (1 - beta) * parents[1][d])
                            else:
                                child[d] = 0.5 * ((1 - beta) * parents[0][d] + (1 + beta) * parents[1][d])
                        else:
                            child[d] = parents[0][d]
                else:
                    child = parents[0].copy()
                
                # Mutation (polynomial)
                for d in range(self.dim):
                    if rng.random() < 0.9:
                        delta = rng.random()
                        if delta < 0.5:
                            mut = (2 * delta) ** (1 / 20) - 1
                        else:
                            mut = 1 - (2 * (1 - delta)) ** (1 / 20)
                        child[d] += mut * (hi[d] - lo[d])
                
                child = np.clip(child, lo, hi)
                offspring_X.append(child)
                offspring_F.append(problem(child))
                used += 1
                if time.time() > deadline - 2:
                    break
            
            if not offspring_X:
                break
            
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # Combine parents and offspring
            X_all = np.vstack([X, offspring_X])
            F_all = np.vstack([F, offspring_F])
            
            # Non-dominated sort
            fronts, front_idx = non_dominated_sort(F_all)
            
            # Select next generation
            new_X = []
            new_F = []
            cd = crowding_distance(F_all, front_idx)
            
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for i in front:
                        new_X.append(X_all[i])
                        new_F.append(F_all[i])
                else:
                    # Sort by crowding distance
                    front_sorted = sorted(front, key=lambda i: cd[i], reverse=True)
                    needed = pop_size - len(new_X)
                    for i in front_sorted[:needed]:
                        new_X.append(X_all[i])
                        new_F.append(F_all[i])
                    break
            
            if new_X:
                X = np.array(new_X)
                F = np.array(new_F)
        
        return X
# EVOLVE-BLOCK-END
