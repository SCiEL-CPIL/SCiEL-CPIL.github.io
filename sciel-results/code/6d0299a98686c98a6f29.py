"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # Population size
        pop_size = min(50, self.budget // 2)
        if pop_size < 10:
            pop_size = 10
            
        # Initial population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        def fast_non_dominated_sort(F):
            """Return list of fronts (indices) sorted by dominance level."""
            n = len(F)
            S = [[] for _ in range(n)]  # Dominated set
            n_dominated = np.zeros(n, dtype=int)
            fronts = []
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dominated[i] += 1
            
            # First front
            front = [i for i in range(n) if n_dominated[i] == 0]
            if not front:
                return [list(range(n))]
            fronts.append(front)
            
            while front:
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_dominated[j] -= 1
                        if n_dominated[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                front = next_front
            
            return fronts

        def crowding_distance(F, front_indices):
            """Compute crowding distance for points in a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            
            distances = np.zeros(n)
            n_obj = F.shape[1]
            
            for m in range(n_obj):
                obj_vals = F[front_indices, m]
                sorted_idx = np.argsort(obj_vals)
                distances[sorted_idx[0]] = np.inf
                distances[sorted_idx[-1]] = np.inf
                
                obj_range = obj_vals[sorted_idx[-1]] - obj_vals[sorted_idx[0]]
                if obj_range > 0:
                    for k in range(1, n - 1):
                        distances[sorted_idx[k]] += (obj_vals[sorted_idx[k+1]] - obj_vals[sorted_idx[k-1]]) / obj_range
            
            return distances

        def tournament_select(pop_X, pop_F, size=2):
            """Select parents via tournament selection based on non-dominated rank and crowding distance."""
            fronts = fast_non_dominated_sort(pop_F)
            rank = np.zeros(len(pop_F), dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = r
            
            # Crowding distance per front
            cd = np.zeros(len(pop_F))
            for front in fronts:
                if len(front) > 0:
                    cd[front] = crowding_distance(pop_F, front)
            
            # Tournament
            candidates = rng.integers(0, len(pop_X), size=size)
            best = candidates[0]
            for c in candidates[1:]:
                if rank[c] < rank[best] or (rank[c] == rank[best] and cd[c] > cd[best]):
                    best = c
            return best

        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Create offspring
            Q = []
            QF = []
            for _ in range(pop_size):
                p1 = tournament_select(X, F)
                p2 = tournament_select(X, F)
                
                # SBX crossover
                child = X[p1].copy()
                for d in range(self.dim):
                    if rng.random() < 0.5:
                        child[d] = X[p2][d]
                
                # Mutation
                if rng.random() < 0.8:
                    sigma = 0.1 * span
                    child += sigma * rng.standard_normal(self.dim)
                    child = np.clip(child, lo, hi)
                
                Q.append(child)
                QF.append(problem(child))
                used += 1
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
            
            if not Q:
                break
            
            Q = np.array(Q)
            QF = np.array(QF)
            
            # Combine parents and offspring
            R_X = np.vstack([X, Q])
            R_F = np.vstack([F, QF])
            R_size = len(R_X)
            
            # Non-dominated sort
            fronts = fast_non_dominated_sort(R_F)
            rank = np.zeros(R_size, dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = r
            
            # Crowding distance
            cd = np.zeros(R_size)
            for front in fronts:
                if len(front) > 0:
                    cd[front] = crowding_distance(R_F, front)
            
            # Select next population
            X_new = []
            F_new = []
            for front in fronts:
                if len(X_new) + len(front) <= pop_size:
                    for idx in front:
                        X_new.append(R_X[idx])
                        F_new.append(R_F[idx])
                else:
                    remaining = pop_size - len(X_new)
                    # Sort by crowding distance descending
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    for idx in sorted_front[:remaining]:
                        X_new.append(R_X[idx])
                        F_new.append(R_F[idx])
                    break
            
            X = np.array(X_new)
            F = np.array(F_new)
        
        return X
# EVOLVE-BLOCK-END
