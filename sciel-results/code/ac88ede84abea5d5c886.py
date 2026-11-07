import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        mid = (lo + hi) / 2.0
        half_range = (hi - lo) / 2.0
        
        def non_dominated_sort(F):
            n = len(F)
            if n == 0:
                return np.array([], dtype=int), np.array([])
            
            ranks = np.zeros(n, dtype=int)
            crowding = np.zeros(n)
            
            front = set(range(n))
            rank = 0
            while len(front) > 0:
                front_list = list(front)
                nd_in_front = set()
                for i in front_list:
                    is_nd = True
                    for j in front_list:
                        if i == j:
                            continue
                        if np.all(F[i] >= F[j]) and np.any(F[i] > F[j]):
                            is_nd = False
                            break
                    if is_nd:
                        nd_in_front.add(i)
                
                for i in nd_in_front:
                    ranks[i] = rank
                
                if len(nd_in_front) > 1:
                    for k in range(n_obj):
                        sorted_front = sorted(nd_in_front, key=lambda x: F[x, k])
                        f_min = F[sorted_front[0], k]
                        f_max = F[sorted_front[-1], k]
                        if f_max <= f_min:
                            continue
                        for idx, i in enumerate(sorted_front):
                            if idx == 0 or idx == len(sorted_front) - 1:
                                crowding[i] = np.inf
                            else:
                                crowding[i] += (F[sorted_front[idx+1], k] - F[sorted_front[idx-1], k]) / (f_max - f_min)
                
                rank += 1
                front = front - nd_in_front
            
            return ranks, crowding
        
        def reference_directions_3d(n):
            """Generate n reference directions on the 2-simplex for 3 objectives."""
            dirs = []
            # Use systematic grid: for i+j+k = m
            m = int(round((2 * n) ** 0.5))
            while m < 2:
                m += 1
            for i in range(m + 1):
                for j in range(m + 1 - i):
                    k = m - i - j
                    if i + j + k == m:
                        dirs.append(np.array([i, j, k], dtype=float) / m)
            dirs = np.array(dirs)
            if len(dirs) > n:
                idx = np.linspace(0, len(dirs) - 1, n).astype(int)
                dirs = dirs[idx]
            elif len(dirs) < n:
                extra = np.random.dirichlet(np.ones(3), size=n - len(dirs))
                dirs = np.vstack([dirs, extra])
            return dirs
        
        # Number of reference directions / subproblems
        n_sub = min(30, max(15, self.budget // 10))
        ref_dirs = reference_directions_3d(n_sub)
        
        # Population size: 2 per subproblem
        pop_size = 2 * n_sub
        
        # Phase 1: Initialize population with diversity
        X = np.random.uniform(lo, hi, size=(pop_size, dim))
        F = np.array([problem(x) for x in X])
        evaluations = pop_size
        
        # Assign each individual to its closest reference direction (by weighted sum)
        def assign_to_directions(X, F, ref_dirs):
            n = len(X)
            assignment = np.zeros(n, dtype=int)
            for i in range(n):
                scores = F[i] @ ref_dirs.T
                assignment[i] = np.argmin(scores)
            return assignment
        
        assignment = assign_to_directions(X, F, ref_dirs)
        
        # Track best per direction
        best_idx_per_dir = np.full(n_sub, -1, dtype=int)
        for d in range(n_sub):
            members = np.where(assignment == d)[0]
            if len(members) > 0:
                # Pick the one with best weighted sum for this direction
                ws = F[members] @ ref_dirs[d]
                best_idx_per_dir[d] = members[np.argmin(ws)]
        
        # Evolutionary loop with decomposition guidance
        while evaluations < self.budget:
            if time.time() > deadline:
                break
            
            # Generate offspring: for each subproblem, select parents biased toward that direction
            X_off = []
            F_off = []
            
            for d in range(n_sub):
                if evaluations >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                # Select 2 parents: one from this direction's best, one random
                if best_idx_per_dir[d] >= 0:
                    p1 = X[best_idx_per_dir[d]]
                else:
                    p1 = X[np.random.randint(len(X))]
                
                # Second parent: random from population
                p2 = X[np.random.randint(len(X))]
                
                # SBX crossover
                child = np.zeros(dim)
                for i in range(dim):
                    if abs(p1[i] - p2[i]) < 1e-14:
                        child[i] = p1[i]
                    else:
                        u = np.random.rand()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / 21.0)
                        else:
                            beta = (1.0 / (2 * (1 - u))) ** (1.0 / 21.0)
                        if np.random.rand() < 0.5:
                            child[i] = 0.5 * ((p1[i] + p2[i]) - beta * (p1[i] - p2[i]))
                        else:
                            child[i] = 0.5 * ((p1[i] + p2[i]) + beta * (p1[i] - p2[i]))
                
                # Polynomial mutation
                for i in range(dim):
                    if np.random.rand() < 1.0 / dim:
                        range_i = hi[i] - lo[i]
                        if range_i < 1e-14:
                            continue
                        y = (child[i] - lo[i]) / range_i
                        y = np.clip(y, 0, 1)
                        u = np.random.rand()
                        if u < 0.5:
                            delta1 = (1 + 2 * y) ** 21
                            delta = (1.0 / delta1) ** (1.0 / 21.0) - 1
                        else:
                            delta2 = (1 + 2 * (1 - y)) ** 21
                            delta = 1 - (1.0 / delta2) ** (1.0 / 21.0)
                        if delta >= 0:
                            child[i] = lo[i] + (y + delta * (1 - y)) * range_i
                        else:
                            child[i] = lo[i] + (y + delta * y) * range_i
                
                child = np.clip(child, lo, hi)
                f_child = np.array(problem(child))
                evaluations += 1
                X_off.append(child)
                F_off.append(f_child)
            
            if len(X_off) == 0:
                break
            
            X_off = np.array(X_off)
            F_off = np.array(F_off)
            
            # Combine parent and offspring
            X_all = np.vstack([X, X_off])
            F_all = np.vstack([F, F_off])
            
            # Non-dominated sorting
            ranks, crowding = non_dominated_sort(F_all)
            
            # Select next population: prioritize by rank, then crowding
            sort_idx = np.lexsort((-crowding, ranks))
            X = X_all[sort_idx[:pop_size]]
            F = F_all[sort_idx[:pop_size]]
            
            # Reassign to directions
            assignment = assign_to_directions(X, F, ref_dirs)
            best_idx_per_dir = np.full(n_sub, -1, dtype=int)
            for d in range(n_sub):
                members = np.where(assignment == d)[0]
                if len(members) > 0:
                    ws = F[members] @ ref_dirs[d]
                    best_idx_per_dir[d] = members[np.argmin(ws)]
        
        return np.zeros((1, dim))
