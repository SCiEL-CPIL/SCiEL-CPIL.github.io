# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for RE34 (3 objectives, 5 variables, bounds [1,3]).
    
    Strategy:
    1. Latin Hypercube Sampling for initial spread
    2. Differential Evolution with crowding distance for diversity
    3. Fast non-dominated sorting
    4. Hypervolume-driven selection
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.elapsed_start = None
        self.time_limit = 325.0

    def _elapsed(self):
        if self.elapsed_start is None:
            return 0.0
        return time.time() - self.elapsed_start

    def _dominates(self, a, b):
        """Returns True if a dominates b (all <=, at least one <)"""
        return np.all(a <= b) and np.any(a < b)

    def _fast_non_dominated_sort(self, F):
        """
        Fast non-dominated sorting.
        Returns: list of fronts (each front is a list of indices), 
                 and a dictionary of crowding distances.
        """
        n = len(F)
        if n == 0:
            return [], {}
        
        # S[i] = set of indices dominated by i
        # n_i = number of solutions dominating i
        S = [[] for _ in range(n)]
        n_i = [0] * n
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    n_i[i] += 1
        
        # First front
        front1 = [i for i in range(n) if n_i[i] == 0]
        fronts = []
        if front1:
            fronts.append(front1)
        
        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in S[i]:
                    n_i[j] -= 1
                    if n_i[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            k += 1
        
        # Crowding distance
        crowding = np.zeros(n)
        for front in fronts:
            if len(front) <= 2:
                for idx in front:
                    crowding[idx] = float('inf')
                continue
            for m in range(self.n_obj):
                # Sort by objective m
                sorted_idx = sorted(front, key=lambda i: F[i][m])
                crowding[sorted_idx[0]] = float('inf')
                crowding[sorted_idx[-1]] = float('inf')
                f_min = F[sorted_idx[0]][m]
                f_max = F[sorted_idx[-1]][m]
                if f_max - f_min > 1e-12:
                    for i in range(1, len(sorted_idx) - 1):
                        idx = sorted_idx[i]
                        crowding[idx] += (F[sorted_idx[i+1]][m] - F[sorted_idx[i-1]][m]) / (f_max - f_min)
        
        return fronts, crowding

    def __call__(self, problem, bounds):
        self.elapsed_start = time.time()
        rng = np.random.default_rng(42)
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        if self._elapsed() > self.time_limit:
            return np.empty((0, dim))
        
        # Phase 1: Latin Hypercube Sampling for initial population
        pop_size = min(100, max(50, budget // 4))
        X = np.empty((pop_size, dim))
        for j in range(dim):
            perm = rng.permutation(pop_size)
            X[:, j] = (perm + rng.random(pop_size)) / pop_size
        X = lo + (hi - lo) * X
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Maintain archive of all non-dominated solutions
        archive_X = list(X)
        archive_F = list(F)
        
        # Track all evaluated solutions
        all_X = [X.copy()]
        all_F = [F.copy()]
        
        # DE parameters
        CR = 0.9
        F_mut = 0.8
        archive_size = 50
        
        # Iterative DE loop
        while used < budget:
            if self._elapsed() > self.time_limit - 5:
                break
            
            # Get current non-dominated archive
            fronts, crowding = self._fast_non_dominated_sort(np.array(archive_F))
            
            # Select parents from first few fronts
            parent_pool = []
            for front in fronts[:3]:
                parent_pool.extend(front)
            
            if len(parent_pool) < 3:
                # Pad with random archive points
                while len(parent_pool) < 3 and len(archive_F) > 0:
                    parent_pool.append(rng.integers(len(archive_F)))
            
            if len(parent_pool) < 3:
                break
            
            # Generate offspring
            n_offspring = min(20, budget - used)
            offspring_X = np.empty((n_offspring, dim))
            
            for i in range(n_offspring):
                # Select 3 distinct parents
                idxs = rng.choice(parent_pool, size=3, replace=False)
                p1, p2, p3 = archive_X[idxs[0]], archive_X[idxs[1]], archive_X[idxs[2]]
                
                # Mutation
                v = p1 + F_mut * (p2 - p3)
                
                # Crossover
                cross_point = rng.integers(dim)
                j_rand = rng.integers(dim)
                u = np.zeros(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = p1[j]
                
                offspring_X[i] = np.clip(u, lo, hi)
            
            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += n_offspring
            
            # Update archive with non-dominated offspring
            for x, f in zip(offspring_X, offspring_F):
                # Check if f is dominated by any in archive
                dominated = False
                for af in archive_F:
                    if np.all(af <= f) and np.any(af < f):
                        dominated = True
                        break
                
                if not dominated:
                    # Remove archive points dominated by f
                    new_archive_X = []
                    new_archive_F = []
                    for ax, af in zip(archive_X, archive_F):
                        if not (np.all(f <= af) and np.any(f < af)):
                            new_archive_X.append(ax)
                            new_archive_F.append(af)
                    new_archive_X.append(x)
                    new_archive_F.append(f)
                    archive_X = new_archive_X
                    archive_F = new_archive_F
                    
                    # Keep archive size manageable
                    if len(archive_X) > archive_size * 2:
                        # Sort by hypervolume contribution and keep top
                        fronts, crowding = self._fast_non_dominated_sort(np.array(archive_F))
                        if fronts:
                            keep_idx = fronts[0][:archive_size]
                            if len(keep_idx) < archive_size:
                                for front in fronts[1:]:
                                    for idx in front:
                                        if idx not in keep_idx and len(keep_idx) < archive_size:
                                            keep_idx.append(idx)
                            archive_X = [archive_X[i] for i in keep_idx]
                            archive_F = [archive_F[i] for i in keep_idx]
            
            # Occasionally add random points for exploration
            if used % 50 == 0 and used < budget - 10:
                n_random = min(5, budget - used)
                random_X = lo + (hi - lo) * rng.random((n_random, dim))
                random_F = np.array([problem(x) for x in random_X])
                used += n_random
                
                for x, f in zip(random_X, random_F):
                    dominated = False
                    for af in archive_F:
                        if np.all(af <= f) and np.any(af < f):
                            dominated = True
                            break
                    if not dominated:
                        new_archive_X = []
                        new_archive_F = []
                        for ax, af in zip(archive_X, archive_F):
                            if not (np.all(f <= af) and np.any(f < af)):
                                new_archive_X.append(ax)
                                new_archive_F.append(af)
                        new_archive_X.append(x)
                        new_archive_F.append(f)
                        archive_X = new_archive_X
                        archive_F = new_archive_F
        
        # Return archive
        if len(archive_X) > 0:
            return np.array(archive_X)
        else:
            return np.empty((0, dim))
# EVOLVE-BLOCK-END
