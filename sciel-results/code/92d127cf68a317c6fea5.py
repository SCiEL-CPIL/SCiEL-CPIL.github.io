# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for RE34 (3 objectives, 5 variables, bounds [1,3]).
    
    Strategy:
    1. Latin Hypercube Sampling for initial spread.
    2. NSGA-II style evolutionary search with crowding distance selection.
    3. Adaptive mutation and crossover.
    4. Tight time guard to ensure completion within 330s.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.elapsed_start = None
        self.time_limit = 325.0  # safety margin from 330s kill

    def _elapsed(self):
        if self.elapsed_start is None:
            return 0.0
        return time.time() - self.elapsed_start

    def _dominates(self, a, b):
        """Returns True if a dominates b (all <=, at least one <)"""
        return np.all(a <= b) and np.any(a < b)

    def _fast_non_dominated_sort(self, F):
        """
        Fast non-dominated sort.
        Returns list of fronts, where each front is a list of indices.
        """
        n = len(F)
        S = [[] for _ in range(n)]  # S[i] = indices dominated by i
        n_count = [0] * n  # n[i] = number of solutions dominating i
        fronts = [[]]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(F[i], F[j]):
                    S[i].append(j)
                elif self._dominates(F[j], F[i]):
                    n_count[i] += 1
            if n_count[i] == 0:
                fronts[0].append(i)
        
        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in S[i]:
                    n_count[j] -= 1
                    if n_count[j] == 0:
                        next_front.append(j)
            k += 1
            fronts.append(next_front)
        
        return fronts[:-1] if fronts[-1] else fronts

    def _crowding_distance(self, F, indices):
        """
        Compute crowding distance for solutions in indices.
        Returns array of crowding distances.
        """
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        dist = np.zeros(n)
        F_sub = F[indices]
        
        for m in range(self.n_obj):
            sorted_idx = np.argsort(F_sub[:, m])
            dist[sorted_idx[0]] = np.inf
            dist[sorted_idx[-1]] = np.inf
            f_range = F_sub[sorted_idx[-1], m] - F_sub[sorted_idx[0], m]
            if f_range > 0:
                for i in range(1, n - 1):
                    dist[sorted_idx[i]] += (F_sub[sorted_idx[i+1], m] - F_sub[sorted_idx[i-1], m]) / f_range
        
        return dist

    def __call__(self, problem, bounds):
        self.elapsed_start = time.time()
        rng = np.random.default_rng(0)
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Deadline check
        if self._elapsed() > self.time_limit:
            return np.empty((0, dim))
        
        # Phase 1: Latin Hypercube Sampling for initial spread
        n_lhs = min(budget, max(50, dim * 10))
        X_lhs = np.empty((n_lhs, dim))
        for j in range(dim):
            perm = rng.permutation(n_lhs)
            X_lhs[:, j] = (perm + rng.random(n_lhs)) / n_lhs
        X_lhs = lo + (hi - lo) * X_lhs
        
        # Evaluate initial population
        F_lhs = np.array([problem(x) for x in X_lhs])
        used = n_lhs
        
        # Maintain archive of non-dominated solutions
        archive_X = []
        archive_F = []
        for x, f in zip(X_lhs, F_lhs):
            dominated = False
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    dominated = True
                    break
            if not dominated:
                # Remove dominated from archive
                new_X = []
                new_F = []
                for ax, af in zip(archive_X, archive_F):
                    if not (np.all(f <= af) and np.any(f < af)):
                        new_X.append(ax)
                        new_F.append(af)
                new_X.append(x)
                new_F.append(f)
                archive_X = new_X
                archive_F = new_F
        
        # Population for NSGA-II
        pop_size = min(100, max(20, budget // 4))
        
        # Select initial population from archive + random
        if len(archive_X) >= pop_size:
            idx = rng.choice(len(archive_X), size=pop_size, replace=False)
            pop_X = np.array([archive_X[i] for i in idx])
            pop_F = np.array([archive_F[i] for i in idx])
        else:
            pop_X = np.array(archive_X) if archive_X else X_lhs[:pop_size]
            pop_F = np.array(archive_F) if archive_F else F_lhs[:pop_size]
            # Pad with random
            while len(pop_X) < pop_size and used < budget:
                x_new = lo + (hi - lo) * rng.random(dim)
                f_new = problem(x_new)
                used += 1
                pop_X = np.vstack([pop_X, [x_new]]) if len(pop_X) > 0 else np.array([x_new])
                pop_F = np.vstack([pop_F, [f_new]]) if len(pop_F) > 0 else np.array([f_new])
                # Update archive
                dominated = False
                for af in archive_F:
                    if np.all(af <= f_new) and np.any(af < f_new):
                        dominated = True
                        break
                if not dominated:
                    new_X = []
                    new_F = []
                    for ax, af in zip(archive_X, archive_F):
                        if not (np.all(f_new <= af) and np.any(f_new < af)):
                            new_X.append(ax)
                            new_F.append(af)
                    new_X.append(x_new)
                    new_F.append(f_new)
                    archive_X = new_X
                    archive_F = new_F
        
        # Adaptive mutation scale
        mut_scale = 0.2
        
        while used < budget:
            # Time check
            if self._elapsed() > self.time_limit - 5:
                break
            
            # Generate offspring
            offspring_X = []
            offspring_F = []
            
            n_offspring = min(pop_size, budget - used)
            
            for i in range(n_offspring):
                if used >= budget:
                    break
                if self._elapsed() > self.time_limit - 2:
                    break
                
                # Selection: tournament selection from population
                if len(pop_X) >= 2:
                    t1 = rng.integers(len(pop_X))
                    t2 = rng.integers(len(pop_X))
                    # Simple tournament: pick the one with better sum of objectives
                    if np.sum(pop_F[t1]) <= np.sum(pop_F[t2]):
                        parent1 = pop_X[t1]
                    else:
                        parent1 = pop_X[t2]
                    
                    t3 = rng.integers(len(pop_X))
                    t4 = rng.integers(len(pop_X))
                    if np.sum(pop_F[t3]) <= np.sum(pop_F[t4]):
                        parent2 = pop_X[t3]
                    else:
                        parent2 = pop_X[t4]
                else:
                    parent1 = pop_X[0]
                    parent2 = pop_X[0]
                
                # Crossover
                if rng.random() < 0.8:
                    # SBX crossover
                    beta = 20.0
                    child = np.zeros(dim)
                    for j in range(dim):
                        if rng.random() < 0.5:
                            if abs(parent1[j] - parent2[j]) > 1e-10:
                                u = rng.random()
                                if u <= 1.0 / (beta + 1):
                                    beta_q = ((2 * u) ** (1.0 / (beta + 1)))
                                else:
                                    beta_q = ((1.0 / (2 * (1.0 - u))) ** (1.0 / (beta + 1)))
                                child[j] = 0.5 * ((1 + beta_q) * parent1[j] + (1 - beta_q) * parent2[j])
                            else:
                                child[j] = parent1[j]
                        else:
                            if abs(parent1[j] - parent2[j]) > 1e-10:
                                u = rng.random()
                                if u <= 1.0 / (beta + 1):
                                    beta_q = ((2 * u) ** (1.0 / (beta + 1)))
                                else:
                                    beta_q = ((1.0 / (2 * (1.0 - u))) ** (1.0 / (beta + 1)))
                                child[j] = 0.5 * ((1 - beta_q) * parent1[j] + (1 + beta_q) * parent2[j])
                            else:
                                child[j] = parent1[j]
                else:
                    child = parent1.copy()
                
                # Mutation
                for j in range(dim):
                    if rng.random() < 0.8:  # mutation probability
                        u = rng.random()
                        if u <= 1.0 / (beta + 1):
                            delta_q = ((2 * u) ** (1.0 / (beta + 1)) - 1.0)
                        else:
                            delta_q = (1.0 - ((2.0 * (1.0 - u)) ** (1.0 / (beta + 1))))
                        child[j] += mut_scale * (hi[j] - lo[j]) * delta_q
                
                child = np.clip(child, lo, hi)
                
                # Evaluate
                f_child = problem(child)
                used += 1
                offspring_X.append(child)
                offspring_F.append(f_child)
                
                # Update archive
                dominated = False
                for af in archive_F:
                    if np.all(af <= f_child) and np.any(af < f_child):
                        dominated = True
                        break
                if not dominated:
                    new_X = []
                    new_F = []
                    for ax, af in zip(archive_X, archive_F):
                        if not (np.all(f_child <= af) and np.any(f_child < af)):
                            new_X.append(ax)
                            new_F.append(af)
                    new_X.append(child)
                    new_F.append(f_child)
                    archive_X = new_X
                    archive_F = new_F
            
            if not offspring_X:
                break
            
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # Combine parent and offspring
            combined_X = np.vstack([pop_X, offspring_X])
            combined_F = np.vstack([pop_F, offspring_F])
            
            # Non-dominated sort
            fronts = self._fast_non_dominated_sort(combined_F)
            
            # Select next population
            new_pop_X = []
            new_pop_F = []
            for front in fronts:
                if len(new_pop_X) + len(front) <= pop_size:
                    for idx in front:
                        new_pop_X.append(combined_X[idx])
                        new_pop_F.append(combined_F[idx])
                else:
                    # Need to select from this front using crowding distance
                    remaining = pop_size - len(new_pop_X)
                    front_F = combined_F[front]
                    front_indices = list(front)
                    cd = self._crowding_distance(combined_F, front_indices)
                    sorted_idx = np.argsort(-cd)
                    for i in range(min(remaining, len(sorted_idx))):
                        idx = front_indices[sorted_idx[i]]
                        new_pop_X.append(combined_X[idx])
                        new_pop_F.append(combined_F[idx])
                    break
            
            if new_pop_X:
                pop_X = np.array(new_pop_X)
                pop_F = np.array(new_pop_F)
            
            # Adapt mutation scale
            mut_scale = max(0.05, mut_scale * 0.99)
        
        # Return best set of points (archive)
        if len(archive_X) > 0:
            return np.array(archive_X)
        else:
            return np.empty((0, dim))
# EVOLVE-BLOCK-END
