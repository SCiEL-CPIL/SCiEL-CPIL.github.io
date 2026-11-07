"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using NSGA-II style steady-state evolution."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = None

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        deadline = self.start_time + 280  # 300s limit, keep 20s margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Parameters
        pop_size = 40
        F = 0.5  # Mutation factor
        CR = 0.5  # Crossover rate
        
        # Helper: Dominance check
        # Returns True if a dominates b (a is strictly better in at least one, and no worse in others)
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        # Helper: Crowding distance
        def crowding_distance(F):
            n = len(F)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(n_obj):
                # Sort by objective m
                idx = np.argsort(F[:, m])
                cd[idx[0]] = np.inf
                cd[idx[-1]] = np.inf
                if F[idx[-1], m] - F[idx[0], m] == 0:
                    continue
                for i in range(1, n - 1):
                    cd[idx[i]] += (F[idx[i+1], m] - F[idx[i-1], m]) / (F[idx[-1], m] - F[idx[0], m])
            return cd

        # Helper: Non-dominated sorting
        def non_dominated_sort(F):
            n = len(F)
            S = [set() for _ in range(n)]  # S[i] = set of points dominated by i
            n_d = np.zeros(n, dtype=int)   # n_d[i] = count of points dominating i
            fronts = [[]]
            
            for i in range(n):
                for j in range(i + 1, n):
                    if dominates(F[i], F[j]):
                        S[i].add(j)
                        n_d[j] += 1
                    elif dominates(F[j], F[i]):
                        S[j].add(i)
                        n_d[i] += 1
            
            for i in range(n):
                if n_d[i] == 0:
                    fronts[0].append(i)
            
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_d[j] -= 1
                        if n_d[j] == 0:
                            next_front.append(j)
                k += 1
                fronts.append(next_front)
            fronts.pop()  # Remove last empty front
            return fronts

        # Initialization: Latin Hypercube Sampling
        rng = np.random.default_rng(42)
        X = np.zeros((pop_size, dim))
        for j in range(dim):
            # LHS: divide [0,1] into pop_size intervals, pick one random point in each
            perm = rng.permutation(pop_size)
            X[:, j] = (perm + rng.random(pop_size)) / pop_size
        # Scale to bounds
        X = lo + X * (hi - lo)
        
        F = np.array([problem(x) for x in X])
        used = pop_size

        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Select target parent
            p_idx = rng.integers(pop_size)
            x_p = X[p_idx].copy()
            
            # Select 3 distinct parents for DE mutation
            # Ensure distinctness
            while True:
                a, b, c = rng.integers(pop_size, size=3)
                if a != b and b != c and a != c and a != p_idx and b != p_idx and c != p_idx:
                    break
            
            x_a, x_b, x_c = X[a], X[b], X[c]
            
            # DE Mutation
            v = x_a + F * (x_b - x_c)
            
            # Crossover
            u = x_p.copy()
            j_rand = rng.integers(dim)
            for j in range(dim):
                if rng.random() < CR or j == j_rand:
                    u[j] = v[j]
            
            # Clip to bounds
            u = np.clip(u, lo, hi)
            
            # Evaluate offspring
            f_u = problem(u)
            used += 1
            
            # Update Population using NSGA-II selection
            # Add u to population
            X_new = np.vstack([X, u.reshape(1, -1)])
            F_new = np.vstack([F, f_u.reshape(1, -1)])
            
            # Sort into fronts
            fronts = non_dominated_sort(F_new)
            
            # Select best pop_size individuals
            X_sel = []
            F_sel = []
            for front in fronts:
                if len(X_sel) + len(front) <= pop_size:
                    # Take all
                    for idx in front:
                        X_sel.append(X_new[idx])
                        F_sel.append(F_new[idx])
                else:
                    # Take only some, based on crowding distance
                    need = pop_size - len(X_sel)
                    # Calculate crowding distance for this front
                    front_F = F_new[front]
                    cd = crowding_distance(front_F)
                    # Sort by cd descending
                    sorted_indices = np.argsort(-cd)
                    for i in range(need):
                        idx = front[sorted_indices[i]]
                        X_sel.append(X_new[idx])
                        F_sel.append(F_new[idx])
                    break
            
            X = np.array(X_sel)
            F = np.array(F_sel)

        return X
# EVOLVE-BLOCK-END
