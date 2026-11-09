"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """NSGA-II based multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.rng = np.random.default_rng(0)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        
        # Population size
        # 200 budget. 10 init + 19*10 offspring = 200.
        pop_size = 10
        if self.budget < 20:
             pop_size = max(2, self.budget // 2)
             # If budget is very small, just do random search
             if self.budget <= pop_size:
                 X = lo + (hi - lo) * self.rng.random((self.budget, n))
                 for x in X:
                     problem(x)
                 return X
             
        # Initial population
        X = lo + (hi - lo) * self.rng.random((pop_size, n))
        F = np.array([problem(x) for x in X])
        evals_used = pop_size
        
        # Helper functions for NSGA-II
        def fast_non_dominated_sort(F):
            N = len(F)
            S = [set() for _ in range(N)]
            n = [0] * N
            rank = [0] * N
            I0 = set()
            
            for p in range(N):
                for q in range(N):
                    if p == q: continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].add(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n[p] += 1
            for p in range(N):
                if n[p] == 0:
                    rank[p] = 0
                    I0.add(p)
                    
            i = 0
            frontiers = []
            while len(I0) > 0:
                frontiers.append(list(I0))
                I_next = set()
                for p in I0:
                    for q in S[p]:
                        n[q] -= 1
                        if n[q] == 0:
                            rank[q] = i + 1
                            I_next.add(q)
                I0 = I_next
                i += 1
            return frontiers, rank

        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return {i: float('inf') for i in indices}
            m = len(F[0])
            dist = {i: 0.0 for i in indices}
            for k in range(m):
                f_vals = np.array([F[i][k] for i in indices])
                f_min = np.min(f_vals)
                f_max = np.max(f_vals)
                if f_max == f_min:
                    continue
                sorted_indices = sorted(indices, key=lambda i: F[i][k])
                dist[sorted_indices[0]] = float('inf')
                dist[sorted_indices[-1]] = float('inf')
                for i in range(1, len(sorted_indices) - 1):
                    p = sorted_indices[i-1]
                    q = sorted_indices[i+1]
                    dist[sorted_indices[i]] += (F[q][k] - F[p][k]) / (f_max - f_min)
            return dist

        def tournament_select(pop_X, pop_F, ranks, cdists, k=2):
            best_idx = -1
            best_rank = float('inf')
            best_cd = -1
            for _ in range(k):
                idx = self.rng.integers(len(pop_X))
                if ranks[idx] < best_rank or (ranks[idx] == best_rank and cdists[idx] > best_cd):
                    best_idx = idx
                    best_rank = ranks[idx]
                    best_cd = cdists[idx]
            return pop_X[best_idx].copy()

        def sbx_crossover(x1, x2, eta=20.0):
            child1 = x1.copy()
            child2 = x2.copy()
            for i in range(n):
                if self.rng.random() < 0.5:
                    if abs(x1[i] - x2[i]) > 1e-14:
                        u = self.rng.random()
                        if u <= 0.5:
                            alpha = (2 * u) ** (1.0 / (eta + 1.0))
                        else:
                            alpha = (2 * (1 - u)) ** (1.0 / (eta + 1.0))
                        c1 = 0.5 * (1 + alpha) * x1[i] + 0.5 * (1 - alpha) * x2[i]
                        c2 = 0.5 * (1 - alpha) * x1[i] + 0.5 * (1 + alpha) * x2[i]
                        child1[i] = c1
                        child2[i] = c2
            return child1, child2

        def poly_mutate(x, eta=20.0):
            y = x.copy()
            for i in range(n):
                if self.rng.random() < 1.0 / n:
                    y[i] = x[i]
                    u = self.rng.random()
                    delta1 = (2 * u) ** (1.0 / (eta + 1.0)) - 1.0
                    delta2 = 1.0 - (2 * (1 - u)) ** (1.0 / (eta + 1.0))
                    yl = lo[i]
                    yh = hi[i]
                    y[i] += 2 * (y[i] - yl) * delta1 if u < 0.5 else 2 * (yh - y[i]) * delta2
                    y[i] = np.clip(y[i], lo[i], hi[i])
            return y

        while evals_used < self.budget:
            # Check time
            if time.time() > deadline - 5.0:
                break
                
            # Create offspring
            Q_X = []
            Q_F = []
            
            # We need to generate enough offspring to fill the next generation
            # Usually pop_size offspring per generation
            num_offspring = pop_size
            # But we might have less budget left
            num_offspring = min(num_offspring, self.budget - evals_used)
            
            if num_offspring == 0:
                break
                
            # Get ranks and crowding distances for current population
            frontiers, ranks = fast_non_dominated_sort(F)
            cdists = {}
            for front in frontiers:
                front_F = F[front]
                d = crowding_distance(front_F, front)
                for idx, dist in d.items():
                    cdists[idx] = dist
            
            for _ in range(num_offspring):
                p1 = tournament_select(X, F, ranks, cdists)
                p2 = tournament_select(X, F, ranks, cdists)
                c1, c2 = sbx_crossover(p1, p2)
                c1 = poly_mutate(c1)
                c2 = poly_mutate(c2)
                
                # Evaluate c1
                f1 = problem(c1)
                evals_used += 1
                Q_X.append(c1)
                Q_F.append(f1)
                
                # Check time again
                if time.time() > deadline - 5.0:
                    break
                    
                # Evaluate c2 if budget remains
                if evals_used < self.budget:
                    f2 = problem(c2)
                    evals_used += 1
                    Q_X.append(c2)
                    Q_F.append(f2)
                
                if len(Q_X) >= num_offspring:
                    break

            if not Q_X:
                break
                
            Q_F = np.array(Q_F)
            
            # Combine P and Q
            R_X = np.vstack([X, np.array(Q_X)])
            R_F = np.vstack([F, Q_F])
            
            # Non-dominated sort R
            frontiers, ranks = fast_non_dominated_sort(R_F)
            
            # Select next P
            new_P_X = []
            new_P_F = []
            new_ranks = [0] * len(R_X)
            
            for i, front in enumerate(frontiers):
                if len(new_P_X) + len(front) <= pop_size:
                    for idx in front:
                        new_P_X.append(R_X[idx])
                        new_P_F.append(R_F[idx])
                        new_ranks[idx] = i
                else:
                    # Need to select from this front using crowding distance
                    front_F = R_F[front]
                    d = crowding_distance(front_F, front)
                    sorted_front = sorted(front, key=lambda idx: d[idx], reverse=True)
                    needed = pop_size - len(new_P_X)
                    for idx in sorted_front[:needed]:
                        new_P_X.append(R_X[idx])
                        new_P_F.append(R_F[idx])
                        new_ranks[idx] = i
                    break
            
            if new_P_X:
                X = np.array(new_P_X)
                F = np.array(new_P_F)
            else:
                break

        return X
# EVOLVE-BLOCK-END
