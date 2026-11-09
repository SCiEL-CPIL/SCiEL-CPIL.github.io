"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer using NSGA-II with adaptive mutation and local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """Return list of fronts (list of indices)."""
        n = len(F)
        S = [[] for _ in range(n)]
        nd = np.zeros(n, dtype=int)
        
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                    S[p].append(q)
                elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                    nd[p] += 1
        
        front = [i for i in range(n) if nd[i] == 0]
        if not front:
            front = [0]
        
        fronts = [front]
        while front:
            next_front = []
            for p in front:
                for q in S[p]:
                    nd[q] -= 1
                    if nd[q] == 0:
                        next_front.append(q)
            if next_front:
                fronts.append(next_front)
            front = next_front
        return fronts

    def _crowding_distance(self, F, indices):
        """Calculate crowding distance for a front."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        dist = np.zeros(n)
        F_sub = F[indices]
        
        for m in range(self.n_obj):
            order = np.argsort(F_sub[:, m])
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            
            min_val = F_sub[order[0], m]
            max_val = F_sub[order[-1], m]
            if max_val == min_val:
                continue
                
            for k in range(1, n - 1):
                dist[order[k]] += (F_sub[order[k + 1], m] - F_sub[order[k - 1], m]) / (max_val - min_val)
        return dist

    def _sbx_crossover(self, X1, X2, lo, hi, eta=15.0):
        """Simulated Binary Crossover."""
        if np.random.random() > 0.9:
            return X1.copy(), X2.copy()
        
        c1 = np.zeros_like(X1)
        c2 = np.zeros_like(X1)
        
        for i in range(self.dim):
            u = np.random.random()
            if u <= 0.5:
                beta = (2 * u) ** (1.0 / (eta + 1))
            else:
                beta = (1 / (2 * (1 - u))) ** (1.0 / (eta + 1))
            
            c1[i] = 0.5 * ((1 + beta) * X1[i] + (1 - beta) * X2[i])
            c2[i] = 0.5 * ((1 - beta) * X1[i] + (1 + beta) * X2[i])
        
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        return c1, c2

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # Smaller population for more generations
        pop_size = 30
        total_evals = 0
        
        # Initial population: mix of random and grid
        X = lo + span * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        total_evals += pop_size
        
        generation = 0
        max_generations = (self.budget - total_evals) // pop_size + 1
        
        while total_evals + pop_size <= self.budget:
            if time.time() > deadline - 5.0:
                break
            
            generation += 1
            progress = generation / max(1, max_generations)
            
            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(F)
            rank = np.zeros(len(F), dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i
            
            # Crowding distance per front
            cd = np.zeros(len(F))
            for front in fronts:
                cd[front] = self._crowding_distance(F, front)
            
            # Selection: build mating pool
            mating_pool = []
            for front in fronts:
                if len(mating_pool) + len(front) <= pop_size:
                    mating_pool.extend(front)
                else:
                    need = pop_size - len(mating_pool)
                    cd_front = cd[front]
                    order = np.argsort(cd_front)[::-1]
                    mating_pool.extend([front[i] for i in order[:need]])
                    break
            
            # Generate offspring
            offspring = []
            # Adaptive mutation: starts at 0.3, shrinks to 0.05
            mut_rate = 0.3 * (1 - progress) + 0.05 * progress
            
            while len(offspring) < pop_size:
                # Binary tournament
                t1, t2 = np.random.randint(0, len(F), 2)
                if rank[t1] < rank[t2]:
                    p1 = t1
                elif rank[t2] < rank[t1]:
                    p1 = t2
                else:
                    p1 = t1 if cd[t1] > cd[t2] else t2
                
                t1, t2 = np.random.randint(0, len(F), 2)
                if rank[t1] < rank[t2]:
                    p2 = t1
                elif rank[t2] < rank[t1]:
                    p2 = t2
                else:
                    p2 = t1 if cd[t1] > cd[t2] else t2
                
                c1, c2 = self._sbx_crossover(X[p1], X[p2], lo, hi)
                
                # Polynomial mutation
                for c in [c1, c2]:
                    for i in range(self.dim):
                        if np.random.random() < 0.9:
                            u = np.random.random()
                            if u < 0.5:
                                delta = (2 * u) ** (1/20) - 1
                            else:
                                delta = 1 - (2 * (1 - u)) ** (1/20)
                            c[i] += mut_rate * span[i] * delta
                            c[i] = np.clip(c[i], lo[i], hi[i])
                
                offspring.append(c1)
                offspring.append(c2)
            
            offspring = np.array(offspring[:pop_size])
            F_off = np.array([problem(x) for x in offspring])
            total_evals += pop_size
            
            # Combine and select
            X = np.vstack([X, offspring])
            F = np.vstack([F, F_off])
            
            fronts = self._fast_non_dominated_sort(F)
            rank = np.zeros(len(F), dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i
            cd = np.zeros(len(F))
            for front in fronts:
                cd[front] = self._crowding_distance(F, front)
            
            next_gen = []
            for front in fronts:
                if len(next_gen) + len(front) <= pop_size:
                    next_gen.extend(front)
                else:
                    need = pop_size - len(next_gen)
                    cd_front = cd[front]
                    order = np.argsort(cd_front)[::-1]
                    next_gen.extend([front[i] for i in order[:need]])
                    break
            
            X = X[next_gen]
            F = F[next_gen]
        
        # Final local polish: use remaining budget to refine Pareto front
        remaining = self.budget - total_evals
        if remaining > 10 and time.time() < deadline - 30.0:
            fronts = self._fast_non_dominated_sort(F)
            pareto_front_idx = fronts[0]
            
            # Use remaining evaluations to locally optimize each Pareto point
            per_point = max(1, remaining // max(1, len(pareto_front_idx)))
            polished_X = []
            polished_F = []
            
            for idx in pareto_front_idx:
                if time.time() > deadline - 10.0:
                    break
                
                x0 = X[idx].copy()
                f0 = F[idx].copy()
                
                # Try small perturbations in each direction
                best_x = x0.copy()
                best_f = f0.copy()
                evals_used = 0
                
                for _ in range(min(per_point, remaining)):
                    if time.time() > deadline - 10.0:
                        break
                    
                    # Random small step
                    step_size = 0.05 * span
                    perturbation = np.random.standard_normal(self.dim) * step_size
                    x_new = np.clip(x0 + perturbation, lo, hi)
                    
                    f_new = np.array(problem(x_new))
                    evals_used += 1
                    remaining -= 1
                    
                    # Accept if it dominates or is non-dominated with better crowding
                    if np.all(f_new <= best_f) and np.any(f_new < best_f):
                        best_x = x_new
                        best_f = f_new
                    elif not np.any(np.all(best_f <= f_new) and np.any(best_f < f_new)):
                        # Non-dominated, keep if randomly better
                        if np.random.random() < 0.3:
                            best_x = x_new
                            best_f = f_new
                
                if evals_used > 0:
                    polished_X.append(best_x)
                    polished_F.append(best_f)
            
            if polished_X:
                X = np.array(polished_X)
                F = np.array(polished_F)
        
        return X
# EVOLVE-BLOCK-END
