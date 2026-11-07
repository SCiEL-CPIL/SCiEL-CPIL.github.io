import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _latin_hypercube(self, N, dim, lo, hi):
        """Generate Latin Hypercube sample points"""
        points = np.zeros((N, dim))
        for d in range(dim):
            # Create N equal intervals and pick a random point in each
            intervals = np.linspace(0, 1, N + 1)
            for i in range(N):
                points[i, d] = lo[d] + (hi[d] - lo[d]) * (intervals[i] + np.random.random() * (intervals[i+1] - intervals[i]))
        # Shuffle each dimension independently
        for d in range(dim):
            perm = np.random.permutation(N)
            points[:, d] = points[perm, d]
        return points

    def _crowding_distance(self, F):
        """Compute crowding distance for each point in objective space"""
        n = len(F)
        if n <= 2:
            return np.full(n, np.inf)
        
        dist = np.zeros(n)
        for m in range(F.shape[1]):
            # Sort by objective m
            idx = np.argsort(F[:, m])
            # Boundary points get infinite distance
            dist[idx[0]] = np.inf
            dist[idx[-1]] = np.inf
            # Compute distances for interior points
            for i in range(1, n-1):
                dist[idx[i]] += (F[idx[i+1], m] - F[idx[i-1], m]) / (F[idx[-1], m] - F[idx[0], m] + 1e-30)
        return dist

    def _find_non_dominated(self, F):
        """Find indices of non-dominated points in objective matrix F"""
        n = len(F)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        return np.where(~dominated)[0]

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Population size
        N = 20
        if self.budget < N:
            N = self.budget
            
        # Initialize population with Latin Hypercube sampling
        X = self._latin_hypercube(N, dim, lo, hi)
        F = np.array([problem(x) for x in X])
        used = N
        
        # Parameters
        F_mut = 0.8
        CR = 0.5
        
        # Reserve budget for local polish
        polish_budget = max(20, self.budget // 5)
        evo_budget = self.budget - polish_budget
        
        while used < evo_budget:
            if time.time() > deadline:
                break
                
            # Select a random individual i
            i = np.random.randint(N)
            
            # Select three distinct random indices r1, r2, r3 != i
            candidates = [j for j in range(N) if j != i]
            r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)
            
            # Mutation
            v = X[r1] + F_mut * (X[r2] - X[r3])
            
            # Crossover
            j_rand = np.random.randint(dim)
            trial = np.empty(dim)
            for j in range(dim):
                if np.random.random() <= CR or j == j_rand:
                    trial[j] = v[j]
                else:
                    trial[j] = X[i][j]
            
            # Clip to bounds
            trial = np.clip(trial, lo, hi)
            
            # Evaluate
            f_trial = problem(trial)
            used += 1
            
            # Selection using crowding distance for diversity
            if np.all(f_trial <= F[i]) and np.any(f_trial < F[i]):
                # Trial dominates i, replace
                X[i] = trial
                F[i] = f_trial
            elif not (np.all(F[i] <= f_trial) and np.any(F[i] < f_trial)):
                # Non-dominated: use crowding distance to decide
                cd = self._crowding_distance(F)
                mean_F = np.mean(F, axis=0)
                dist_trial = np.linalg.norm(f_trial - mean_F)
                if dist_trial > cd[i]:
                    X[i] = trial
                    F[i] = f_trial
        
        # Local Nelder-Mead polish on top Pareto points
        if time.time() < deadline and used < self.budget:
            # Find non-dominated points
            nd_idx = self._find_non_dominated(F)
            
            # Sort by hypervolume contribution (proxy: sum of objectives, lower is better)
            # For 2 objectives, we want to polish points that are "good" in both objectives
            # Use a weighted sum as a proxy for quality
            weights = np.array([1.0, 1.0])
            scores = F[nd_idx] @ weights
            # Take top few non-dominated points to polish
            n_polish = min(5, len(nd_idx), max(1, (self.budget - used) // 10))
            top_idx = nd_idx[np.argsort(scores)[:n_polish]]
            
            for idx in top_idx:
                if time.time() > deadline or used >= self.budget:
                    break
                
                # Define scalar objective for Nelder-Mead (minimize weighted sum)
                def obj_func(x, _lo=lo, _hi=hi):
                    x_clipped = np.clip(x, _lo, _hi)
                    f = problem(x_clipped)
                    used_ref[0] += 1
                    return np.sum(f * weights)
                
                used_ref = [0]
                
                # Nelder-Mead with tight tolerance
                try:
                    result = minimize(
                        obj_func, 
                        X[idx], 
                        method='Nelder-Mead',
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8}
                    )
                    
                    if result.success and used_ref[0] > 0:
                        x_polished = np.clip(result.x, lo, hi)
                        f_polished = problem(x_polished)
                        used += used_ref[0]
                        
                        # Check if polished point is better
                        if np.all(f_polished <= F[idx]) and np.any(f_polished < F[idx]):
                            X[idx] = x_polished
                            F[idx] = f_polished
                        elif not (np.all(F[idx] <= f_polished) and np.any(F[idx] < f_polished)):
                            # Non-dominated: keep if it improves hypervolume proxy
                            if np.sum(f_polished * weights) < np.sum(F[idx] * weights):
                                X[idx] = x_polished
                                F[idx] = f_polished
                except Exception:
                    pass
        
        return X
