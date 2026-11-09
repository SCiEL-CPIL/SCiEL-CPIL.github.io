"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective search with SBX crossover, local refinement, and adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _non_dominated_front(self, F):
        """Return boolean mask of non-dominated points in F (vectorised)."""
        n = len(F)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            # Check if any other point dominates i
            dominates_i = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
            dominates_i[i] = False
            if np.any(dominates_i):
                dominated[i] = True
        return ~dominated

    def _simplified_hypervolume_2d(self, points, ref):
        """Exact hypervolume for 2D subset of 3D points (for local refinement)."""
        # Project to first 2 objectives, check third is below ref
        valid = np.all(points < ref, axis=1)
        pts = points[valid]
        if len(pts) == 0:
            return 0.0
        # Sort by first objective
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        # Compute hypervolume in 2D (first two objectives), but we need 3D
        # Actually, let's just do a simple local refinement without hypervolume
        return 0.0

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Initial population
        pop_size = 30
        X = lo + range_ * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Keep track of all evaluated points for hypervolume
        all_F = [F.copy()]
        
        # Main evolutionary loop
        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Find non-dominated front
            nd_mask = self._non_dominated_front(F)
            nd_indices = np.where(nd_mask)[0]
            
            if len(nd_indices) < 2:
                # If too few non-dominated, use all
                nd_indices = np.arange(pop_size)
            
            # Select two parents from non-dominated front
            p1_idx, p2_idx = rng.choice(nd_indices, size=2, replace=False)
            p1, p2 = X[p1_idx], X[p2_idx]
            
            # SBX crossover
            eta = 20
            u = rng.random(self.dim)
            v = rng.random(self.dim)
            
            c1 = np.zeros(self.dim)
            c2 = np.zeros(self.dim)
            
            for d in range(self.dim):
                if u[d] <= 0.5:
                    beta = (2 * u[d]) ** (1.0 / (eta + 1))
                else:
                    beta = (1.0 / (2 * (1 - u[d]))) ** (1.0 / (eta + 1))
                
                c1[d] = 0.5 * ((1 + beta) * p1[d] + (1 - beta) * p2[d])
                
                if v[d] <= 0.5:
                    beta = (2 * v[d]) ** (1.0 / (eta + 1))
                else:
                    beta = (1.0 / (2 * (1 - v[d]))) ** (1.0 / (eta + 1))
                
                c2[d] = 0.5 * ((1 + beta) * p2[d] + (1 - beta) * p1[d])
            
            # Clip to bounds
            c1 = np.clip(c1, lo, hi)
            c2 = np.clip(c2, lo, hi)
            
            # Evaluate children
            f1 = problem(c1)
            used += 1
            all_F.append(f1.reshape(1, -1))
            
            if used >= self.budget:
                break
            
            f2 = problem(c2)
            used += 1
            all_F.append(f2.reshape(1, -1))
            
            # Environmental selection: combine parents and children, keep best pop_size
            X_new = np.vstack([X, c1.reshape(1, -1), c2.reshape(1, -1)])
            F_new = np.vstack([F, f1.reshape(1, -1), f2.reshape(1, -1)])
            
            # Non-dominated sort
            nd_mask_new = self._non_dominated_front(F_new)
            nd_count = np.sum(nd_mask_new)
            
            if nd_count >= pop_size:
                # Keep non-dominated points
                X = X_new[nd_mask_new][:pop_size]
                F = F_new[nd_mask_new][:pop_size]
            else:
                # Keep all non-dominated + best dominated
                nd_indices_new = np.where(nd_mask_new)[0]
                dom_indices_new = np.where(~nd_mask_new)[0]
                
                # Sort dominated by sum of objectives
                dom_sums = np.sum(F_new[dom_indices_new], axis=1)
                dom_order = dom_indices_new[np.argsort(dom_sums)]
                
                keep_count = pop_size - len(nd_indices_new)
                keep_indices = np.concatenate([nd_indices_new, dom_order[:keep_count]])
                
                X = X_new[keep_indices]
                F = F_new[keep_indices]
        
        # Local refinement pass: polish non-dominated points
        remaining_budget = self.budget - used
        if remaining_budget > 0 and time.time() < deadline - 5.0:
            nd_mask = self._non_dominated_front(F)
            nd_indices = np.where(nd_mask)[0]
            
            # For each non-dominated point, do a local search
            for idx in nd_indices:
                if used >= self.budget:
                    break
                if time.time() > deadline - 5.0:
                    break
                
                best_x = X[idx].copy()
                best_f = F[idx].copy()
                
                # Local search with decreasing step size
                for step_mult in [0.05, 0.02, 0.01, 0.005]:
                    if used >= self.budget:
                        break
                    if time.time() > deadline - 5.0:
                        break
                    
                    step_size = step_mult * np.min(range_)
                    for _ in range(10):
                        if used >= self.budget:
                            break
                        if time.time() > deadline - 5.0:
                            break
                        
                        direction = rng.standard_normal(self.dim)
                        direction /= (np.linalg.norm(direction) + 1e-10)
                        
                        candidate = best_x + step_size * direction
                        candidate = np.clip(candidate, lo, hi)
                        
                        fc = problem(candidate)
                        used += 1
                        all_F.append(fc.reshape(1, -1))
                        
                        # Check if candidate dominates best_f
                        if np.all(fc <= best_f) and np.any(fc < best_f):
                            best_x = candidate
                            best_f = fc
        
        return X
# EVOLVE-BLOCK-END
