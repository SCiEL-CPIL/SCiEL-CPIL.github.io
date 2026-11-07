"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using an evolutionary strategy with vectorized operations."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Initialize population with larger size for better coverage
        pop_size = min(30, self.budget // 3)
        X = lo + range_ * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track all evaluated points for hypervolume
        all_X = list(X)
        all_F = list(F)
        
        def vectorized_non_dominated(F):
            """Return boolean mask of non-dominated solutions using vectorized ops."""
            n = len(F)
            if n == 0:
                return np.array([], dtype=bool)
            leq = (F[:, np.newaxis, :] <= F[np.newaxis, :, :])  # (n, n, m)
            strict = (F[:, np.newaxis, :] < F[np.newaxis, :, :])  # (n, n, m)
            dom_mask = np.all(leq, axis=2) & np.any(strict, axis=2)
            dominated = np.any(dom_mask, axis=0)
            return ~dominated
        
        def crowding_distance(F, nd_indices):
            """Compute crowding distance for non-dominated solutions."""
            n_nd = len(nd_indices)
            if n_nd <= 2:
                return np.full(n_nd, np.inf)
            
            cd = np.zeros(n_nd)
            F_nd = F[nd_indices]
            
            for m in range(self.n_obj):
                order = np.argsort(F_nd[:, m])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                denom = F_nd[order[-1], m] - F_nd[order[0], m]
                if denom == 0:
                    continue
                for k in range(1, n_nd - 1):
                    cd[order[k]] += (F_nd[order[k+1], m] - F_nd[order[k-1], m]) / denom
            
            return cd
        
        def select_parent(F, X, rng):
            """Tournament selection with non-domination and crowding distance."""
            nd_mask = vectorized_non_dominated(F)
            nd_indices = np.where(nd_mask)[0]
            
            if len(nd_indices) <= 1:
                return X[nd_indices[0]] if len(nd_indices) == 1 else X[rng.integers(len(F))]
            
            cd = crowding_distance(F, nd_indices)
            
            # Tournament size 3 for stronger selection pressure
            i1, i2, i3 = rng.integers(len(nd_indices), size=3)
            best = i1
            if cd[i2] > cd[best]:
                best = i2
            if cd[i3] > cd[best]:
                best = i3
            return X[nd_indices[best]]
        
        # Adaptive step size: start large, decrease over time
        base_step = 0.20
        min_step = 0.005
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            progress = used / self.budget
            step_size = base_step * (1.0 - progress) + min_step * progress
            
            # Generate multiple children per iteration
            n_children = min(5, self.budget - used)
            
            for _ in range(n_children):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                parent = select_parent(F, X, rng)
                
                # Generate child with adaptive step size
                child = parent + step_size * range_ * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                used += 1
                
                all_X.append(child)
                all_F.append(fc)
                
                # Replace worst solution in population using dominance
                worst_idx = None
                for i in range(len(F)):
                    if np.all(fc <= F[i]) and np.any(fc < F[i]):
                        worst_idx = i
                        break
                if worst_idx is None:
                    worst_idx = int(np.argmax(np.sum(F, axis=1)))
                
                X[worst_idx] = child
                F[worst_idx] = fc
            
            # Periodically prune to maintain quality
            if used % 15 == 0:
                nd_mask = vectorized_non_dominated(F)
                n_nd = np.sum(nd_mask)
                if n_nd < len(F) // 2:
                    nd_indices = np.where(nd_mask)[0]
                    dom_indices = np.where(~nd_mask)[0]
                    if len(dom_indices) > 0:
                        keep_dom = rng.choice(dom_indices, size=min(5, len(dom_indices)), replace=False)
                    else:
                        keep_dom = np.array([], dtype=int)
                    keep = np.concatenate([nd_indices, keep_dom])
                    keep = np.unique(keep)
                    if len(keep) < len(F):
                        X = X[keep]
                        F = F[keep]
        
        # Final polish: use remaining budget for small-step refinement
        remaining = self.budget - used
        if remaining > 0 and time.time() < deadline - 5:
            nd_mask = vectorized_non_dominated(F)
            nd_indices = np.where(nd_mask)[0]
            if len(nd_indices) > 0:
                n_polish = min(remaining, 15)
                polish_points = rng.choice(nd_indices, size=min(n_polish, len(nd_indices)), replace=False)
                for pi in polish_points:
                    if used >= self.budget or time.time() > deadline:
                        break
                    center = X[pi]
                    for _ in range(min(4, self.budget - used)):
                        if used >= self.budget or time.time() > deadline:
                            break
                        child = center + 0.015 * range_ * rng.standard_normal(self.dim)
                        child = np.clip(child, lo, hi)
                        fc = problem(child)
                        used += 1
                        all_X.append(child)
                        all_F.append(fc)
        
        return all_X
# EVOLVE-BLOCK-END
