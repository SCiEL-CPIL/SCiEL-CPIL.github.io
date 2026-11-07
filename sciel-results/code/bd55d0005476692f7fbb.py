# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-objective Optimizer using Adaptive Sampling with Multi-Weight Nelder-Mead refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 280

    def _check_time(self):
        return time.time() > self.deadline

    def _evaluate(self, x, problem):
        if self.eval_count >= self.budget:
            return None
        self.eval_count += 1
        try:
            return problem(x)
        except Exception:
            return None

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        X_list = []
        F_list = []
        
        def add_point(x, f):
            if f is not None and len(f) == self.n_obj and np.all(np.isfinite(f)):
                x_clipped = np.clip(x, lo, hi)
                for i in range(len(X_list)):
                    if np.allclose(X_list[i], x_clipped, atol=1e-8):
                        return
                X_list.append(x_clipped)
                F_list.append(f)
        
        # Phase 1: Grid-based systematic coverage
        # Partition the 4D domain into a coarse grid and evaluate all vertices
        # For dim=4 with 3 divisions per axis: 3^4 = 81 grid vertices
        # This gives systematic coverage without stochastic variance
        
        grid_divs = 3  # 3 divisions per axis -> 4 points per axis
        n_grid_points = grid_divs ** self.dim  # 3^4 = 81 for dim=4
        
        # Generate grid vertices
        grid_points = []
        for i in range(grid_divs ** self.dim):
            p = []
            for d in range(self.dim):
                # Convert index to base-grid_divs digit for dimension d
                digit = (i // (grid_divs ** (self.dim - 1 - d))) % grid_divs
                # Map digit to position in [lo, hi]
                frac = digit / (grid_divs - 1) if grid_divs > 1 else 0.5
                p.append(lo[d] + frac * (hi[d] - lo[d]))
            grid_points.append(p)
        
        # Add center and corners for extra coverage
        corners = []
        for i in range(2**self.dim):
            c = []
            for d in range(self.dim):
                c.append(hi[d] if (i >> d) & 1 else lo[d])
            corners.append(c)
        
        center = [(lo[d] + hi[d])/2 for d in range(self.dim)]
        
        # Initial points: grid vertices + corners + center
        init_points = grid_points + corners + [center]
        
        # Evaluate initial points
        for x in init_points:
            if self.eval_count >= self.budget or self._check_time():
                break
            f = self._evaluate(x, problem)
            if f is not None:
                add_point(x, f)
        
        if not F_list:
            x0 = (lo + hi) / 2
            f0 = self._evaluate(x0, problem)
            if f0 is not None: add_point(x0, f0)
            return X_list if X_list else [x0]
        
        F = np.array(F_list)
        X = np.array(X_list)
        
        # Phase 2: Identify promising grid cells and refine with Nelder-Mead
        # Weight vectors to cover the Pareto front uniformly
        weight_vectors = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [1.0/3, 1.0/3, 1.0/3],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.7, 0.1, 0.2],
            [0.1, 0.7, 0.2],
            [0.2, 0.1, 0.7],
            [0.6, 0.3, 0.1],
            [0.3, 0.6, 0.1],
            [0.1, 0.3, 0.6],
            [0.4, 0.4, 0.2],
            [0.4, 0.2, 0.4],
            [0.2, 0.4, 0.4],
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
            [0.5, 0.3, 0.2],
            [0.3, 0.5, 0.2],
            [0.2, 0.3, 0.5],
        ]
        
        # Find non-dominated points
        def get_non_dominated(F):
            n = len(F)
            if n == 0:
                return np.array([], dtype=int)
            is_dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if is_dominated[i]: continue
                for j in range(i+1, n):
                    if is_dominated[j]: continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        is_dominated[j] = True
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_dominated[i] = True
                        break
            return np.where(~is_dominated)[0]
        
        # Phase 2a: Refine using weight vectors from non-dominated starting points
        n_refine_rounds = 4
        evals_per_run = 20
        
        for r in range(n_refine_rounds):
            if self.eval_count >= self.budget or self._check_time():
                break
            
            F = np.array(F_list)
            X = np.array(X_list)
            
            non_dom_indices = get_non_dominated(F)
            
            if len(non_dom_indices) == 0:
                # Fallback: use all points
                non_dom_indices = np.arange(len(F))
            
            # Select start points: use non-dominated points, cycling through weight vectors
            n_starts = min(6, len(non_dom_indices))
            
            # Sort non-dominated points by sum of objectives (best first)
            non_dom_F = F[non_dom_indices]
            sums = np.sum(non_dom_F, axis=1)
            sorted_non_dom = non_dom_indices[np.argsort(sums)]
            
            start_indices = sorted_non_dom[:n_starts]
            
            for s_idx, idx in enumerate(start_indices):
                if self.eval_count >= self.budget or self._check_time():
                    break
                
                # Select a weight vector based on round and start index
                w_idx = (r * n_starts + s_idx) % len(weight_vectors)
                w = np.array(weight_vectors[w_idx])
                
                x_start = X[idx].copy()
                
                best_f = None
                best_x = x_start.copy()
                
                def objective(x, w=w):
                    nonlocal best_f, best_x
                    f = self._evaluate(x, problem)
                    if f is None:
                        return 1e6
                    val = np.dot(w, f)
                    if best_f is None or val < np.dot(w, best_f):
                        best_f = f
                        best_x = x.copy()
                    return val
                
                try:
                    res = minimize(
                        objective, 
                        x_start, 
                        method='Nelder-Mead', 
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 40, 'maxfev': evals_per_run, 'xatol': 1e-7, 'fatol': 1e-7}
                    )
                    
                    if best_f is not None:
                        add_point(best_x, best_f)
                            
                except Exception:
                    pass
        
        # Phase 3: Final polish pass with tighter tolerances
        if self.eval_count < self.budget and not self._check_time():
            F = np.array(F_list)
            X = np.array(X_list)
            
            non_dom_indices = get_non_dominated(F)
            
            if len(non_dom_indices) > 0:
                evals_left = self.budget - self.eval_count
                n_polish = min(len(non_dom_indices), max(1, evals_left // 10))
                
                # Sort by sum of objectives
                non_dom_F = F[non_dom_indices]
                sums = np.sum(non_dom_F, axis=1)
                sorted_non_dom = non_dom_indices[np.argsort(sums)]
                
                for idx in sorted_non_dom[:n_polish]:
                    if self.eval_count >= self.budget or self._check_time():
                        break
                    
                    x_start = X[idx].copy()
                    best_f = None
                    best_x = x_start.copy()
                    
                    def polish_obj(x):
                        nonlocal best_f, best_x
                        f = self._evaluate(x, problem)
                        if f is None:
                            return 1e6
                        val = np.sum(f)
                        if best_f is None or val < np.sum(best_f):
                            best_f = f
                            best_x = x.copy()
                        return val
                    
                    try:
                        minimize(
                            polish_obj,
                            x_start,
                            method='Nelder-Mead',
                            bounds=list(zip(lo, hi)),
                            options={'maxiter': 50, 'maxfev': 10, 'xatol': 1e-8, 'fatol': 1e-8}
                        )
                        if best_f is not None:
                            add_point(best_x, best_f)
                    except Exception:
                        pass
        
        return X_list
# EVOLVE-BLOCK-END
