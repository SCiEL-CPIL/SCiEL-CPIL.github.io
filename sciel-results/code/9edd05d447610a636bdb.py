# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-objective optimizer for RE37 using weighted-sum decomposition with Nelder-Mead refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # 30s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)
        
        evaluated_points = []
        evaluated_f = []
        eval_count = 0
        
        def evaluate(x):
            nonlocal eval_count
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            evaluated_points.append(x.copy())
            evaluated_f.append(f.copy())
            eval_count += 1
            return f
        
        # Phase 1: Stratified initial sampling for broad coverage
        n_init = 30
        X_init = []
        
        # Corners of the hypercube
        corners = np.array(np.meshgrid(*[[lo[i], hi[i]] for i in range(self.dim)])).T.reshape(-1, self.dim)
        X_init.extend(corners.tolist())
        
        # Center point
        X_init.append(((lo + hi) / 2).tolist())
        
        # Midpoints of edges
        for i in range(self.dim):
            for j in range(i+1, self.dim):
                base = (lo + hi) / 2
                for c1 in [0, 1]:
                    for c2 in [0, 1]:
                        pt = base.copy()
                        pt[i] = lo[i] if c1 == 0 else hi[i]
                        pt[j] = lo[j] if c2 == 0 else hi[j]
                        X_init.append(pt.tolist())
        
        # Add random points to fill up to n_init
        while len(X_init) < n_init:
            X_init.append((lo + (hi - lo) * rng.random(self.dim)).tolist())
        
        X_init = np.array(X_init[:n_init])
        
        # Evaluate initial points
        for x in X_init:
            if eval_count >= self.budget or time.time() > deadline:
                break
            evaluate(x)
        
        # Phase 2: Weighted-sum decomposition with Nelder-Mead refinement
        # Generate diverse weight vectors
        n_weights = 12
        weights = []
        # Ensure coverage: include extreme weights and balanced ones
        for _ in range(n_weights):
            w = rng.dirichlet(np.ones(self.n_obj) * 0.5)  # More spread out
            weights.append(w)
        # Add some extreme weights
        for i in range(self.n_obj):
            w = np.ones(self.n_obj) * 0.05
            w[i] = 1.0 - 0.05 * (self.n_obj - 1)
            weights.append(w)
        weights = np.array(weights[:n_weights])
        
        # For each weight vector, find best starting point and refine
        F_arr = np.array(evaluated_f) if evaluated_f else np.empty((0, self.n_obj))
        X_arr = np.array(evaluated_points) if evaluated_points else np.empty((0, self.dim))
        
        for w in weights:
            if eval_count >= self.budget or time.time() > deadline:
                break
            
            # Find best starting point for this weight direction
            if len(X_arr) > 0:
                scores = F_arr @ w
                best_idx = np.argmin(scores)
                x0 = X_arr[best_idx].copy()
            else:
                x0 = (lo + hi) / 2
            
            # Define scalarized objective with evaluation tracking
            def make_scalar_obj(w_vec):
                def scalar_obj(x):
                    if time.time() > deadline:
                        return 1e10
                    x = np.clip(x, lo, hi)
                    f = np.array(problem(x))
                    evaluated_points.append(x.copy())
                    evaluated_f.append(f.copy())
                    eval_count += 1
                    return float(f @ w_vec)
                return scalar_obj
            
            scalar_obj = make_scalar_obj(w)
            
            # Run Nelder-Mead with limited iterations
            try:
                result = minimize(scalar_obj, x0, method='Nelder-Mead',
                                  options={'maxiter': 15, 'xatol': 1e-3, 'fatol': 1e-3})
            except Exception:
                pass
        
        return X_arr if len(X_arr) > 0 else np.array(evaluated_points)
# EVOLVE-BLOCK-END
