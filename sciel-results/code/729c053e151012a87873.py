"""Multi-objective optimizer using grid-based decomposition with local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    """
    Multi-objective optimizer using grid-based decomposition.
    Strategy:
    1. Partition the 4D domain into a coarse hypercube grid.
    2. Evaluate grid vertices to build a coarse map.
    3. Identify promising regions based on non-dominated solutions.
    4. Perform targeted local refinement using Nelder-Mead on promising cells.
    5. Use remaining budget for additional random sampling and refinement.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.evals = 0
        self.start_time = None
        self.timeout = 280.0  # Safety margin below 330s

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Store all evaluated points
        all_X = []
        all_F = []
        
        def safe_eval(x):
            """Wraps problem call with time and budget checks."""
            if time.time() - self.start_time > self.timeout:
                raise TimeoutError("Time limit exceeded")
            if self.evals >= self.budget:
                raise ValueError("Budget exceeded")
            
            x = np.asarray(x)
            fx = problem(x)
            self.evals += 1
            return np.array(fx)

        def is_dominated(f, F):
            """Check if f is dominated by any point in F."""
            if len(F) == 0:
                return False
            return np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1))

        def get_nondominated(F):
            """Return indices of non-dominated points."""
            if len(F) == 0:
                return np.array([], dtype=int)
            nd_idx = []
            for i in range(len(F)):
                if not is_dominated(F[i], np.delete(F, i, axis=0)):
                    nd_idx.append(i)
            return np.array(nd_idx, dtype=int)

        # 1. Grid-based coarse exploration
        # For 4D with 3 points per dimension: 3^4 = 81 vertices
        # We can afford this within budget
        n_per_dim = 3
        n_grid = n_per_dim ** self.dim
        
        # Generate grid vertices
        grid_points = []
        for i1 in range(n_per_dim):
            for i2 in range(n_per_dim):
                for i3 in range(n_per_dim):
                    for i4 in range(n_per_dim):
                        x = np.array([
                            lo[0] + (hi[0] - lo[0]) * i1 / (n_per_dim - 1),
                            lo[1] + (hi[1] - lo[1]) * i2 / (n_per_dim - 1),
                            lo[2] + (hi[2] - lo[2]) * i3 / (n_per_dim - 1),
                            lo[3] + (hi[3] - lo[3]) * i4 / (n_per_dim - 1)
                        ])
                        grid_points.append(x)
        
        grid_points = np.array(grid_points)
        
        # Evaluate grid vertices
        for x in grid_points:
            if time.time() - self.start_time > self.timeout:
                break
            if self.evals >= self.budget:
                break
            try:
                fx = safe_eval(x)
                all_X.append(x)
                all_F.append(fx)
            except (TimeoutError, ValueError):
                break
        
        if not all_F:
            return np.empty((0, self.dim))
        
        F_arr = np.array(all_F)
        X_arr = np.array(all_X)
        
        # 2. Identify non-dominated solutions from grid
        nd_idx = get_nondominated(F_arr)
        if len(nd_idx) > 0:
            nd_X = X_arr[nd_idx]
            nd_F = F_arr[nd_idx]
        else:
            nd_X = X_arr
            nd_F = F_arr
        
        # 3. Local refinement using Nelder-Mead on promising regions
        # Use weighted sum scalarization with diverse weights
        weights = []
        # Uniform weight
        weights.append(np.ones(self.n_obj) / self.n_obj)
        # Extreme weights
        for i in range(self.n_obj):
            w = np.zeros(self.n_obj)
            w[i] = 0.7
            rem = 0.3 / (self.n_obj - 1)
            for j in range(self.n_obj):
                if j != i:
                    w[j] = rem
            weights.append(w)
        # Mixed weights
        weights.append(np.array([0.5, 0.3, 0.2]))
        weights.append(np.array([0.2, 0.3, 0.5]))
        weights.append(np.array([0.4, 0.4, 0.2]))
        weights.append(np.array([0.2, 0.4, 0.4]))
        weights.append(np.array([0.4, 0.2, 0.4]))
        weights.append(np.array([0.3, 0.5, 0.2]))
        weights.append(np.array([0.2, 0.5, 0.3]))
        weights.append(np.array([0.5, 0.2, 0.3]))
        weights.append(np.array([0.6, 0.2, 0.2]))
        weights.append(np.array([0.2, 0.6, 0.2]))
        weights.append(np.array([0.2, 0.2, 0.6]))
        
        # For each weight, find best starting point and run Nelder-Mead
        for w in weights:
            if time.time() - self.start_time > self.timeout:
                break
            if self.evals >= self.budget - 10:
                break
            
            # Find best starting point for this weight
            if len(nd_X) > 0:
                weighted_sums = nd_F @ w
                best_idx = np.argmin(weighted_sums)
                x0 = nd_X[best_idx]
            else:
                x0 = X_arr[0]
            
            # Define scalar objective
            def scalar_obj(x, w=w):
                try:
                    fx = safe_eval(x)
                    return np.dot(fx, w)
                except (TimeoutError, ValueError):
                    return np.inf
            
            # Run Nelder-Mead
            try:
                res = minimize(scalar_obj, x0, method='Nelder-Mead', 
                              bounds=bounds, options={'maxiter': 20, 'xatol': 1e-4, 'fatol': 1e-4})
            except Exception:
                pass
        
        # 4. Additional random sampling to fill gaps
        n_random = min(30, self.budget - self.evals)
        for _ in range(n_random):
            if time.time() - self.start_time > self.timeout:
                break
            if self.evals >= self.budget:
                break
            x = np.random.uniform(lo, hi)
            try:
                fx = safe_eval(x)
                all_X.append(x)
                all_F.append(fx)
            except (TimeoutError, ValueError):
                break
        
        # 5. Final local refinement around best non-dominated points
        if time.time() - self.start_time <= self.timeout and self.evals < self.budget - 5:
            F_arr = np.array(all_F)
            X_arr = np.array(all_X)
            nd_idx = get_nondominated(F_arr)
            if len(nd_idx) > 0:
                nd_X = X_arr[nd_idx]
                # Pick top non-dominated points for local refinement
                n_refine = min(5, len(nd_X), (self.budget - self.evals) // 10)
                for i in range(n_refine):
                    if time.time() - self.start_time > self.timeout:
                        break
                    if self.evals >= self.budget - 2:
                        break
                    
                    x0 = nd_X[i]
                    
                    # Try small perturbations
                    for step in [0.1, 0.05, 0.02]:
                        if self.evals >= self.budget - 1:
                            break
                        for d in range(self.dim):
                            if self.evals >= self.budget - 1:
                                break
                            for sign in [-1, 1]:
                                if self.evals >= self.budget - 1:
                                    break
                                x_new = x0.copy()
                                x_new[d] += sign * step
                                x_new = np.clip(x_new, lo, hi)
                                try:
                                    fx = safe_eval(x_new)
                                    all_X.append(x_new)
                                    all_F.append(fx)
                                except (TimeoutError, ValueError):
                                    break
        
        # Return the population array
        if all_X:
            return np.array(all_X)
        else:
            return np.empty((0, self.dim))

# EVOLVE-BLOCK-END
