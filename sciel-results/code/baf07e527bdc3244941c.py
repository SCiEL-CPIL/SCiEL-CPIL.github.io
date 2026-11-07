"""Multi-objective optimizer using a Model-and-Optimize approach with scalarization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    """
    Multi-objective optimizer using a portfolio of scalarized local optimizations.
    Strategy:
    1. Sample a diverse set of random points.
    2. Identify promising regions (non-dominated).
    3. Use SciPy's optimizer to minimize weighted sums of objectives starting from good initial points.
    4. Aggregate all evaluated points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.evals = 0
        self.start_time = None
        self.timeout = 320.0  # Safety margin below 330s

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Store all evaluated points to return (though ignored, good practice)
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

        # 1. Initial Random Sampling
        # Use a moderate portion of the budget for exploration
        n_rand = int(self.budget * 0.25)
        if n_rand < 20:
            n_rand = 20
            
        # Ensure we don't exceed budget
        n_rand = min(n_rand, self.budget)
        
        for _ in range(n_rand):
            if time.time() - self.start_time > self.timeout:
                break
            x = np.random.uniform(lo, hi)
            try:
                fx = safe_eval(x)
                all_X.append(x)
                all_F.append(fx)
            except (TimeoutError, ValueError):
                break
        
        if not all_F:
             # If even random sampling failed or timed out immediately, return empty or single point if possible
             return np.array([np.random.uniform(lo, hi)])

        F_arr = np.array(all_F)
        X_arr = np.array(all_X)
        
        # 2. Scalarization Portfolio with diverse weight vectors
        # Generate weight vectors on the 2-simplex for 3 objectives
        weights = []
        
        # Systematic grid on simplex: w1 + w2 + w3 = 1, all >= 0
        n_grid = 5  # 5x5 grid on triangle
        for i in range(n_grid + 1):
            for j in range(n_grid + 1 - i):
                w1 = i / n_grid
                w2 = j / n_grid
                w3 = 1.0 - w1 - w2
                if w3 >= 0:
                    weights.append(np.array([w1, w2, w3]))
        
        # Add some extreme points for better coverage of front edges
        weights.append(np.array([0.9, 0.05, 0.05]))
        weights.append(np.array([0.05, 0.9, 0.05]))
        weights.append(np.array([0.05, 0.05, 0.9]))
        
        # Filter to non-dominated set for better starts
        def is_dominated(f, F):
            """Check if f is dominated by any point in F."""
            return np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1))
        
        def get_nondominated(F):
            """Return mask of non-dominated points."""
            n = len(F)
            nondom = np.ones(n, dtype=bool)
            for i in range(n):
                if not nondom[i]:
                    continue
                for j in range(n):
                    if i == j or not nondom[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nondom[i] = False
                        break
            return nondom
        
        nondom_mask = get_nondominated(F_arr)
        start_pool = X_arr[nondom_mask] if np.any(nondom_mask) else X_arr
        start_pool_F = F_arr[nondom_mask] if np.any(nondom_mask) else F_arr
        
        # Shuffle weights for diversity in starting points
        np.random.shuffle(weights)
        
        for w in weights:
            if time.time() - self.start_time > self.timeout:
                break
            if self.evals >= self.budget - 10:  # Leave room for refinement
                break
            
            # Find the best starting point for this weight
            weighted_sums = start_pool_F @ w
            best_idx = np.argmin(weighted_sums)
            x0 = start_pool[best_idx]
            
            # Define scalar objective
            def scalar_obj(x, w=w):
                try:
                    fx = safe_eval(x)
                    return np.dot(fx, w)
                except (TimeoutError, ValueError):
                    return np.inf
            
            # Optimize
            try:
                res = minimize(scalar_obj, x0, method='L-BFGS-B', bounds=bounds, 
                              options={'maxiter': 20, 'maxfun': 15})
            except Exception:
                pass
            
            # Update non-dominated set after each optimization
            if all_F:
                F_arr = np.array(all_F)
                X_arr = np.array(all_X)
                nondom_mask = get_nondominated(F_arr)
                if np.any(nondom_mask):
                    start_pool = X_arr[nondom_mask]
                    start_pool_F = F_arr[nondom_mask]
        
        # 3. Local refinement around non-dominated points
        if time.time() - self.start_time < self.timeout and self.evals < self.budget - 5:
            if all_F:
                F_arr = np.array(all_F)
                X_arr = np.array(all_X)
                nondom_mask = get_nondominated(F_arr)
                if np.any(nondom_mask):
                    nondom_X = X_arr[nondom_mask]
                    nondom_F = F_arr[nondom_mask]
                    
                    # Refine each non-dominated point with small perturbations
                    for idx in range(len(nondom_X)):
                        if time.time() - self.start_time > self.timeout:
                            break
                        if self.evals >= self.budget - 2:
                            break
                        
                        x0 = nondom_X[idx]
                        # Try small random perturbations
                        for trial in range(3):
                            if self.evals >= self.budget - 1:
                                break
                            delta = np.random.uniform(-0.05, 0.05, size=self.dim)
                            x_new = np.clip(x0 + delta, lo, hi)
                            try:
                                fx = safe_eval(x_new)
                                all_X.append(x_new)
                                all_F.append(fx)
                            except (TimeoutError, ValueError):
                                break
        
        # 4. Final cleanup / Random fill if budget remains
        while self.evals < self.budget:
            if time.time() - self.start_time > self.timeout:
                break
            x = np.random.uniform(lo, hi)
            try:
                fx = safe_eval(x)
                all_X.append(x)
                all_F.append(fx)
            except (TimeoutError, ValueError):
                break
        
        if all_X:
            return np.array(all_X)
        else:
            return np.empty((0, self.dim))

# EVOLVE-BLOCK-END
