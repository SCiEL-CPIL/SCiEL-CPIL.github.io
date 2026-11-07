"""Powell's Method with random initialization for Bent Cigar optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar


class Optimizer:
    """Powell's directional search optimizer for Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = np.inf
        best_x = np.zeros(dim)

        # Wrapper to count calls and clip
        def wrapper(x):
            nonlocal calls, best_f, best_x
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Initial random sampling to find a good starting point
        # Use up to 10% of budget, at least 5, at most 20
        n_init = max(5, min(20, budget // 10))
        x_start = np.zeros(dim)
        f_start = np.inf
        
        rng = np.random.default_rng(42)
        # Evaluate origin
        f0 = wrapper(np.zeros(dim))
        if f0 < f_start:
            f_start = f0
            x_start = np.zeros(dim)

        for _ in range(n_init - 1):
            if calls >= budget:
                break
            x_try = rng.uniform(-5, 5, dim)
            f_try = wrapper(x_try)
            if f_try < f_start:
                f_start = f_try
                x_start = x_try.copy()
        
        x = x_start.copy()

        # Directions: standard basis
        directions = np.eye(dim)
        
        # Powell's method parameters
        tol = 1e-10
        max_iter = 100 
        line_search_maxiter = 15 # Limit line search cost

        for _ in range(max_iter):
            if calls >= budget - 5: # Leave some budget
                break
            
            x_prev = x.copy()
            f_prev = wrapper(x_prev)
            
            # Track improvements along each direction
            improvements = []
            best_dir_idx = 0
            best_improvement = 0.0

            for i in range(dim):
                if calls >= budget - 5:
                    break
                
                d = directions[i]
                
                # Determine bounds for t
                t_min = 0.0
                t_max = 0.0
                
                for j in range(dim):
                    if d[j] > 1e-10:
                        t_max = min(t_max, (5.0 - x[j]) / d[j])
                    elif d[j] < -1e-10:
                        t_min = max(t_min, (-5.0 - x[j]) / d[j])
                
                if t_min > t_max:
                    continue

                # Define objective for line search
                def line_obj(t):
                    return wrapper(x + t * d)

                # Perform line search
                if t_max - t_min < 1e-8:
                    continue

                try:
                    res = minimize_scalar(line_obj, bounds=(t_min, t_max), method='bounded', 
                                          options={'xatol': 1e-4, 'maxiter': line_search_maxiter})
                    if res.success:
                        x = x + res.x * d
                        f_new = res.fun
                        improvement = f_prev - f_new
                        improvements.append(improvement)
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_dir_idx = i
                    else:
                        improvements.append(0.0)
                except Exception:
                    improvements.append(0.0)
            
            # Update directions
            if len(improvements) == dim:
                sum_improvements = sum(improvements)
                min_improvement = min(improvements)
                
                # Check convergence
                if abs(sum_improvements) < tol:
                    break
                
                # Rotation condition
                if sum_improvements > 2 * min_improvement:
                    min_idx = np.argmin(improvements)
                    new_dir = x - x_prev
                    norm_new_dir = np.linalg.norm(new_dir)
                    if norm_new_dir > 1e-10:
                        directions[min_idx] = new_dir / norm_new_dir
                    else:
                        break
            else:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
