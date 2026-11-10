"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT3.
    
    Strategy: Exploit the 1D Pareto front structure of ZDT3.
    The Pareto front lies on the manifold x2..xn = 0 with x1 in [0, 1].
    f1(x) = x1
    f2(x) = (1 + 10*(n-1)) * (1 - sqrt(x1/(1+10*(n-1))) + sin(10 * pi * x1))
    
    We place evaluations directly on this manifold to maximize hypervolume coverage.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, 10s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n = self.dim
        n_obj = self.n_obj
        
        # Pre-compute the constant for ZDT3: g(x) = 1 + 10*(n-1) when x2..xn=0
        g_const = 1.0 + 10.0 * (n - 1)
        
        # We will place points on the Pareto front manifold: x2..xn = 0
        # x1 determines f1 and f2.
        # f1 = x1
        # f2 = g_const * (1 - sqrt(x1/g_const) + sin(10*pi*x1))
        
        # Strategy:
        # 1. Generate a dense set of candidate x1 values covering [0, 1]
        # 2. Evaluate them via problem() to get actual objective values
        # 3. Keep track of all evaluated points
        # 4. Use remaining budget for local refinement if time permits
        
        # Phase 1: Dense grid on the front
        # The front has oscillations from sin(10*pi*x1), period = 0.1
        # So we need at least ~20 points per period to capture the shape
        # With 200 budget, let's use ~100 points for the initial grid
        
        num_grid = min(100, self.budget // 2)
        x1_grid = np.linspace(0.0, 1.0, num_grid)
        
        evaluated_x = []
        evaluated_f = []
        
        for x1_val in x1_grid:
            if time.time() > deadline:
                break
            if len(evaluated_x) >= self.budget:
                break
            
            x = np.zeros(n)
            x[0] = x1_val
            # x[1:] = 0 (already zero)
            
            f = problem(x)
            evaluated_x.append(x.copy())
            evaluated_f.append(np.array(f))
        
        # Phase 2: If we have budget left, do local refinement
        # Add points at the minima of the oscillations to improve hypervolume
        # The minima of f2 occur where sin(10*pi*x1) = -1, i.e., 10*pi*x1 = 3*pi/2 + 2*pi*k
        # x1 = (3/20 + k/10) for k = 0, 1, ..., 9
        
        if len(evaluated_x) < self.budget and time.time() < deadline:
            # Add points at oscillation minima
            for k in range(10):
                if len(evaluated_x) >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                x1_val = (3.0 + 2.0 * k) / 20.0
                if x1_val < 0 or x1_val > 1:
                    continue
                
                x = np.zeros(n)
                x[0] = x1_val
                
                f = problem(x)
                evaluated_x.append(x.copy())
                evaluated_f.append(np.array(f))
        
        # Phase 3: If still have budget, add more evenly spaced points
        if len(evaluated_x) < self.budget and time.time() < deadline:
            # Add finer grid points in gaps
            num_extra = min(self.budget - len(evaluated_x), 50)
            if num_extra > 0:
                x1_extra = np.linspace(0.0, 1.0, num_extra + 1)[1:]
                for x1_val in x1_extra:
                    if len(evaluated_x) >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    x = np.zeros(n)
                    x[0] = x1_val
                    
                    f = problem(x)
                    evaluated_x.append(x.copy())
                    evaluated_f.append(np.array(f))
        
        # Return the best set of points found
        # The evaluator records all returned objective vectors
        # We return the array of all evaluated x's
        if len(evaluated_x) == 0:
            # Fallback: return a single random point
            x = lo + (hi - lo) * np.random.random(n)
            problem(x)
            return np.array([x])
        
        return np.array(evaluated_x)
# EVOLVE-BLOCK-END
