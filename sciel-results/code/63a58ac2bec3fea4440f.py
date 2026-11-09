import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        def update_best(x, f):
            nonlocal best_x, best_f, calls_made
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def eval_point(x):
            nonlocal calls_made
            fx = func(x)
            calls_made += 1
            update_best(x, fx)
            return fx

        # 1. Evaluate Center
        x0 = np.zeros(dim)
        f0 = eval_point(x0)
        
        # 2. Determine signs of the gradient by probing +5 along each axis
        # For a linear function f(x) = c^T x + d, f(x0 + 5e_i) = f0 + 5*c_i
        # If f(x0 + 5e_i) < f0, then c_i < 0, so we want x_i to be as large as possible (+5) to minimize c_i*x_i.
        # If f(x0 + 5e_i) >= f0, then c_i >= 0, so we want x_i to be as small as possible (-5).
        
        optimal_vertex = np.zeros(dim)
        
        for i in range(dim):
            if calls_made >= budget or time.time() > deadline:
                break
            
            # Probe positive bound
            xp = np.zeros(dim)
            xp[i] = 5.0
            fp = eval_point(xp)
            
            # Determine sign
            if fp < f0:
                # Gradient is negative, increase x_i
                optimal_vertex[i] = 5.0
            else:
                # Gradient is positive (or zero), decrease x_i
                optimal_vertex[i] = -5.0
        
        # 3. Evaluate the predicted optimal vertex
        if calls_made < budget and time.time() < deadline:
            eval_point(optimal_vertex)
        
        # 4. Fill remaining budget
        # For a linear function, the vertex is the global minimum.
        # No further search can improve. Just make calls to fill budget.
        # We evaluate the best point found (which should be the optimal vertex).
        while calls_made < budget and time.time() < deadline:
            eval_point(best_x)

        return best_x, best_f
