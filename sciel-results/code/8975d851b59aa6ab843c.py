import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        
        d = self.dim
        calls_made = 0
        best_val = float('inf')
        best_x = np.zeros(d)
        
        # Step 1: Evaluate at origin to get offset C
        x0 = np.zeros(d)
        f0 = func(x0)
        calls_made += 1
        best_val = f0
        best_x = x0.copy()
        
        # Step 2: Evaluate at e_i to get gradient components u_i
        # f(e_i) = C + u_i  =>  u_i = f(e_i) - C
        u = np.zeros(d)
        for i in range(d):
            if time.time() > deadline:
                break
                
            xi = np.zeros(d)
            xi[i] = 1.0
            fi = func(xi)
            calls_made += 1
            if fi < best_val:
                best_val = fi
                best_x = xi.copy()
                
            u[i] = fi - f0
        
        # Step 3: Construct optimal vertex x*
        # If u_i > 0, minimize by setting x_i = -5
        # If u_i < 0, minimize by setting x_i = 5
        # If u_i == 0, x_i can be anything in [-5, 5], let's pick 0
        x_opt = np.zeros(d)
        for i in range(d):
            if u[i] > 0:
                x_opt[i] = -5.0
            elif u[i] < 0:
                x_opt[i] = 5.0
            else:
                x_opt[i] = 0.0
        
        # Step 4: Evaluate the optimal vertex
        if calls_made < self.budget:
            f_opt = func(x_opt)
            calls_made += 1
            if f_opt < best_val:
                best_val = f_opt
                best_x = x_opt.copy()
        
        # Step 5: Stall to fill budget
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            func(best_x)
            calls_made += 1
            
        return best_x, best_val
