import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        n_calls = 0
        
        best_x = np.zeros(dim)
        best_fx = float('inf')
        
        # We will use a multi-start strategy with Nelder-Mead
        # Nelder-Mead is fast for 5D.
        # We allocate a fixed number of calls per restart.
        
        # Determine number of restarts
        # Each NM run typically takes ~50-100 evaluations to converge in 5D.
        # Let's try 5 restarts of 100 calls each? Or 10 restarts of 50?
        # More restarts = more exploration.
        
        n_restarts = 10
        calls_per_restart = budget // n_restarts
        
        rng = np.random.default_rng(42)
        
        for r in range(n_restarts):
            if time.time() > deadline:
                break
            if n_calls >= budget:
                break
            
            # Initial point for this restart
            # First restart at 0, others random
            if r == 0:
                x0 = np.zeros(dim)
            else:
                x0 = rng.uniform(-4, 4, dim)
            
            # Evaluate x0
            fx0 = func(x0)
            n_calls += 1
            
            if fx0 < best_fx:
                best_fx = fx0
                best_x = x0.copy()
            
            # Run Nelder-Mead
            # We need to limit the number of function evaluations in scipy
            # scipy.optimize.minimize does not have a strict "max_evals" for Nelder-Mead 
            # in all versions, but we can use a wrapper or just let it run 
            # and break if we exceed budget.
            
            # However, calling scipy inside a loop might have overhead.
            # A manual Nelder-Mead implementation is safer for budget control.
            
            # Let's implement a simple Nelder-Mead manually.
            x = x0.copy()
            fx = fx0
            
            # Simplex initialization
            simplex = []
            simplex.append(x.copy())
            simplex_f = [fx]
            
            # Other vertices
            for i in range(dim):
                xi = x.copy()
                if xi[i] == 0:
                    xi[i] = 0.1
                else:
                    xi[i] += 0.1 * abs(xi[i])
                xi = np.clip(xi, -5, 5)
                fxi = func(xi)
                n_calls += 1
                simplex.append(xi)
                simplex_f.append(fxi)
                
                if fxi < best_fx:
                    best_fx = fxi
                    best_x = xi.copy()
            
            # Main NM loop
            max_iter = calls_per_restart - (1 + dim) # Account for initial simplex
            for _ in range(max_iter):
                if n_calls >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Sort simplex
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fxr = func(xr)
                n_calls += 1
                
                if fxr < best_fx:
                    best_fx = fxr
                    best_x = xr.copy()
                
                if simplex_f[0] <= fxr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fxr
                elif fxr < simplex_f[0]:
                    # Expansion
                    xe = centroid + 2 * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fxe = func(xe)
                    n_calls += 1
                    if fxe < best_fx:
                        best_fx = fxe
                        best_x = xe.copy()
                    if fxe < fxr:
                        simplex[-1] = xe
                        simplex_f[-1] = fxe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fxr
                else:
                    # Contraction
                    if fxr < simplex_f[-1]:
                        # Outside contraction
                        xc = centroid + 0.5 * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid + 0.5 * (simplex[-1] - centroid)
                    
                    xc = np.clip(xc, -5, 5)
                    fxc = func(xc)
                    n_calls += 1
                    if fxc < best_fx:
                        best_fx = fxc
                        best_x = xc.copy()
                    
                    if fxc < max(fxr, simplex_f[-1]):
                        simplex[-1] = xc
                        simplex_f[-1] = fxc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fxi = func(simplex[i])
                            n_calls += 1
                            simplex_f[i] = fxi
                            if fxi < best_fx:
                                best_fx = fxi
                                best_x = simplex[i].copy()
        
        return best_x, best_fx
