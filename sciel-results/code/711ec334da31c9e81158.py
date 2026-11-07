import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        d = self.dim
        budget = self.budget
        calls = 0
        
        best_x = None
        best_f = float('inf')
        
        rng = np.random.default_rng()
        
        # Phase 1: Random Exploration
        # Use 30 calls to find a good starting point
        n_random = min(30, budget)
        for _ in range(n_random):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, d)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        if calls >= budget:
            return best_x, best_f
            
        # Phase 2: Nelder-Mead Local Search
        # Initialize simplex around best_x
        # Start with a small step size
        step = 0.5
        simplex = np.zeros((d + 1, d))
        simplex[0] = best_x
        for i in range(d):
            simplex[i + 1] = best_x
            simplex[i + 1, i] += step
            
        # Evaluate initial simplex
        fvals = np.zeros(d + 1)
        for i in range(d + 1):
            if time.time() > deadline:
                break
            simplex[i] = np.clip(simplex[i], -5, 5)
            fvals[i] = func(simplex[i])
            calls += 1
            if fvals[i] < best_f:
                best_f = fvals[i]
                best_x = simplex[i].copy()
                
        if calls >= budget:
            return best_x, best_f

        # Nelder-Mead Loop
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrink
        
        while calls < budget:
            if time.time() > deadline:
                break
                
            # Sort simplex by fitness
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]
            
            # p_best is simplex[0], p_worst is simplex[-1]
            p_best = simplex[0]
            p_worst = simplex[-1]
            f_best = fvals[0]
            f_worst = fvals[-1]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            p_ref = centroid + alpha * (centroid - p_worst)
            p_ref = np.clip(p_ref, -5, 5)
            
            if calls >= budget:
                break
            
            f_ref = func(p_ref)
            calls += 1
            if f_ref < best_f:
                best_f = f_ref
                best_x = p_ref.copy()
                
            if f_ref < f_best:
                # Expansion
                if calls >= budget:
                    break
                p_exp = centroid + gamma * (centroid - p_worst)
                p_exp = np.clip(p_exp, -5, 5)
                f_exp = func(p_exp)
                calls += 1
                if f_exp < best_f:
                    best_f = f_exp
                    best_x = p_exp.copy()
                    
                if f_exp < f_ref:
                    simplex[-1] = p_exp
                    fvals[-1] = f_exp
                else:
                    simplex[-1] = p_ref
                    fvals[-1] = f_ref
            elif f_ref < fvals[-2]:
                # Accept reflection
                simplex[-1] = p_ref
                fvals[-1] = f_ref
            else:
                # Contraction
                if calls >= budget:
                    break
                p_con = centroid + rho * (p_worst - centroid)
                p_con = np.clip(p_con, -5, 5)
                f_con = func(p_con)
                calls += 1
                if f_con < best_f:
                    best_f = f_con
                    best_x = p_con.copy()
                    
                if f_con < f_worst:
                    simplex[-1] = p_con
                    fvals[-1] = f_con
                else:
                    # Shrink
                    # Need to evaluate d-1 new points (all except best)
                    # Check if we have enough budget
                    if calls + (d - 1) > budget:
                        break
                        
                    for i in range(1, d + 1):
                        if time.time() > deadline:
                            break
                        simplex[i] = p_best + sigma * (simplex[i] - p_best)
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls += 1
                        if fvals[i] < best_f:
                            best_f = fvals[i]
                            best_x = simplex[i].copy()

        return best_x, best_f
