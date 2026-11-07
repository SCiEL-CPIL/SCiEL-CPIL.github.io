"""Multi-start Nelder-Mead with adaptive refinement for BBOB f23 (Katsuura)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        rng = np.random.default_rng(123)
        
        # Global best tracking
        x_best = rng.uniform(-5, 5, self.dim)
        f_best = func(x_best)
        calls = 1
        
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Multi-start Nelder-Mead to find good basins
        # Use 10 initial points, each with a small Nelder-Mead budget
        num_starts = 10
        nm_budget_per_start = 30  # 10 * 30 = 300 calls for exploration
        
        for s in range(num_starts):
            if time.time() > deadline or calls >= budget:
                break
            
            # Initialize Nelder-Mead simplex
            # Start from a random point
            x0 = rng.uniform(-4, 4, dim)
            f0 = func(x0)
            calls += 1
            if f0 < f_best:
                f_best = f0
                x_best = x0.copy()
            
            # Create simplex: x0 and x0 + delta * e_i for each dimension
            delta = 0.5
            simplex = [x0.copy()]
            f_simplex = [f0]
            
            for i in range(dim):
                x_i = x0.copy()
                x_i[i] += delta
                x_i = np.clip(x_i, -5, 5)
                f_i = func(x_i)
                calls += 1
                if f_i < f_best:
                    f_best = f_i
                    x_best = x_i.copy()
                simplex.append(x_i)
                f_simplex.append(f_i)
            
            # Nelder-Mead iterations
            for nm_step in range(nm_budget_per_start):
                if time.time() > deadline or calls >= budget:
                    break
                
                # Sort simplex by function value
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                x_worst = simplex[-1]
                f_worst = f_simplex[-1]
                
                # Reflection
                gamma = 1.0
                x_ref = centroid + gamma * (centroid - x_worst)
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = func(x_ref)
                calls += 1
                
                if f_ref < f_simplex[0]:
                    # Expansion
                    x_exp = centroid + 2.0 * (centroid - x_worst)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = func(x_exp)
                    calls += 1
                    
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_simplex[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_simplex[-1] = f_ref
                elif f_ref < f_simplex[-2]:
                    # Just replace worst with reflection
                    simplex[-1] = x_ref
                    f_simplex[-1] = f_ref
                else:
                    # Contraction
                    rho = 0.5
                    x_con = centroid + rho * (x_worst - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = func(x_con)
                    calls += 1
                    
                    if f_con < f_worst:
                        simplex[-1] = x_con
                        f_simplex[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            f_simplex[i] = func(simplex[i])
                            calls += 1
                            if calls >= budget:
                                break
                        
                        if calls >= budget:
                            break
                
                # Update global best
                for i in range(dim + 1):
                    if f_simplex[i] < f_best:
                        f_best = f_simplex[i]
                        x_best = simplex[i].copy()
        
        # Phase 2: Refine the best point with a tighter Nelder-Mead
        if calls < budget and time.time() < deadline:
            x = x_best.copy()
            f = f_best
            
            # Smaller delta for refinement
            delta = 0.1
            simplex = [x.copy()]
            f_simplex = [f]
            
            for i in range(dim):
                x_i = x.copy()
                x_i[i] += delta
                x_i = np.clip(x_i, -5, 5)
                f_i = func(x_i)
                calls += 1
                if f_i < f_best:
                    f_best = f_i
                    x_best = x_i.copy()
                simplex.append(x_i)
                f_simplex.append(f_i)
            
            # Nelder-Mead iterations for refinement
            remaining = budget - calls
            for nm_step in range(remaining):
                if time.time() > deadline or calls >= budget:
                    break
                
                # Sort simplex by function value
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                x_worst = simplex[-1]
                f_worst = f_simplex[-1]
                
                # Reflection
                gamma = 1.0
                x_ref = centroid + gamma * (centroid - x_worst)
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = func(x_ref)
                calls += 1
                
                if f_ref < f_simplex[0]:
                    # Expansion
                    x_exp = centroid + 2.0 * (centroid - x_worst)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = func(x_exp)
                    calls += 1
                    
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_simplex[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_simplex[-1] = f_ref
                elif f_ref < f_simplex[-2]:
                    # Just replace worst with reflection
                    simplex[-1] = x_ref
                    f_simplex[-1] = f_ref
                else:
                    # Contraction
                    rho = 0.5
                    x_con = centroid + rho * (x_worst - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = func(x_con)
                    calls += 1
                    
                    if f_con < f_worst:
                        simplex[-1] = x_con
                        f_simplex[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            f_simplex[i] = func(simplex[i])
                            calls += 1
                            if calls >= budget:
                                break
                        
                        if calls >= budget:
                            break
                
                # Update global best
                for i in range(dim + 1):
                    if f_simplex[i] < f_best:
                        f_best = f_simplex[i]
                        x_best = simplex[i].copy()
        
        return x_best, f_best
# EVOLVE-BLOCK-END
