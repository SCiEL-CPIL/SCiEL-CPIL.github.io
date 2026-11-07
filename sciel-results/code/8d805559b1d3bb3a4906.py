"""Halton-grid exploration followed by Nelder-Mead refinement for BBOB f23 (Katsuura)."""
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
        
        dim = self.dim
        budget = self.budget
        
        # Halton sequence generator
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        
        # Phase 1: Halton-grid exploration
        # Use 100 calls for exploration to reliably locate the global basin early
        num_exploration = 100
        
        x_best = None
        f_best = float('inf')
        calls = 0
        
        # Generate Halton points in [0, 1]^dim and map to [-5, 5]^dim
        for i in range(1, num_exploration + 1):
            if time.time() > deadline or calls >= budget:
                break
            
            x = np.zeros(dim)
            for d in range(dim):
                # Use different prime bases for each dimension
                base = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][d % 10]
                x[d] = -5.0 + 10.0 * halton(i, base)
            
            f = func(x)
            calls += 1
            
            if f < f_best:
                f_best = f
                x_best = x.copy()
        
        # Phase 2: Nelder-Mead refinement from the best sample
        if calls < budget and time.time() < deadline and x_best is not None:
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
        
        # Return the best result found
        if x_best is None:
            # Fallback: return a random point if no evaluations were made
            rng = np.random.default_rng(0)
            x_best = rng.uniform(-5, 5, dim)
            f_best = func(x_best)
            calls += 1
        
        return x_best, f_best
# EVOLVE-BLOCK-END
