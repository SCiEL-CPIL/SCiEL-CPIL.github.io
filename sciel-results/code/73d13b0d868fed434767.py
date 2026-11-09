"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start local search with Halton initialization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        dim = self.dim
        total_budget = self.budget
        
        # Halton sequence generator for quasi-random space-filling
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def generate_halton(n, dim, offset=1):
            points = []
            for i in range(offset, offset + n):
                point = []
                # Use first 'dim' primes as bases
                bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][:dim]
                for j in range(dim):
                    point.append(halton(i, bases[j]))
                points.append(point)
            return np.array(points)

        # Initialize points in [-5, 5]^dim
        # We want to cover the space well. 
        # Let's pick a set of starting points.
        # Since budget is 500 and dim is 5, we can afford several restarts.
        # Let's say we split the budget into chunks.
        # Chunk size: ~50-100 calls per restart?
        # Let's try 10 restarts of 50 calls each.
        
        num_restarts = 10
        calls_per_restart = max(10, total_budget // num_restarts)
        
        # Generate Halton points
        halton_points = generate_halton(num_restarts, dim)
        
        # Map [0,1]^dim to [-5,5]^dim
        init_points = halton_points * 10.0 - 5.0
        
        best_x = None
        best_f = np.inf
        total_calls = 0
        
        # Local search function: Simple (1+1)-ES or Nelder-Mead?
        # Nelder-Mead is robust but can be slow if not tuned.
        # (1+1)-ES is very fast but can get stuck.
        # Given the "Büche-Rastrigin" nature (many local minima), 
        # a simple local search might converge fast to a local min.
        # Let's use a simple adaptive step size local search.
        
        def local_search(x_start, max_calls):
            x = x_start.copy()
            f = func(x)
            calls = 1
            sigma = 1.0 # Initial step size
            dim = len(x)
            
            # Simple (1+1)-ES with adaptation
            # If improvement, increase sigma slightly? No, usually decrease if stuck, increase if moving well.
            # Standard (1+1)-ES: 
            # y = x + sigma * N(0,1)
            # if f(y) < f(x): x = y, sigma *= 1.05 (or similar)
            # else: sigma /= 1.05
            
            # But for AOCC, we want fast progress.
            # Maybe a slightly larger initial sigma is better to jump out of flat areas?
            # Büche-Rastrigin has a specific structure.
            
            while calls < max_calls:
                if time.time() >= deadline:
                    break
                
                # Generate candidate
                z = np.random.standard_normal(dim)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                
                f_y = func(y)
                calls += 1
                
                if f_y < f:
                    x = y
                    f = f_y
                    # Increase step size slightly to explore further? 
                    # Or keep it constant? 
                    # In (1+1)-ES, constant sigma is common, but adaptive is better.
                    # Let's try a mild increase on success to escape local minima faster?
                    # Actually, standard theory says constant sigma is optimal for unimodal.
                    # For multimodal, maybe we should restart more often.
                    # Let's stick to a simple adaptive scheme.
                    sigma *= 1.01 
                else:
                    sigma /= 1.02
                    # Prevent sigma from becoming too small
                    if sigma < 1e-6:
                        break
            
            return x, f, calls

        # Execute restarts
        for i in range(num_restarts):
            if total_calls >= total_budget:
                break
            if time.time() >= deadline:
                break
                
            x_init = init_points[i]
            
            # Allocate remaining budget to this restart
            remaining = total_budget - total_calls
            alloc = min(calls_per_restart, remaining)
            
            if alloc <= 0:
                break
            
            x_res, f_res, calls_used = local_search(x_init, alloc)
            
            total_calls += calls_used
            
            if f_res < best_f:
                best_f = f_res
                best_x = x_res.copy()
                
        # If we have budget left, do a final fine-tuning around the best point?
        if best_x is not None and total_calls < total_budget:
            remaining = total_budget - total_calls
            if remaining > 10:
                # Fine tune with smaller initial sigma
                x_final, f_final, _ = local_search(best_x, remaining)
                if f_final < best_f:
                    best_f = f_final
                    best_x = x_final

        if best_x is None:
            # Fallback
            best_x = np.zeros(dim)
            best_f = func(best_x)
            total_calls += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
