"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Linear function optimizer using exact gradient recovery."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # Track calls and best
        call_count = 0
        best_x = None
        best_f = np.inf
        
        def safe_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return best_f
            call_count += 1
            fx = func(np.clip(x, -5, 5))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx
        
        # Phase 1: Exact gradient recovery using d+1 evaluations
        # Evaluate at origin and at points e_i * delta for i=1..d
        # This gives us f(0) and f(delta*e_i) for each dimension
        # For linear f(x) = w^T x + b:
        #   f(0) = b
        #   f(delta*e_i) = w_i * delta + b
        #   w_i = (f(delta*e_i) - f(0)) / delta
        
        origin = np.zeros(dim)
        f0 = safe_func(origin)
        
        delta = 1.0
        w = np.zeros(dim)
        
        for i in range(dim):
            if call_count >= budget or time.time() > deadline:
                break
            x_i = np.zeros(dim)
            x_i[i] = delta
            f_i = safe_func(x_i)
            w[i] = (f_i - f0) / delta
        
        # Phase 2: Jump to the optimal vertex
        # For minimization of w^T x + b on [-5,5]^d:
        #   x_i = -5 if w_i > 0, x_i = +5 if w_i < 0, x_i = 0 if w_i == 0
        if call_count < budget and time.time() < deadline:
            x_vertex = np.zeros(dim)
            for i in range(dim):
                if w[i] > 0:
                    x_vertex[i] = -5.0
                elif w[i] < 0:
                    x_vertex[i] = 5.0
                else:
                    x_vertex[i] = 0.0
            
            f_vertex = safe_func(x_vertex)
        
        # Phase 3: Local refinement around best point using (1+1)-ES
        if best_x is None:
            best_x = origin.copy()
            best_f = f0
        
        x = best_x.copy()
        fx = best_f
        sigma = 0.1
        
        # Adaptive (1+1)-ES for remaining budget
        rng = np.random.default_rng(42)
        while call_count < budget and time.time() < deadline:
            y = x + sigma * rng.standard_normal(dim)
            fy = safe_func(y)
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.1
            else:
                sigma *= 0.9
            
            # Early stop if sigma becomes very small
            if sigma < 1e-8:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
