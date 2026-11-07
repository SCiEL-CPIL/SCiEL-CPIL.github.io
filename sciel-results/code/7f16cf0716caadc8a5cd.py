"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for BBOB f5 (Linear Slope).
    Strategy:
    1. Estimate gradient signs using minimal evaluations (center + 5 axis points).
    2. Determine the optimal vertex of the box [-5, 5]^dim.
    3. Evaluate the vertex to get the global minimum (for linear functions).
    4. Use remaining budget for a (1+1)-ES polish to handle potential non-linearities
       or numerical issues, ensuring robustness.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls_used = 0

        def safe_call(x):
            nonlocal calls_used, best_x, best_f
            if calls_used >= budget or time.time() >= deadline:
                return None
            try:
                calls_used += 1
                f = float(func(x))
            except Exception:
                return None
            if not np.isfinite(f):
                return None
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Gradient Sign Estimation
        # Evaluate at center (0, 0, ..., 0)
        x0 = np.zeros(dim)
        f0 = safe_call(x0)
        
        if f0 is None:
            # Fallback if center fails
            rng = np.random.default_rng(0)
            x_rand = rng.uniform(-5, 5, dim)
            f_rand = safe_call(x_rand)
            if f_rand is not None:
                return best_x, best_f
            return best_x, best_f

        # Evaluate at x0 + h * e_i for i in range(dim)
        # h=4.0 is a good balance: large enough for signal, small enough to stay in bounds
        h = 4.0
        grad_signs = np.zeros(dim)
        
        for i in range(dim):
            if time.time() >= deadline:
                break
            xi = x0.copy()
            xi[i] += h
            fi = safe_call(xi)
            
            if fi is None:
                # If evaluation fails, assume no gradient (keep 0)
                grad_signs[i] = 0.0
                continue
            
            # Determine sign of gradient component i
            # If fi > f0, function increases in +i direction -> min is in -i direction (sign -1)
            # If fi < f0, function decreases in +i direction -> min is in +i direction (sign +1)
            # If fi == f0, gradient is likely 0 (sign 0)
            if fi > f0:
                grad_signs[i] = -1.0
            elif fi < f0:
                grad_signs[i] = 1.0
            else:
                grad_signs[i] = 0.0

        # Phase 2: Construct and Evaluate Optimal Vertex
        # For a linear function, the minimum on the box is at a vertex.
        # x_i = -5 if grad_i > 0 (we want to go against the increase)
        # x_i = +5 if grad_i < 0 (we want to go with the decrease)
        # x_i = 0 if grad_i == 0 (doesn't matter, pick center)
        
        x_vertex = np.zeros(dim)
        for i in range(dim):
            if grad_signs[i] > 0:
                x_vertex[i] = 5.0
            elif grad_signs[i] < 0:
                x_vertex[i] = -5.0
            else:
                x_vertex[i] = 0.0
        
        f_vertex = safe_call(x_vertex)
        
        if f_vertex is None:
            # If vertex evaluation failed, return best found so far
            return best_x, best_f

        # Phase 3: Polish with (1+1)-ES
        # The vertex should be the global minimum for a strictly linear function.
        # However, if there are slight non-linearities or if the gradient estimation
        # was noisy (unlikely with h=4.0 and finite precision), a local search
        # can refine the solution.
        
        # Initialize ES at the vertex
        x = x_vertex.copy()
        fx = f_vertex
        sigma = 0.1  # Small step size since we are likely at the optimum
        
        rng = np.random.default_rng(42)
        
        # Use remaining budget for polishing
        while calls_used < budget and time.time() < deadline:
            # Generate candidate
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5.0, 5.0)
            
            fy = safe_call(y)
            if fy is None:
                break
            
            # Adapt step size
            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.1  # Expand slightly on success
            else:
                sigma *= 0.9  # Shrink on failure
            
            # Early termination if sigma is very small
            if sigma < 1e-8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
