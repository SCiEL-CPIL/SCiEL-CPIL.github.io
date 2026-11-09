"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for Linear Slope function.
    Strategy:
    1. Estimate gradient using center and axis points.
    2. Predict the optimal vertex based on gradient signs.
    3. Evaluate predicted vertex.
    4. Refine with (1+1)-ES if necessary.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, keep 10s margin
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        # Step 1: Evaluate center point
        center = np.zeros(dim)
        f_center = func(center)
        calls_made += 1
        best_f = f_center
        best_x = center.copy()
        
        # Step 2: Use large steps (h=5) to evaluate at box bounds for each axis
        # This gives much better intermediate best values and identifies optimal vertex
        # For linear function, evaluating at -5 and +5 along each axis from center
        # directly reveals which bound is better for each coordinate
        
        # Evaluate at large offset to find which direction is better for each axis
        # Use h=5 so we hit the box boundaries
        h = 5.0
        optimal_vertex = np.zeros(dim)
        
        for i in range(dim):
            if calls_made >= budget:
                break
            
            # Point in positive direction (hits +5 bound)
            x_pos = center.copy()
            x_pos[i] += h
            x_pos = np.clip(x_pos, -5, 5)
            f_pos = func(x_pos)
            calls_made += 1
            
            # Point in negative direction (hits -5 bound)
            x_neg = center.copy()
            x_neg[i] -= h
            x_neg = np.clip(x_neg, -5, 5)
            f_neg = func(x_neg)
            calls_made += 1
            
            # Determine which bound is better for coordinate i
            if f_pos < f_neg:
                optimal_vertex[i] = 5.0
            else:
                optimal_vertex[i] = -5.0
            
            # Track best so far
            if f_pos < best_f:
                best_f = f_pos
                best_x = x_pos.copy()
            if f_neg < best_f:
                best_f = f_neg
                best_x = x_neg.copy()
        
        # Step 3: Evaluate predicted optimal vertex
        if calls_made < budget:
            f_pred = func(optimal_vertex)
            calls_made += 1
            if f_pred < best_f:
                best_f = f_pred
                best_x = optimal_vertex.copy()
        
        # Step 4: Refine with (1+1)-ES from best point found
        if best_x is None:
            best_x = center.copy()
            best_f = f_center
        
        x = best_x.copy()
        fx = best_f
        sigma = 0.5
        
        # Run ES for remaining budget
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
            else:
                sigma *= 0.9
                
            if sigma < 1e-6:
                break

        return x, fx
# EVOLVE-BLOCK-END
