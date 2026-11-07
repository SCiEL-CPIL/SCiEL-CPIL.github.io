import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: The limit is 300s. We stop at 290s to be safe.
        deadline = start_time + 290.0
        
        # The function is linear: f(x) = w^T x + b.
        # The minimum on the box [-5, 5]^d is at a vertex.
        # We determine the sign of each component of w to find the optimal vertex.
        
        # Start at the center
        x_center = np.zeros(self.dim)
        f_center = func(x_center)
        
        best_x = x_center.copy()
        best_f = f_center
        
        # Step size for probing the gradient
        # Since the domain is [-5, 5], a step of 1.0 is safe and distinct.
        # Using 1.0 ensures we are well within the box from the center.
        step = 1.0
        
        optimal_x = np.zeros(self.dim)
        
        for i in range(self.dim):
            if time.time() > deadline:
                break
                
            # Probe positive direction
            x_pos = x_center.copy()
            x_pos[i] += step
            
            # Probe negative direction
            x_neg = x_center.copy()
            x_neg[i] -= step
            
            f_pos = func(x_pos)
            f_neg = func(x_neg)
            
            # Update best
            if f_pos < best_f:
                best_f = f_pos
                best_x = x_pos.copy()
            if f_neg < best_f:
                best_f = f_neg
                best_x = x_neg.copy()
                
            # Determine optimal direction for dimension i
            # f(x) = w^T x + b
            # f(x + step*e_i) = f(x) + step * w_i
            # f(x - step*e_i) = f(x) - step * w_i
            # If w_i > 0, f_pos > f_neg. Function increases with x_i. Min at lower bound (-5).
            # If w_i < 0, f_pos < f_neg. Function decreases with x_i. Min at upper bound (5).
            
            if f_pos < f_neg:
                # w_i is negative
                optimal_x[i] = 5.0
            else:
                # w_i is positive (or zero)
                optimal_x[i] = -5.0
                
        # Evaluate the constructed optimal vertex
        if time.time() < deadline:
            f_opt = func(optimal_x)
            if f_opt < best_f:
                best_f = f_opt
                best_x = optimal_x.copy()
                
        # Optional: If we have budget left and time, we could do a few random restarts 
        # just in case the function wasn't perfectly linear, but the contract says it is f5.
        # Given the high score potential of the linear strategy, we stick to it.
        
        return best_x, best_f
