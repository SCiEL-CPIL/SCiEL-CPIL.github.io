import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        # 1. Vertex Jump Strategy for Linear Functions
        x_center = np.zeros(self.dim)
        f_center = func(x_center)
        best_x = x_center.copy()
        best_f = f_center
        
        # Estimate gradient signs using larger steps for robustness
        signs = np.zeros(self.dim)
        calls_used = 1
        
        for i in range(self.dim):
            if time.time() > deadline:
                break
            x_plus = x_center.copy()
            x_plus[i] += 2.0
            x_minus = x_center.copy()
            x_minus[i] -= 2.0
            
            f_plus = func(x_plus)
            f_minus = func(x_minus)
            calls_used += 2
            
            # If f_plus > f_minus, gradient in dim i is positive, so go to -5
            # If f_plus < f_minus, gradient in dim i is negative, so go to +5
            if f_plus > f_minus:
                signs[i] = 1.0
            else:
                signs[i] = -1.0
            
            # Update best
            if f_plus < best_f:
                best_f = f_plus
                best_x = x_plus.copy()
            if f_minus < best_f:
                best_f = f_minus
                best_x = x_minus.copy()
                
        # Construct vertex: if sign[i] > 0, go to -5 (minimize positive gradient)
        # if sign[i] < 0, go to +5 (minimize negative gradient)
        x_vertex = np.where(signs > 0, -5.0, 5.0)
        f_vertex = func(x_vertex)
        calls_used += 1
        
        if f_vertex < best_f:
            best_f = f_vertex
            best_x = x_vertex.copy()
            
        # 2. For linear functions, vertex is optimal. Use remaining budget
        # for a lightweight verification/refinement only if needed.
        remaining_budget = self.budget - calls_used
        
        if remaining_budget > 10 and time.time() < deadline:
            self._lightweight_refine(func, best_x, best_f, remaining_budget, deadline)
            
        return best_x, best_f

    def _lightweight_refine(self, func, x, fx, budget, deadline):
        """Lightweight refinement for linear functions: verify vertex is optimal."""
        dim = self.dim
        best_x = x.copy()
        best_f = fx
        calls = 0
        
        # Check if we're at a boundary and verify it's truly optimal
        # by trying small moves inward
        for i in range(dim):
            if time.time() > deadline or calls >= budget:
                break
            
            x_test = best_x.copy()
            if best_x[i] == -5.0:
                x_test[i] = -4.9
            elif best_x[i] == 5.0:
                x_test[i] = 4.9
            else:
                continue
            
            f_test = func(x_test)
            calls += 1
            
            if f_test < best_f:
                best_f = f_test
                best_x = x_test.copy()
        
        return best_x, best_f
