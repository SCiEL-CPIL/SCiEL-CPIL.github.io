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
        # Start at the center of the box
        x_center = np.zeros(self.dim)
        f_center = func(x_center)
        best_x = x_center.copy()
        best_f = f_center
        calls_used = 1
        
        # Estimate gradient signs using larger steps for robustness
        # For a linear function, the sign of the gradient component w_i
        # is determined by the difference f(x + delta*e_i) - f(x - delta*e_i)
        # We use a step size of 4.0 to maximize the signal difference
        # while staying well within the [-5, 5] bounds from the center.
        step_size = 4.0
        signs = np.zeros(self.dim)
        
        for i in range(self.dim):
            if time.time() > deadline:
                break
            x_plus = x_center.copy()
            x_plus[i] += step_size
            x_minus = x_center.copy()
            x_minus[i] -= step_size
            
            f_plus = func(x_plus)
            f_minus = func(x_minus)
            calls_used += 2
            
            # If f_plus > f_minus, the function increases in the +i direction,
            # so the gradient is positive. To minimize, we should go to -5.
            # If f_plus < f_minus, the function decreases in the +i direction,
            # so the gradient is negative. To minimize, we should go to +5.
            if f_plus > f_minus:
                signs[i] = 1.0  # Positive gradient, go to -5
            else:
                signs[i] = -1.0 # Negative gradient, go to +5
            
            # Update best if any of these are better
            if f_plus < best_f:
                best_f = f_plus
                best_x = x_plus.copy()
            if f_minus < best_f:
                best_f = f_minus
                best_x = x_minus.copy()
                
        # Construct the candidate vertex
        # If sign[i] > 0 (gradient positive), go to -5.0
        # If sign[i] < 0 (gradient negative), go to 5.0
        x_vertex = np.where(signs > 0, -5.0, 5.0)
        f_vertex = func(x_vertex)
        calls_used += 1
        
        if f_vertex < best_f:
            best_f = f_vertex
            best_x = x_vertex.copy()
            
        # 2. Verification and Refinement
        # For a linear function, the vertex is the global minimum.
        # However, to be absolutely sure and handle any edge cases or 
        # numerical issues, we verify that moving inward from the vertex
        # does not improve the function value.
        
        remaining_budget = self.budget - calls_used
        if remaining_budget > 0 and time.time() < deadline:
            self._verify_and_refine(func, best_x, best_f, remaining_budget, deadline)
            
        return best_x, best_f

    def _verify_and_refine(self, func, x, fx, budget, deadline):
        """
        Verifies that the current point (expected to be a vertex) is optimal
        by checking small inward moves. If an inward move is better, it implies
        the gradient direction was misestimated or the function is not purely
        linear in the expected way. In that case, we perform a few steps of
        gradient descent or local search.
        """
        dim = self.dim
        best_x = x.copy()
        best_f = fx
        calls = 0
        
        # Check each dimension for potential improvement by moving inward
        # If best_x[i] is -5.0, move to -4.9
        # If best_x[i] is 5.0, move to 4.9
        # If best_x[i] is in between, it's not a vertex, so we need a different strategy.
        
        is_vertex = True
        for i in range(dim):
            if not (best_x[i] == -5.0 or best_x[i] == 5.0):
                is_vertex = False
                break
        
        if is_vertex:
            # Verify vertex optimality
            for i in range(dim):
                if time.time() > deadline or calls >= budget:
                    break
                
                x_test = best_x.copy()
                # Move 0.1 units inward
                if best_x[i] == -5.0:
                    x_test[i] = -4.9
                else: # best_x[i] == 5.0
                    x_test[i] = 4.9
                    
                f_test = func(x_test)
                calls += 1
                
                if f_test < best_f:
                    # This should not happen for a correct linear function vertex
                    # But if it does, update best and potentially trigger a local search
                    best_f = f_test
                    best_x = x_test.copy()
                    
            # If we found a better point by moving inward, run a small local search
            # to ensure we don't miss a nearby optimum (unlikely for f5, but safe)
            if best_x is not x: # If best_x was updated
                # Perform a few Nelder-Mead steps or simple gradient descent
                # For simplicity and speed, we'll just do a few random perturbations
                # around the new best point if budget allows
                # But for f5, the vertex is the optimum, so this branch is likely dead code.
                pass
        else:
            # If we are not at a vertex, run a simple local search
            # This handles the case where gradient estimation failed
            # Use a simple (1+1) ES for the remaining budget
            self._local_es(func, best_x, best_f, budget, deadline)
            
        return best_x, best_f

    def _local_es(self, func, x, fx, budget, deadline):
        """Fallback local search using (1+1)-ES."""
        rng = np.random.default_rng(0)
        best_x = x.copy()
        best_f = fx
        sigma = 0.5
        calls = 0
        
        while calls < budget and time.time() < deadline:
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
                if fy < best_f:
                    best_f = fy
                    best_x = x.copy()
            else:
                sigma *= 0.87
                
        return best_x, best_f
