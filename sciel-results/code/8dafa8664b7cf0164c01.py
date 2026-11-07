"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Linear model extraction to jump to the optimal vertex, followed by a safeguard local search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: evaluator kills at 330s, we aim for 290s
        deadline = start_time + 290.0
        
        best_x = None
        best_f = np.inf
        
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Determine the sign of the gradient components
        # For a linear function f(x) = v^T x + c, the minimum is at the vertex
        # x_i = -5 if v_i > 0, else 5.
        
        x_center = np.zeros(self.dim)
        f_center = func(x_center)
        update_best(x_center, f_center)
        
        if time.time() > deadline:
            return best_x, best_f

        # Probe in each dimension to determine the slope sign
        # We use a step size of 1.0 for robustness against numerical noise, 
        # though for linear functions any h > 0 works.
        h = 1.0
        slopes = np.zeros(self.dim)
        
        for i in range(self.dim):
            if time.time() > deadline:
                break
            
            x_pos = x_center.copy()
            x_pos[i] += h
            f_pos = func(x_pos)
            update_best(x_pos, f_pos)
            
            x_neg = x_center.copy()
            x_neg[i] -= h
            f_neg = func(x_neg)
            update_best(x_neg, f_neg)
            
            # Central difference for slope
            slopes[i] = (f_pos - f_neg) / (2 * h)
            
            # Check deadline inside loop
            if time.time() > deadline:
                break

        # Construct the candidate vertex
        x_vertex = np.zeros(self.dim)
        for i in range(self.dim):
            if slopes[i] > 0:
                x_vertex[i] = -5.0
            else:
                x_vertex[i] = 5.0
        
        # Evaluate the vertex
        f_vertex = func(x_vertex)
        update_best(x_vertex, f_vertex)
        
        # Phase 2: Safeguard
        # If the function is strictly linear, x_vertex is the global minimum.
        # However, if there is any non-linearity or if the slope estimation was
        # flawed (e.g., due to floating point issues with very small slopes),
        # we might be at a suboptimal vertex or the function might have a minimum
        # inside the box (though unlikely for "Linear slope").
        
        # We check if the vertex is actually better than the center.
        # If not, we might have made a mistake or the function is flat.
        # In either case, we can try a local search from the best point found so far.
        
        # Use remaining budget for a robust local search (Nelder-Mead or similar)
        # to ensure we don't miss a better point if the linear assumption is slightly off.
        # However, for a linear function, moving away from the vertex is bad.
        # So we only run a local search if the vertex is NOT significantly better than the center,
        # or if we suspect non-linearity.
        
        # Actually, a simple check: if f_vertex is not the best, something is wrong.
        # But for linear functions, f_vertex should be the best.
        
        # Let's use the remaining budget for a very cautious local search 
        # starting from the best point found (likely the vertex).
        # We use a small step size to avoid moving away from the optimum if it's linear.
        
        x_current = best_x.copy()
        f_current = best_f
        
        # If we have calls left, we can do a few random perturbations to check stability.
        # But for linear, any perturbation increases f.
        # So we just return.
        
        # However, to be safe against the "hidden" nature, let's do a few steps of 
        # a simple local search with a tiny step size.
        
        calls_used = 1 + 2 * self.dim + 1
        remaining_calls = self.budget - calls_used
        
        if remaining_calls > 0 and time.time() < deadline:
            # If the slope was very small, the vertex choice might be arbitrary.
            # A local search can help if the function is not perfectly linear.
            # We use a very small step size to stay near the vertex.
            sigma = 0.1
            rng = np.random.default_rng(42)
            
            for _ in range(remaining_calls):
                if time.time() > deadline:
                    break
                
                # Generate a small perturbation
                # If the function is linear, this will likely increase f.
                # If it decreases f, we move.
                y = x_current + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5.0, 5.0)
                
                fy = func(y)
                update_best(y, fy)
                
                if fy < f_current:
                    x_current = y
                    f_current = fy
                    # If we found a better point, it suggests non-linearity or error.
                    # Keep searching.
                else:
                    # If we didn't find a better point, shrink sigma to explore finer details
                    # or just stop if sigma is too small.
                    sigma *= 0.9
                    if sigma < 1e-6:
                        # If sigma is very small and we haven't improved, 
                        # we are likely at the optimum (or a local min if non-linear).
                        # For linear, we are at the vertex, so we are done.
                        # We can break early to save time, but we have time.
                        # Let's just continue with tiny steps to be sure.
                        pass

        return best_x, best_f
# EVOLVE-BLOCK-END
