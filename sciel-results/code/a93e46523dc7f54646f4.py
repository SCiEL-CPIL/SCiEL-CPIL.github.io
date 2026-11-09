import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # 1. Evaluate center
        x_center = np.zeros(dim)
        f_center = func(x_center)
        best_x = x_center.copy()
        best_f = f_center
        calls_used = 1
        
        # 2. Determine the sign of the gradient for each coordinate
        # Strategy: Sequential coordinate descent / greedy fixing.
        # For a linear function f(x) = w^T x + b, the minimum over [-5, 5]^d is at a vertex.
        # The coordinate i should be set to -5 if w_i > 0, and to 5 if w_i < 0.
        # We can determine the sign of w_i by evaluating f at x where x_i = 5 and x_j = 0 for j != i.
        # However, a more efficient approach for AOCC is to greedily construct the vertex.
        
        # Let's stick to the robust sign-determination method from Inspiration 1/2 but optimized for early progress.
        # We will evaluate f(5*e_i) for each i.
        # If f(5*e_i) < f(0), then w_i < 0, so optimal x_i = 5.
        # If f(5*e_i) > f(0), then w_i > 0, so optimal x_i = -5.
        
        # We can update best_x as we go by setting the determined coordinates to their optimal values
        # and keeping the rest at 0. This provides intermediate progress.
        
        x_opt = np.zeros(dim)
        signs = np.zeros(dim)
        
        for i in range(dim):
            if time.time() > deadline or calls_used >= budget:
                break
                
            # Evaluate at 5 * e_i
            x_probe = np.zeros(dim)
            x_probe[i] = 5.0
            f_probe = func(x_probe)
            calls_used += 1
            
            # Update best if this probe is better
            if f_probe < best_f:
                best_f = f_probe
                best_x = x_probe.copy()
            
            # Determine sign
            if f_probe < f_center:
                signs[i] = -1.0  # w_i < 0, go to +5
                x_opt[i] = 5.0
            else:
                signs[i] = 1.0   # w_i > 0, go to -5
                x_opt[i] = -5.0
            
            # Construct a partial vertex: set determined coords to optimal, others to 0.
            # This is a valid point in the box.
            # Is it better than the previous best?
            # Actually, for linear functions, moving towards the true vertex monotonically improves the objective
            # IF we move in the correct direction.
            # Let's evaluate the partial vertex to get better intermediate scores.
            # Note: x_opt currently has determined coords set, others 0.
            
            # But wait, evaluating x_opt at every step costs extra calls.
            # The probe x_probe IS the point where only coord i is 5, others 0.
            # If we determined coord i should be 5, then x_probe is the partial vertex for coord i=5.
            # If we determined coord i should be -5, then x_probe (at 5) is the worst direction.
            # We should evaluate the point where coord i is -5 if we determined -5.
            
            # Let's refine:
            # We know the optimal value for coord i is either 5 or -5.
            # Let's evaluate the point where coord i is at its optimal value, and other *determined* coords are at their optimal values?
            # No, that's complex.
            
            # Simpler: Just determine all signs first (5 calls), then evaluate the full vertex (1 call).
            # Total 6 calls.
            # But we can do better for AOCC by evaluating partial vertices.
            
            # Let's try this:
            # After determining sign for i, let's evaluate the point where:
            # - coords 0..i-1 are at their determined optimal values
            # - coord i is at its determined optimal value
            # - coords i+1..dim-1 are 0
            # This point is a vertex of the sub-box defined by the first i+1 dimensions.
            # For a linear function, this point is closer to the global optimum than the previous partial vertex.
            
            # However, we already evaluated x_probe (which is e_i * 5).
            # If optimal is 5, x_probe is the new partial vertex.
            # If optimal is -5, we need to evaluate -5 * e_i + sum_{j<i} opt_j * e_j.
            
            # To keep it simple and robust, let's just determine all signs, then evaluate the full vertex.
            # The intermediate probes (5*e_i) might be better than center, giving some AOCC gain.
            
            # Let's just finish the loop to get all signs.
            
        # After the loop, we have signs for all dimensions (unless broke early).
        # Construct the full vertex.
        x_vertex = np.zeros(dim)
        for i in range(dim):
            if signs[i] == 0:
                # If we broke early, we don't know the sign. Default to 0? Or 5?
                # With budget 500 and dim 5, we won't break early due to budget.
                # Only due to time.
                x_vertex[i] = 0.0
            else:
                x_vertex[i] = 5.0 if signs[i] < 0 else -5.0
                
        # Evaluate the full vertex
        if calls_used < budget and time.time() < deadline:
            f_vertex = func(x_vertex)
            calls_used += 1
            
            if f_vertex < best_f:
                best_f = f_vertex
                best_x = x_vertex.copy()
        
        # 3. Refinement / Verification
        # Since it's a linear function, the vertex is the global minimum.
        # We can use remaining budget to verify or just return.
        # Given the high score of the inspirations, the vertex identification is the key.
        # We can add a lightweight check to ensure we didn't miss a sign due to numerical issues (unlikely here).
        
        # Let's do a quick check: evaluate a point slightly inside the vertex to see if it's worse.
        # If it's worse, we are at the optimum.
        # Actually, for AOCC, once we hit the optimum, n_i becomes small (good).
        # The vertex evaluation is the 6th call.
        # Calls 1-5 are probes. Call 6 is the vertex.
        # This is very efficient.
        
        return best_x, best_f
