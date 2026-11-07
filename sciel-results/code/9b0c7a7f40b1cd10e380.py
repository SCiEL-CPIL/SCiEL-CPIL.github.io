"""
Strategy: Gradient-Descent with Finite Differences and Grid Initialization.

1. Initialization: Perform a coarse deterministic grid search (Halton or simple grid) 
   to identify promising basins. This ensures early progress by quickly finding 
   a "decent" point, which is crucial for AOCC.
2. Refinement: Use a gradient-based method. Since the Weierstrass function is 
   differentiable, we can estimate the gradient using central differences. 
   We then use an adaptive step size (similar to backtracking line search) to 
   descend efficiently. This is much more efficient than random walks in smooth 
   landscapes once we are in the basin of attraction.
3. Hybrid: If gradient descent stalls (small gradient or no improvement), we 
   perform a small local random perturbation to escape flat regions or numerical 
   noise, then resume gradient descent.
"""

import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Safety margin for time (evaluator kills at 330s, we stop at 295s)
        start_time = time.time()
        deadline = start_time + 295.0
        
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        def wrapped_func(x):
            nonlocal calls_made, best_f, best_x
            if time.time() > deadline:
                return best_f
            if calls_made >= budget:
                return best_f
            
            # Ensure input is a valid numpy array
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5, 5)
            
            try:
                f_val = func(x)
            except Exception:
                f_val = np.inf
            
            calls_made += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
                
            return f_val

        # --- Phase 1: Coarse Grid Search for Initialization ---
        # Weierstrass is multi-modal. A grid helps find the global basin or a good local one.
        # Let's use a small number of points, e.g., 3^dim or 4^dim? 
        # 4^5 = 1024 points is too many for 500 budget.
        # 3^5 = 243 points. This leaves ~250 for refinement.
        # Let's try a Halton sequence for better space-filling properties with fewer points.
        
        # We will spend 20% of the budget on initialization sampling.
        init_budget = int(budget * 0.2) 
        # But we want to be fast. Let's use a deterministic set of points.
        # Simple stratified sampling: divide [-4, 4] into 3 bins per dim -> 3^5 = 243 points.
        # 243 is close to 20% of 500 (100). Maybe 2^5 = 32 points for init?
        # Let's use 32 points for init (6.4% budget) to leave more for gradient descent.
        
        # Generate 32 points: 2 levels per dimension in [-4, 4]
        # Actually, let's use a Halton sequence for the first ~50 points to cover space well.
        
        def halton_sequence(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def halton(i, dim):
            return [halton_sequence(i, base) for base in range(2, dim + 2)]

        # Sample 50 points from Halton sequence in [-4, 4]^dim
        num_init_points = 50
        init_points = []
        for i in range(1, num_init_points + 1):
            h = halton(i, dim)
            # Map [0,1] to [-4,4]
            x = np.array([h[j] * 8.0 - 4.0 for j in range(dim)])
            init_points.append(x)
        
        # Evaluate all init points
        for x in init_points:
            if time.time() > deadline or calls_made >= budget:
                break
            wrapped_func(x)
            
        # If we ran out of budget during init, return best
        if calls_made >= budget:
            return best_x, best_f

        # --- Phase 2: Gradient Descent Refinement ---
        # Start from the best point found in Phase 1
        
        current_x = best_x.copy()
        current_f = best_f
        
        # Gradient estimation parameters
        # Central difference step size
        # Weierstrass has features at multiple scales. 
        # Start with a moderate epsilon.
        eps = 0.01
        
        # Learning rate / Step size for gradient descent
        # We will use a simple adaptive rule: if f decreases, keep step; if not, reduce.
        step_size = 0.5
        
        # Track improvement
        no_improve_count = 0
        max_no_improve = 20 # If no improvement for 20 steps, try a random kick
        
        # Main optimization loop
        while calls_made < budget and time.time() < deadline:
            
            # 1. Estimate Gradient
            # We need 2*dim evaluations for central differences.
            # If we are close to budget limit, we might not have enough for a full gradient step.
            # Check if we have at least 2*dim + 1 calls left? 
            # Actually, we just need to make sure we don't exceed budget.
            
            if calls_made + 2 * dim > budget:
                # Not enough budget for a full gradient step, do a simple random step or break
                # Let's just break to be safe, or do a single random perturbation if budget allows
                if calls_made < budget:
                    # Single random perturbation
                    x_new = np.clip(current_x + np.random.normal(0, step_size * 0.1, dim), -5, 5)
                    f_new = wrapped_func(x_new)
                    if f_new < current_f:
                        current_x = x_new
                        current_f = f_new
                break

            grad = np.zeros(dim)
            for i in range(dim):
                x_plus = current_x.copy()
                x_minus = current_x.copy()
                x_plus[i] += eps
                x_minus[i] -= eps
                
                # Clip to box
                x_plus = np.clip(x_plus, -5, 5)
                x_minus = np.clip(x_minus, -5, 5)
                
                f_plus = wrapped_func(x_plus)
                f_minus = wrapped_func(x_minus)
                
                # If we hit the boundary, the difference might be biased, but it's okay for now.
                grad[i] = (f_plus - f_minus) / (2 * eps)
                
                # Early exit if time/budget exceeded during gradient calc
                if time.time() > deadline or calls_made >= budget:
                    break
            
            if time.time() > deadline or calls_made >= budget:
                break

            # 2. Calculate Step
            # Normalize gradient? Weierstrass gradients can be small or large.
            # Use a fixed step size scaled by the norm of the gradient?
            # Or just use the gradient direction with a fixed step size?
            # Let's use: x_new = x - step_size * grad / ||grad|| * some_scale?
            # Actually, standard GD: x_new = x - step_size * grad.
            # But if grad is very small, step is tiny. If grad is large, step is huge.
            # Let's normalize the gradient direction and use a fixed step size.
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-8:
                # Gradient is zero, we are at a critical point.
                # Try a random kick to escape
                x_new = np.clip(current_x + np.random.normal(0, 0.1, dim), -5, 5)
            else:
                # Direction is -grad / ||grad||
                direction = -grad / grad_norm
                # Step size: let's try a fixed step size relative to the domain
                # 0.1 is a reasonable step in a 10-unit wide box.
                x_new = np.clip(current_x + direction * step_size, -5, 5)

            # 3. Evaluate
            f_new = wrapped_func(x_new)
            
            # 4. Update
            if f_new < current_f:
                # Success
                current_x = x_new
                current_f = f_new
                no_improve_count = 0
                # Maybe increase step size slightly?
                # step_size *= 1.1
            else:
                # Failure
                no_improve_count += 1
                # Reduce step size
                step_size *= 0.7
                # If step size becomes too small, try a random kick
                if step_size < 0.01:
                    x_new = np.clip(current_x + np.random.normal(0, 0.5, dim), -5, 5)
                    f_new = wrapped_func(x_new)
                    if f_new < current_f:
                        current_x = x_new
                        current_f = f_new
                        step_size = 0.5 # Reset step size
                        no_improve_count = 0
                    else:
                        # Stuck? Try a larger random kick
                        x_new = np.clip(current_x + np.random.normal(0, 1.0, dim), -5, 5)
                        f_new = wrapped_func(x_new)
                        if f_new < current_f:
                            current_x = x_new
                            current_f = f_new
                            step_size = 0.5
                            no_improve_count = 0
                        else:
                            # Give up on this basin, move to next best from init?
                            # For simplicity, just break and rely on the global best tracked.
                            # But we can try a few more random kicks.
                            pass

            # If we haven't improved for a long time, maybe restart from a different init point?
            # We didn't store the second best. Let's just continue with random kicks if stuck.
            if no_improve_count > 50:
                # Do a few random kicks and then break to save time/budget
                for _ in range(5):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    x_new = np.clip(current_x + np.random.normal(0, 1.0, dim), -5, 5)
                    f_new = wrapped_func(x_new)
                    if f_new < current_f:
                        current_x = x_new
                        current_f = f_new
                        no_improve_count = 0
                        step_size = 0.5
                        break
                else:
                    break

        return best_x, best_f
