"""Optimizer for BBOB f17 (Schaffer's F7) using Derivative-Free Broyden's Method with Trust-Region-like line search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Derivative-Free Broyden's Method (Quasi-Newton) for BBOB f17."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 300s limit, leave 15s margin

        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_fx = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            fx = float(func(x))
            calls_made += 1
            return fx

        def time_ok():
            return time.time() < deadline and calls_made < budget

        def estimate_gradient(x, h=1e-5):
            """Estimate gradient using forward finite differences."""
            grad = np.zeros(d)
            base_fx = call_func(x)
            for i in range(d):
                x_pert = x.copy()
                x_pert[i] += h
                x_pert = np.clip(x_pert, -5, 5)
                fx_pert = call_func(x_pert)
                # Handle clipping effects by adjusting h if x was at boundary
                actual_h = x_pert[i] - x[i]
                if abs(actual_h) < 1e-12:
                    grad[i] = 0.0
                else:
                    grad[i] = (fx_pert - base_fx) / actual_h
            return grad

        def broyden_optimize(x0, max_calls):
            """
            Derivative-Free Broyden's Method.
            Uses finite differences for gradients.
            Maintains an approximation of the Hessian (B).
            """
            nonlocal best_x, best_fx, calls_made
            
            x = x0.copy()
            fx = call_func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

            if calls_made >= max_calls:
                return best_x, best_fx

            g = estimate_gradient(x)
            if calls_made >= max_calls:
                return best_x, best_fx

            # Initialize Hessian approximation B as Identity
            B = np.eye(d)
            
            # Line search parameters
            alpha = 1.0
            rho = 0.5
            max_ls_iter = 20

            while calls_made < max_calls and time_ok():
                # Solve for step direction: p = -B^{-1} g
                try:
                    p = np.linalg.solve(B, -g)
                except np.linalg.LinAlgError:
                    p = -g

                # Ensure step is not too large (simple trust region constraint)
                norm_p = np.linalg.norm(p)
                if norm_p > 5.0:
                    p = p / norm_p * 5.0
                if norm_p < 1e-10:
                    break

                # Line search
                ls_iter = 0
                found_step = False
                current_alpha = alpha
                while ls_iter < max_ls_iter:
                    x_trial = x + current_alpha * p
                    x_trial = np.clip(x_trial, -5, 5)
                    
                    # Check if we are making progress
                    fx_trial = call_func(x_trial)
                    if fx_trial < best_fx:
                        best_fx = fx_trial
                        best_x = x_trial.copy()
                    
                    if fx_trial < fx:
                        # Successful step
                        x_new = x_trial
                        fx_new = fx_trial
                        found_step = True
                        break
                    else:
                        current_alpha *= rho
                        if current_alpha < 1e-8:
                            break
                    ls_iter += 1

                if not found_step:
                    # If no step found, try a random perturbation or break
                    # To avoid infinite loop, break or try small random step
                    if calls_made < max_calls:
                        x_new = x + rng.normal(0, 0.1, d)
                        x_new = np.clip(x_new, -5, 5)
                        fx_new = call_func(x_new)
                        if fx_new < best_fx:
                            best_fx = fx_new
                            best_x = x_new.copy()
                    else:
                        break
                    # If this random step also failed to improve, we might be stuck
                    # For simplicity, we break if line search failed completely
                    if ls_iter == 0: 
                        break
                    # If we took a random step, update B is not standard, so we restart B or break
                    # Let's break to save budget for other starts
                    break

                # Calculate new gradient
                if calls_made >= max_calls:
                    break
                g_new = estimate_gradient(x_new)
                if calls_made >= max_calls:
                    break

                # Update B using Broyden's formula
                s = x_new - x
                y = g_new - g
                s_norm_sq = np.dot(s, s)
                if s_norm_sq > 1e-12:
                    # Broyden's Good BFGS update
                    B = B + np.outer(y - B @ s, s) / s_norm_sq
                    # Symmetrize to keep it a Hessian approximation (optional but helps stability)
                    # B = (B + B.T) / 2 
                    # Actually, Broyden's is not symmetric. For optimization, we want Hessian.
                    # Using BFGS update on the gradient is equivalent to updating the Hessian approximation.
                    # The formula above is for Jacobian of F(x). Since we are minimizing f, F(x)=grad(f).
                    # So B approximates Hessian.
                    # To ensure positive definiteness, we can project or just use it.
                    # Let's symmetrize to act like a Hessian
                    B = (B + B.T) / 2

                x = x_new
                fx = fx_new
                g = g_new

                # Check convergence
                if np.linalg.norm(g) < 1e-8:
                    break

            return best_x, best_fx

        # Phase 1: Multi-start Broyden's Method
        # We use multiple starting points to ensure global coverage.
        # Budget allocation: 5 starts, ~80 calls each initially.
        
        starting_points = [
            np.zeros(d),
            rng.uniform(-4, 4, d),
            rng.uniform(-4, 4, d),
            np.array([4.0] * d),
            np.array([-4.0] * d)
        ]
        
        # Shuffle starting points to add diversity
        indices = rng.permutation(len(starting_points))
        starting_points = [starting_points[i] for i in indices]

        calls_per_start = 80
        
        for i, x0 in enumerate(starting_points):
            if not time_ok():
                break
            
            x0 = np.clip(x0, -5, 5)
            max_calls_for_this = calls_made + calls_per_start
            
            # If we have used most of the budget, allocate less
            if calls_made + calls_per_start > budget:
                max_calls_for_this = budget
            
            broyden_optimize(x0, max_calls_for_this)
            
            # If we improved significantly, we might want to run a longer refinement
            # But for now, let's just do the fixed number of starts.
            
        # Phase 2: Final Refinement with remaining budget
        if time_ok() and calls_made < budget and best_x is not None:
            # Use remaining budget for a final polish on the best solution found
            remaining = budget - calls_made
            if remaining > 10:
                # Run Broyden again from the best point with the rest of the budget
                broyden_optimize(best_x.copy(), budget)

        return best_x, best_fx
# EVOLVE-BLOCK-END
