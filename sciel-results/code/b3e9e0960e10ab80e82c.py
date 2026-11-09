"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Physics-based optimizer: Damped Gradient Descent with Finite Differences."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        best_x = np.zeros(n)
        best_f = func(best_x)
        calls_used = 1

        def check_deadline():
            return time.time() >= deadline

        def do_call(x):
            nonlocal calls_used
            fx = func(x)
            calls_used += 1
            return fx

        # Finite difference gradient estimator
        # Uses central differences for better accuracy, 2n calls per gradient
        # To save budget, we use forward differences (n+1 calls) but with a small step
        # Actually, for Rosenbrock, central is much better for the valley.
        # Let's use central differences: 2n calls.
        # Budget 500. 500 / 10 = 50 steps. Might be too few to converge from scratch.
        # Let's use a hybrid: 
        # 1. Initial random search to get close? No, gradient is better if we are in basin.
        # Rosenbrock basin is wide. 
        # Strategy: 
        #   - Start at 0.
        #   - Compute gradient.
        #   - Take step.
        #   - If stuck (no improvement), perturb and restart gradient.
        
        # Since we have 500 calls and n=5:
        # Central diff: 10 calls per grad.
        # 500/10 = 50 gradient steps.
        # Is 50 steps enough? 
        # Rosenbrock convergence is linear but slow in valley.
        # However, with adaptive step size (line search or backtracking), it can be fast.
        
        # Let's implement a simple gradient descent with backtracking line search.
        # Each line search step costs O(1) function calls (geometric reduction).
        # So cost per iteration = 2n (grad) + k (line search).
        # If k is small (2-5), total cost ~ 12-15 calls/iter.
        # 500 / 15 ~ 33 iterations.
        
        # Alternative: Use fewer gradient evaluations.
        # We can reuse the gradient if the step is small? No, gradient changes.
        
        # Let's try a different angle: 
        # The current program uses ES which is "blind". 
        # Gradient is "informed". 
        # Even with limited steps, informed steps are better.
        
        # To maximize steps:
        # Use forward differences: n+1 calls.
        # Step cost: (n+1) + line_search.
        # 500 / (6+3) ~ 55 iterations.
        
        # Let's implement:
        # 1. Gradient via forward differences (step size h=1e-4).
        # 2. Backtracking line search.
        # 3. If stuck, add noise and restart.

        h = 1e-4
        x = np.zeros(n)
        f = do_call(x)
        best_x = x.copy()
        best_f = f
        
        # Main loop
        while calls_used < budget and not check_deadline():
            # 1. Estimate Gradient
            if calls_used + n + 1 > budget:
                break
            
            grad = np.zeros(n)
            f0 = f
            for i in range(n):
                x_pert = x.copy()
                x_pert[i] += h
                x_pert = np.clip(x_pert, -5, 5)
                fp = do_call(x_pert)
                grad[i] = (fp - f0) / h
                if calls_used >= budget:
                    break
            
            if calls_used >= budget:
                break
                
            # 2. Check for stagnation
            # If gradient is very small, we are at a minimum (or saddle).
            # Rosenbrock has no saddles in the basin, so it's the min.
            grad_norm = np.linalg.norm(grad)
            
            if grad_norm < 1e-6:
                # Converged
                break

            # 3. Determine Step Size
            # Initial step size alpha
            # Use Armijo backtracking
            alpha = 1.0
            max_alpha = 10.0
            min_alpha = 1e-6
            c = 1e-4  # Sufficient decrease
            rho = 0.5  # Shrink factor
            
            # Ensure we don't exceed budget for line search
            # Max 20 line search steps
            found_step = False
            for _ in range(20):
                if calls_used >= budget:
                    break
                
                x_new = x - alpha * grad
                x_new = np.clip(x_new, -5, 5)
                
                # If x_new is same as x (due to clipping or tiny step), try smaller alpha
                if np.allclose(x_new, x, atol=1e-8):
                    alpha *= rho
                    continue
                
                f_new = do_call(x_new)
                
                # Armijo condition
                if f_new <= f0 - c * alpha * grad_norm**2:
                    found_step = True
                    x = x_new
                    f = f_new
                    break
                else:
                    alpha *= rho
            
            if not found_step:
                # If line search failed, try a random step to escape
                # Or just accept the best from line search? 
                # Let's accept the last f_new if it was better than f0?
                # Actually, if no step satisfied Armijo, we might be stuck.
                # Let's try a random perturbation.
                x = x + 0.1 * rng.standard_normal(n)
                x = np.clip(x, -5, 5)
                f = do_call(x)
                # Reset alpha for next iteration
                alpha = 1.0

            # Update best
            if f < best_f:
                best_f = f
                best_x = x.copy()
            
            # Stagnation check: if we haven't improved in last 50 calls, restart
            # We don't track history explicitly, but we can check if best_f hasn't changed
            # Simple heuristic: if grad_norm is small but f is not close to 0 (unknown), 
            # we might be stuck in a flat area. 
            # For Rosenbrock, this is unlikely unless we are at the min.
            
            # To handle potential local minima (though R is unimodal), 
            # if we take many steps without improvement, add noise.
            # We can track steps_since_improvement.
            # Let's add a simple restart logic:
            # If we have used 50% of budget and haven't improved in last 20% of budget, restart from best_x with noise.
            
            # For now, rely on the gradient flow.
            # If we are at the minimum, grad_norm ~ 0 and we break.
            
        # Final polish: 
        # If we have budget left, do a few random local steps around best_x
        if calls_used < budget - 10 and not check_deadline():
            # Simple random hill climbing around best
            x = best_x.copy()
            f = best_f
            sigma = 0.1
            for _ in range(10):
                if calls_used >= budget:
                    break
                y = x + sigma * rng.standard_normal(n)
                y = np.clip(y, -5, 5)
                fy = do_call(y)
                if fy < f:
                    x = y
                    f = fy
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                else:
                    sigma *= 0.5

        return best_x, best_f
# EVOLVE-BLOCK-END
