"""Rewritten optimizer for BBOB f23 (Katsuura) using Powell's Method."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Powell's Method for Katsuura."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        B = self.budget
        
        # Initial point
        x = np.zeros(d)
        fx = func(x)
        calls = 1
        
        # Initial directions (standard basis)
        directions = np.eye(d)
        
        # Powell's Method
        # We will iterate until budget is exhausted or convergence.
        
        # To use the budget effectively, we need to manage line search evaluations.
        # A line search in 1D might take ~10-20 evaluations.
        # One Powell iteration takes d line searches + 1 extra = ~d+1 line searches.
        # Total evals per iteration ~ (d+1) * 15.
        # For d=5, ~90 evals per iteration.
        # 500 / 90 ~ 5-6 iterations.
        
        # We need a robust line search.
        # Let's use a simple bounded line search with golden section or similar.
        
        def line_search(func, x, direction, f_current, calls_ref):
            # Minimize f(x + t * direction) for t in [-10, 10] (clipped to box)
            # We use a simple golden section search or parabolic.
            # To be safe and simple, let's use a small grid + refinement?
            # No, golden section is better.
            
            # Determine bounds for t
            # x + t * dir must be in [-5, 5]
            # For each component i: -5 <= x[i] + t * dir[i] <= 5
            # If dir[i] > 0: t <= (5 - x[i]) / dir[i], t >= (-5 - x[i]) / dir[i]
            # If dir[i] < 0: t <= (-5 - x[i]) / dir[i], t >= (5 - x[i]) / dir[i]
            
            t_min = np.inf
            t_max = -np.inf
            
            for i in range(d):
                if abs(direction[i]) < 1e-10:
                    continue
                if direction[i] > 0:
                    t_min = max(t_min, (-5 - x[i]) / direction[i])
                    t_max = min(t_max, (5 - x[i]) / direction[i])
                else:
                    t_min = max(t_min, (5 - x[i]) / direction[i])
                    t_max = min(t_max, (-5 - x[i]) / direction[i])
            
            if t_min > t_max:
                # Infeasible direction? Should not happen if x is in box.
                return x, f_current, 0

            # Golden Section Search
            # We want to minimize f(t)
            # We need to handle the function call counting carefully.
            
            # Initial interval [t_min, t_max]
            # But we might want to restrict the search to a reasonable range to avoid noise?
            # Let's assume the minimum is within this range.
            
            phi = (np.sqrt(5) - 1) / 2
            inv_phi = 1 / phi
            inv_phi2 = inv_phi * inv_phi
            
            a = t_min
            b = t_max
            
            # If the interval is too large, we might miss the minimum if it's local?
            # Katsuura is unimodal-ish.
            
            c = b - inv_phi * (b - a)
            dd = a + inv_phi * (b - a)
            
            fc = func(x + c * direction)
            calls_ref[0] += 1
            fd = func(x + dd * direction)
            calls_ref[0] += 1
            
            # Iterate
            max_iters = 20 # Limit line search depth to save budget
            for _ in range(max_iters):
                if calls_ref[0] >= B:
                    break
                if fd < fc:
                    a = c
                    c = dd
                    fc = fd
                    b = b
                    dd = a + inv_phi * (b - a)
                    fd = func(x + dd * direction)
                    calls_ref[0] += 1
                else:
                    b = dd
                    dd = c
                    fd = fc
                    c = b - inv_phi * (b - a)
                    fc = func(x + c * direction)
                    calls_ref[0] += 1
                
                if abs(b - a) < 1e-6:
                    break
            
            # Choose best of c and dd
            if fc < fd:
                t_best = c
                f_best = fc
            else:
                t_best = dd
                f_best = fd
            
            x_new = x + t_best * direction
            return x_new, f_best, 2 # Approx cost

        calls_ref = [calls]
        best_x = x.copy()
        best_f = fx
        
        # Main Powell Loop
        # We repeat until budget is low.
        
        # To improve robustness, we can add a "restart" if we don't improve much.
        
        for _ in range(50): # Max outer iterations
            if calls_ref[0] >= B - 50: # Reserve budget
                break
            
            # Store old x
            x_old = x.copy()
            f_old = fx
            
            # Minimize along each direction
            f_min = np.inf
            min_idx = 0
            
            for i in range(d):
                if calls_ref[0] >= B - 20:
                    break
                
                x, fx, _ = line_search(func, x, directions[i], fx, calls_ref)
                
                if fx < f_min:
                    f_min = fx
                    min_idx = i
            
            # Check for convergence
            if np.linalg.norm(x - x_old) < 1e-5 and abs(fx - f_old) < 1e-5:
                # Try to restart with a smaller step or break?
                # For now, break
                break
            
            # Update directions
            # Replace the direction with the largest improvement with the new direction (x - x_old)
            # Standard Powell: replace direction that gave the largest reduction.
            
            # Calculate improvement for each direction? We didn't store them individually.
            # We only stored the total.
            # Let's refine: store improvements.
            
            # Re-run with storage? No, too expensive.
            # Standard Powell replaces the direction corresponding to the largest drop.
            # We didn't track which direction gave the largest drop.
            # We can approximate by checking the final x vs x_old.
            
            # Actually, let's just keep the directions fixed (Coordinate Descent) if we don't track?
            # No, Powell's power is in the new direction.
            
            # Let's modify the loop to track improvements.
            
            # Restart the iteration with tracking
            x_temp = x_old.copy()
            f_temp = f_old
            improvements = np.zeros(d)
            
            for i in range(d):
                if calls_ref[0] >= B - 20:
                    break
                x_temp, f_temp, _ = line_search(func, x_temp, directions[i], f_temp, calls_ref)
                improvements[i] = f_old - f_temp # This is not quite right, it's cumulative.
                # Actually, improvement_i = f_before_i - f_after_i
                # Let's just use the index of the direction that changed x the most?
            
            # Simpler Powell update:
            # The direction that yields the largest reduction in function value is replaced by (x_new - x_old).
            # We need to know which one it was.
            
            # Let's just do a simple coordinate descent with adaptive step?
            # Or just use the standard Powell update: replace direction[min_idx] where min_idx is the one with largest drop.
            
            # Since we didn't track drops per direction in the first pass, let's do it properly.
            
            # Re-do the iteration properly
            x_cur = x_old
            f_cur = f_old
            max_drop = -np.inf
            max_drop_idx = 0
            
            for i in range(d):
                if calls_ref[0] >= B - 20:
                    break
                f_prev = f_cur
                x_cur, f_cur, _ = line_search(func, x_cur, directions[i], f_cur, calls_ref)
                drop = f_prev - f_cur
                if drop > max_drop:
                    max_drop = drop
                    max_drop_idx = i
            
            x = x_cur
            fx = f_cur
            
            # Update direction
            new_dir = x - x_old
            if np.linalg.norm(new_dir) > 1e-6:
                directions[max_drop_idx] = new_dir / np.linalg.norm(new_dir) # Normalize? Powell doesn't strictly require normalization, but it helps.
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            # Convergence check
            if max_drop < 1e-6:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
