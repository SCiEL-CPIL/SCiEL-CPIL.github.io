import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.func = None

    def __call__(self, func):
        self.func = func
        self.calls = 0
        rng = np.random.default_rng(42)
        
        # Initial point
        x = rng.uniform(-5, 5, self.dim)
        fx = self._safe_call(x)
        
        # Initial directions: standard basis
        directions = [np.eye(self.dim)[i] for i in range(self.dim)]
        # Store last successful step size for each direction to improve line search efficiency
        last_step_sizes = [1.0] * self.dim
        
        def line_search(x, d, fx, d_idx=None):
            # d is the direction vector
            # We want to minimize f(x + alpha * d)
            
            # Determine valid alpha range within [-5, 5]^dim
            # x + alpha * d in [-5, 5]
            # For each i: -5 <= x[i] + alpha * d[i] <= 5
            # If d[i] > 0: (-5 - x[i]) / d[i] <= alpha <= (5 - x[i]) / d[i]
            # If d[i] < 0: (5 - x[i]) / d[i] <= alpha <= (-5 - x[i]) / d[i]
            
            alpha_min = -np.inf
            alpha_max = np.inf
            
            for i in range(self.dim):
                if abs(d[i]) < 1e-12:
                    continue
                if d[i] > 0:
                    a_min = (-5 - x[i]) / d[i]
                    a_max = (5 - x[i]) / d[i]
                else:
                    a_min = (5 - x[i]) / d[i]
                    a_max = (-5 - x[i]) / d[i]
                
                alpha_min = max(alpha_min, a_min)
                alpha_max = min(alpha_max, a_max)
            
            if alpha_min >= alpha_max:
                return x, fx, 0.0

            # Check budget
            if self.calls >= self.budget:
                return x, fx, 0.0

            # Initial alpha guesses
            # If we have a previous step size, use it as a hint
            # Otherwise, try a moderate step
            if d_idx is not None and last_step_sizes[d_idx] > 0:
                alpha_guess = last_step_sizes[d_idx]
            else:
                alpha_guess = (alpha_max - alpha_min) / 2.0
            
            # Ensure alpha_guess is within bounds
            alpha_guess = np.clip(alpha_guess, alpha_min, alpha_max)
            
            # We need 3 points for quadratic fit.
            # Let's try to center around alpha_guess.
            # Span size
            span = max(1e-4, (alpha_max - alpha_min) * 0.1)
            
            a1 = alpha_guess - span
            a2 = alpha_guess
            a3 = alpha_guess + span
            
            # Clip to valid range
            a1 = np.clip(a1, alpha_min, alpha_max)
            a3 = np.clip(a3, alpha_min, alpha_max)
            
            # If points are too close, adjust
            if abs(a2 - a1) < 1e-8:
                a1 = alpha_min
            if abs(a3 - a2) < 1e-8:
                a3 = alpha_max
            
            # Ensure distinct points
            if abs(a1 - a2) < 1e-8 or abs(a2 - a3) < 1e-8:
                # Fallback to simple bisection-like or just return current
                # For robustness, let's just do a few bisection steps
                best_alpha = alpha_guess
                best_f = self._safe_call(x + alpha_guess * d)
                for _ in range(3):
                    if self.calls >= self.budget: break
                    mid = (alpha_min + alpha_max) / 2
                    # This is getting complicated. Let's stick to quadratic if possible.
                    pass
                
                # Simple fallback: evaluate at midpoint of valid range if not done
                if self.calls < self.budget:
                    f_mid = self._safe_call(x + alpha_guess * d)
                    if f_mid < best_f:
                        best_f = f_mid
                        best_alpha = alpha_guess
                return x + best_alpha * d, best_f, fx - best_f

            # Evaluate f at a1, a2, a3
            # We already have fx which is f(x) = f(x + 0*d). 
            # If 0 is in [a1, a3], we can use it. 
            # But a1, a2, a3 are shifted.
            
            f1 = self._safe_call(x + a1 * d)
            if self.calls >= self.budget: return x + best_alpha * d if 'best_alpha' in locals() else x, min(fx, f1) if 'best_alpha' not in locals() else best_f, fx - min(fx, f1) if 'best_alpha' not in locals() else fx - best_f
            f2 = self._safe_call(x + a2 * d)
            if self.calls >= self.budget: return x + a2 * d, f2, fx - f2
            f3 = self._safe_call(x + a3 * d)
            if self.calls >= self.budget: return x + a3 * d, f3, fx - f3
            
            # Find best of the three
            best_f = min(f1, f2, f3)
            if best_f == f1:
                best_alpha = a1
            elif best_f == f2:
                best_alpha = a2
            else:
                best_alpha = a3
                
            # Update last step size
            if d_idx is not None:
                last_step_sizes[d_idx] = abs(best_alpha)

            # Quadratic interpolation
            for _ in range(5):
                if self.calls >= self.budget:
                    break
                
                if abs(a1 - a2) < 1e-10 or abs(a2 - a3) < 1e-10 or abs(a1 - a3) < 1e-10:
                    break
                    
                denom = 2.0 * (f1 * (a2 - a3) + f2 * (a3 - a1) + f3 * (a1 - a2))
                if abs(denom) < 1e-12:
                    break
                    
                num = f1 * (a2**2 - a3**2) + f2 * (a3**2 - a1**2) + f3 * (a1**2 - a2**2)
                alpha_new = num / denom
                alpha_new = np.clip(alpha_new, alpha_min, alpha_max)
                
                if abs(alpha_new - a1) < 1e-6 or abs(alpha_new - a2) < 1e-6 or abs(alpha_new - a3) < 1e-6:
                    break
                    
                if self.calls >= self.budget:
                    break
                    
                f_new = self._safe_call(x + alpha_new * d)
                
                if f_new < best_f:
                    best_f = f_new
                    best_alpha = alpha_new
                    
                # Replace the worst of the three with the new point
                # We want to keep the three points that bracket the minimum
                # If f_new < f1, replace a1
                # If f_new < f2, replace a2
                # Else replace a3
                if f_new < f1:
                    a1, f1 = alpha_new, f_new
                elif f_new < f2:
                    a2, f2 = alpha_new, f_new
                else:
                    a3, f3 = alpha_new, f_new

            x_new = x + best_alpha * d
            improvement = fx - best_f
            return x_new, best_f, improvement

        # Main Loop: Powell's Method
        while self.calls < self.budget:
            x_start = x.copy()
            fx_start = fx
            
            total_improvement = 0.0
            worst_dir_idx = -1
            worst_improvement = np.inf
            
            # Cycle through directions
            for i, d in enumerate(directions):
                if self.calls >= self.budget:
                    break
                
                x, fx, imp = line_search(x, d, fx, i)
                total_improvement += imp
                
                if imp < worst_improvement:
                    worst_improvement = imp
                    worst_dir_idx = i
            
            if self.calls >= self.budget:
                break
            
            # Check if we made significant progress
            # If total improvement is small, we might be converged
            if abs(fx_start - fx) < 1e-8:
                break
            
            # Powell's update:
            # If fx < fx_start + (fx_start - fx_prev) * 0.5 (standard check is different)
            # Standard check: If 2 * (fx_start - fx) > fx_start - fx_prev_cycle
            # We don't have fx_prev_cycle easily here without storing it.
            # Let's use a simpler heuristic: If the improvement is positive, update.
            
            # New direction is x - x_start
            new_dir = x - x_start
            norm_new_dir = np.linalg.norm(new_dir)
            
            if norm_new_dir > 1e-8:
                # Replace the worst direction with the new one
                directions[worst_dir_idx] = new_dir
                # Reset the step size for the new direction? 
                # Or keep it? The new direction is a combination.
                # Let's set its step size to the norm of the move? No, alpha is scalar.
                # The move was x_start + alpha_comb * new_dir? No, x is result of sequential searches.
                # The vector new_dir is the sum of all individual moves.
                # The next line search will find the optimal alpha along this new direction.
                # So we can reset last_step_sizes[worst_dir_idx] to a reasonable value.
                last_step_sizes[worst_dir_idx] = 1.0

        return x, fx

    def _safe_call(self, x):
        if self.calls >= self.budget:
            return np.inf
        self.calls += 1
        return self.func(x)
