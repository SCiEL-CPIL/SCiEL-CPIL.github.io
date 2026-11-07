import numpy as np
import scipy.optimize

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
        
        # Directions: standard basis
        directions = [np.eye(self.dim)[i] for i in range(self.dim)]
        
        # Store last step sizes for each direction to accelerate convergence
        step_sizes = [1.0] * self.dim

        def line_search(x, d, fx, step_size):
            # Determine valid alpha range
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
                return x, fx, 0.0, step_size

            # Check budget
            if self.calls >= self.budget:
                return x, fx, 0.0, step_size

            # Try initial step
            a1 = np.clip(step_size, alpha_min, alpha_max)
            if a1 == 0:
                a1 = 1e-3 * (alpha_max - alpha_min)
            
            if self.calls >= self.budget:
                return x, fx, 0.0, step_size
                
            x1 = x + a1 * d
            f1 = self._safe_call(x1)
            
            best_alpha = 0
            best_f = fx
            if f1 < best_f:
                best_alpha = a1
                best_f = f1

            # If the step didn't improve, try smaller or opposite?
            # For Sphere, if f1 > fx, the min is likely between 0 and a1 or a1 is too big.
            # Let's assume a1 is a reasonable guess.
            
            # Try a second point to fit quadratic
            # If f1 < fx, we are descending. Try a larger step to see if we pass the min.
            # If f1 > fx, we might have overshot or the step is bad.
            
            a2 = np.clip(2.0 * a1, alpha_min, alpha_max)
            if abs(a2 - a1) < 1e-6:
                a2 = np.clip(a1 + 1e-3, alpha_min, alpha_max)
            
            if self.calls >= self.budget:
                return x + best_alpha * d, best_f, fx - best_f, step_size

            x2 = x + a2 * d
            f2 = self._safe_call(x2)
            
            if f2 < best_f:
                best_alpha = a2
                best_f = f2

            # Fit quadratic through (0, fx), (a1, f1), (a2, f2)
            # f(a) = A*a^2 + B*a + C
            # C = fx
            # A*a1^2 + B*a1 = f1 - fx
            # A*a2^2 + B*a2 = f2 - fx
            
            denom = a1 * a2 * (a2 - a1)
            if abs(denom) < 1e-12:
                # Points are too close, fallback to best found
                return x + best_alpha * d, best_f, fx - best_f, step_size
                
            # Solve for A and B
            # B = ( (f1-fx)/a1 - (f2-fx)/a2 ) / (a1 - a2)
            term1 = (f1 - fx) / a1
            term2 = (f2 - fx) / a2
            B = (term1 - term2) / (a1 - a2)
            
            # A = (f1 - fx - B*a1) / a1^2
            A = (f1 - fx - B * a1) / (a1 * a1)
            
            # Minimum at a_min_quadratic = -B / (2*A)
            if abs(A) < 1e-12:
                # Linear case
                a_opt = a2 if f2 < f1 else a1
            else:
                a_opt = -B / (2 * A)
                
            # Clip to valid range
            a_opt = np.clip(a_opt, alpha_min, alpha_max)
            
            # Check if a_opt is distinct from a1 and a2
            if abs(a_opt - a1) > 1e-4 and abs(a_opt - a2) > 1e-4:
                if self.calls < self.budget:
                    x_opt = x + a_opt * d
                    f_opt = self._safe_call(x_opt)
                    if f_opt < best_f:
                        best_alpha = a_opt
                        best_f = f_opt
                        # Update step size for next time?
                        # If a_opt is good, use it as new step size
                        step_size = abs(a_opt)
            
            x_new = x + best_alpha * d
            improvement = fx - best_f
            return x_new, best_f, improvement, step_size

        # Main loop
        while self.calls < self.budget:
            x_start = x.copy()
            fx_start = fx
            
            improvements = []
            
            # Cycle through directions
            for i in range(self.dim):
                if self.calls >= self.budget:
                    break
                
                x, fx, imp, step_sizes[i] = line_search(x, directions[i], fx, step_sizes[i])
                improvements.append(imp)
            
            if self.calls >= self.budget:
                break
                
            # Check convergence
            if abs(fx_start - fx) < 1e-12:
                break
            
            # Powell's Method Update
            new_dir = x - x_start
            norm_new_dir = np.linalg.norm(new_dir)
            if norm_new_dir < 1e-12:
                continue
            
            worst_idx = np.argmin(improvements)
            worst_imp = improvements[worst_idx]
            
            if self.calls >= self.budget:
                break
                
            # Use a default step size for the new direction, or 1.0
            x_new_dir, fx_new_dir, imp_new, _ = line_search(x_start, new_dir, fx_start, 1.0)
            
            if imp_new > worst_imp:
                directions[worst_idx] = new_dir
                x = x_new_dir
                fx = fx_new_dir
            else:
                pass

        return x, fx

    def _safe_call(self, x):
        if self.calls >= self.budget:
            pass
        self.calls += 1
        return self.func(x)
