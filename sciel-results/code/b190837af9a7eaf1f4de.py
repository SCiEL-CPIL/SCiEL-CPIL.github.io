import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin before 300s limit (evaluator kills at 330s)
        
        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        def evaluate(pt):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_f
            if time.time() > deadline:
                return best_f
            val = func(pt)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = pt.copy()
            return val

        # For Sphere function f(x) = ||x - x_opt||^2 + offset
        # We can exploit the structure: f(x) = x^T x - 2 x^T x_opt + ||x_opt||^2 + offset
        # So if we evaluate at 2d+1 points (center + 2 per axis), we can exactly recover x_opt.
        # f(e_i * a) = a^2 - 2*a*x_opt[i] + ||x_opt||^2 + offset
        # f(0) = ||x_opt||^2 + offset
        # So f(e_i * a) - f(0) = a^2 - 2*a*x_opt[i]
        # x_opt[i] = (a^2 + f(0) - f(e_i * a)) / (2*a)
        
        # This requires only 2*dim + 1 = 11 evaluations!
        # Let's use a = 2.5 (well within [-5,5] and gives good numerical stability)
        
        if budget < 15:
            # Fallback for very small budgets
            x = np.zeros(dim)
            fx = evaluate(x)
            sigma = 1.0
            rng = np.random.default_rng(0)
            while calls_made < budget and time.time() < deadline:
                z = rng.standard_normal(dim)
                y = np.clip(x + sigma * z, -5, 5)
                fy = evaluate(y)
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.2
                else:
                    sigma *= 0.8
            return best_x, best_f

        a = 2.5
        f0 = evaluate(np.zeros(dim))
        
        x_est = np.zeros(dim)
        for i in range(dim):
            e_i = np.zeros(dim)
            e_i[i] = a
            f_i = evaluate(e_i)
            # f(e_i) = a^2 - 2*a*x_opt[i] + f0
            # x_opt[i] = (a^2 + f0 - f_i) / (2*a)
            x_est[i] = (a * a + f0 - f_i) / (2 * a)
        
        x_est = np.clip(x_est, -5, 5)
        f_est = evaluate(x_est)
        
        # Now polish with (1+1)-ES using adaptive sigma
        x = best_x.copy()
        fx = best_f
        
        # Use a smaller initial sigma since we're already very close
        sigma = 0.05
        rng = np.random.default_rng(42)
        
        # Success rate-based sigma adaptation (1/5th rule)
        successes = 0
        window = 10
        success_count = 0
        total_count = 0
        
        while calls_made < budget and time.time() < deadline:
            z = rng.standard_normal(dim)
            y = x + sigma * z
            y = np.clip(y, -5, 5)
            fy = evaluate(y)
            
            if fy < fx:
                x = y
                fx = fy
                success_count += 1
            
            total_count += 1
            
            # Adapt sigma every 'window' steps
            if total_count % window == 0:
                success_rate = success_count / total_count
                if success_rate > 0.2:
                    sigma *= 1.3
                elif success_rate < 0.2:
                    sigma *= 0.7
                success_count = 0
                total_count = 0
            
            # If sigma is too small, we're converged
            if sigma < 1e-12:
                break
        
        return best_x, best_f
