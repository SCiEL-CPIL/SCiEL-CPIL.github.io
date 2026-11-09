import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        d = self.dim
        total_budget = self.budget
        
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        n_calls = 0

        def evaluate(x):
            nonlocal n_calls, best_f, best_x
            if n_calls >= total_budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            try:
                f = func(x)
            except:
                return np.inf
            n_calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # 1. Initial Random Search
        init_budget = min(20, total_budget // 10)
        for _ in range(init_budget):
            x = rng.uniform(-5, 5, d)
            evaluate(x)
        
        if best_x is None:
            best_x = np.zeros(d)
            best_f = evaluate(best_x)

        # 2. Powell's Method
        x = best_x.copy()
        f_x = best_f
        D = np.eye(d)
        
        # Line search helper
        def line_search(x0, v, f0):
            nonlocal n_calls
            if np.linalg.norm(v) < 1e-10:
                return x0, f0, 0.0
            
            # Bracketing
            t_low = 0.0
            t_high = 1.0
            f_high = evaluate(x0 + t_high * v)
            
            # If f_high is not better, try smaller step
            if f_high >= f0:
                t_high = 0.1
                f_high = evaluate(x0 + t_high * v)
                if f_high >= f0:
                    return x0, f0, 0.0
            
            # Expand t_high if f_high is still decreasing
            expansions = 0
            while f_high < f0 and expansions < 10 and n_calls < total_budget and time.time() < deadline:
                t_low = t_high
                f_low = f_high
                t_high = t_high * 2.0
                
                # Check bounds
                x_check = x0 + t_high * v
                if np.any(x_check < -5) or np.any(x_check > 5):
                    # Find max t
                    max_t = np.inf
                    for i in range(d):
                        if v[i] > 0:
                            max_t = min(max_t, (5 - x0[i]) / v[i])
                        elif v[i] < 0:
                            max_t = min(max_t, (-5 - x0[i]) / v[i])
                    t_high = max_t * 0.99
                    if t_high <= t_low:
                        break
                f_high = evaluate(x0 + t_high * v)
                if f_high > f_low:
                    break
                expansions += 1
            
            # If we didn't bracket properly (f_high >= f_low), use the best point found
            # Golden Section Search
            gr = (np.sqrt(5) - 1) / 2
            t1 = t_low + (1 - gr) * (t_high - t_low)
            t2 = t_low + gr * (t_high - t_low)
            
            f1 = evaluate(x0 + t1 * v)
            f2 = evaluate(x0 + t2 * v)
            
            iterations = 0
            max_iter = 30
            
            while (t_high - t_low) > 1e-7 * (abs(t_high) + 1e-7) and iterations < max_iter and n_calls < total_budget and time.time() < deadline:
                if f1 < f2:
                    t_high = t2
                    t2 = t1
                    f2 = f1
                    t1 = t_low + (1 - gr) * (t_high - t_low)
                    f1 = evaluate(x0 + t1 * v)
                else:
                    t_low = t1
                    t1 = t2
                    f1 = f2
                    t2 = t_low + gr * (t_high - t_low)
                    f2 = evaluate(x0 + t2 * v)
                iterations += 1
                
            # Determine best point
            if f1 < f2:
                t_best = t1
                f_best = f1
            else:
                t_best = t2
                f_best = f2
                
            # Compare with initial f0
            if f_best > f0:
                return x0, f0, 0.0
                
            x_best = x0 + t_best * v
            return x_best, f_best, t_best

        # Main Powell Loop
        while n_calls < total_budget - 10 and time.time() < deadline:
            x_old = x.copy()
            f_old = f_x
            
            steps = np.zeros(d)
            min_f = np.inf
            min_i = 0
            worst_i = 0
            worst_step = np.inf

            for i in range(d):
                if n_calls >= total_budget - 5 or time.time() > deadline:
                    break
                
                v = D[i]
                x_new, f_new, t_step = line_search(x, v, f_x)
                
                steps[i] = t_step
                
                if f_new < min_f:
                    min_f = f_new
                    min_i = i
                    
                if abs(t_step) < worst_step:
                    worst_step = abs(t_step)
                    worst_i = i
                    
                x = x_new
                f_x = f_new
            
            if x is None or f_x is None:
                break

            # Convergence check
            if abs(f_old - f_x) < 1e-10:
                break

            # Direction Update
            if min_i < d - 1:
                new_dir = x - x_old
                if np.linalg.norm(new_dir) > 1e-10:
                    D[worst_i] = new_dir

        return best_x, best_f
