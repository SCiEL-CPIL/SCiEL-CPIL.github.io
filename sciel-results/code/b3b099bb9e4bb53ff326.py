import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        best_x = np.zeros(n)
        best_f = float('inf')
        calls = 0

        def try_eval(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return None
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            try:
                f = float(func(x))
            except Exception:
                f = float('inf')
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def line_search(x, d, f_x, max_iter=20):
            """
            1D line search along direction d from x.
            Uses a simple bisection/golden hybrid or just golden section.
            We need to find alpha that minimizes f(x + alpha * d).
            We constrain x + alpha * d to be in [-5, 5]^n.
            """
            # Determine valid range for alpha
            # x + alpha * d in [-5, 5]
            # (x_i - 5) <= alpha * d_i <= (5 - x_i)
            alphas = []
            for i in range(n):
                if d[i] > 1e-10:
                    alphas.append((5.0 - x[i]) / d[i])
                elif d[i] < -1e-10:
                    alphas.append((-5.0 - x[i]) / d[i])
                else:
                    alphas.append(np.inf)
                    alphas.append(-np.inf) # placeholder, will filter
            
            # Actually, let's just find min and max alpha
            min_alpha = -np.inf
            max_alpha = np.inf
            for i in range(n):
                if d[i] > 1e-10:
                    max_alpha = min(max_alpha, (5.0 - x[i]) / d[i])
                elif d[i] < -1e-10:
                    min_alpha = max(min_alpha, (-5.0 - x[i]) / d[i])
                else:
                    pass # no constraint
            
            # Ensure we don't go too far out, maybe limit to [-10, 10] if unbounded? 
            # But box is [-5, 5], so alpha is bounded by box constraints.
            
            if min_alpha >= max_alpha:
                return x, f_x, 0.0

            # Golden section search
            gr = (np.sqrt(5.0) - 1.0) / 2.0
            a = min_alpha
            b = max_alpha
            c = b - gr * (b - a)
            d_ = a + gr * (b - a)
            
            # Evaluate initial points
            x_c = x + c * d
            x_d = x + d_ * d
            f_c = try_eval(x_c)
            f_d = try_eval(x_d)
            
            if f_c is None or f_d is None:
                return x, f_x, 0.0
                
            if f_c < f_d:
                b = d_
                d_ = c
                f_d = f_c
                c = b - gr * (b - a)
                x_c = x + c * d
                f_c = try_eval(x_c)
                if f_c is None:
                    break
                if f_c < f_d:
                    b = d_
                    d_ = c
                    f_d = f_c
                    c = b - gr * (b - a)
            else:
                a = c
                c = d_
                f_c = f_d
                d_ = a + gr * (b - a)
                x_d = x + d_ * d
                f_d = try_eval(x_d)
                if f_d is None:
                    break
                if f_d < f_c:
                    a = c
                    c = d_
                    f_c = f_d
                    d_ = a + gr * (b - a)
            
            # Iterative refinement
            for _ in range(max_iter):
                if calls >= budget - 5 or time.time() > deadline - 5:
                    break
                if b - a < 1e-4:
                    break
                if f_c < f_d:
                    b = d_
                    d_ = c
                    f_d = f_c
                    c = b - gr * (b - a)
                    x_c = x + c * d
                    f_c = try_eval(x_c)
                    if f_c is None:
                        break
                else:
                    a = c
                    c = d_
                    f_c = f_d
                    d_ = a + gr * (b - a)
                    x_d = x + d_ * d
                    f_d = try_eval(x_d)
                    if f_d is None:
                        break
            
            # Pick best of last few
            alpha_best = (a + b) / 2.0
            x_best = x + alpha_best * d
            f_best = try_eval(x_best)
            if f_best is None:
                # fallback to best found in search
                if f_c is not None and f_d is not None:
                    if f_c < f_d:
                        return x_c, f_c, c
                    else:
                        return x_d, f_d, d_
                return x, f_x, 0.0
            
            return x_best, f_best, alpha_best

        def powell_method(x0, f0, max_iter=50):
            """
            Powell's Method implementation.
            """
            nonlocal calls, best_x, best_f
            if max_iter < 2:
                return x0, f0

            x = x0.copy()
            f = f0
            directions = np.eye(n)
            
            for _ in range(max_iter):
                if calls >= budget - 10 or time.time() > deadline - 10:
                    break
                
                x_start = x.copy()
                f_start = f
                worst_dir_idx = -1
                worst_decrease = 0.0
                
                for k in range(n):
                    if calls >= budget - 10 or time.time() > deadline - 10:
                        break
                    
                    d = directions[k]
                    x_new, f_new, alpha = line_search(x, d, f)
                    
                    if f_new < f:
                        x = x_new
                        f = f_new
                        decrease = f_start - f
                        if decrease > worst_decrease:
                            worst_decrease = decrease
                            worst_dir_idx = k
                            
                if worst_dir_idx == -1:
                    break
                    
                # Update direction
                # d_new = 2 * (x - x_start) - directions[worst_dir_idx]
                d_new = 2.0 * (x - x_start) - directions[worst_dir_idx]
                
                # Check if d_new is linearly dependent (optional, usually skip for simplicity in low dim)
                # Replace worst direction
                directions[worst_dir_idx] = d_new
                
                # One more search along new direction
                if calls >= budget - 10 or time.time() > deadline - 10:
                    break
                x_new, f_new, alpha = line_search(x, d_new, f)
                if f_new < f:
                    x = x_new
                    f = f_new

            return x, f

        # Phase 1: Global sampling
        def halton(index, dim):
            point = []
            for j in range(dim):
                prime = [2, 3, 5, 7, 11][j]
                f = 1.0
                result = 0.0
                i = index
                while i > 0:
                    f /= prime
                    result += f * (i % prime)
                    i //= prime
                point.append(result * 10.0 - 5.0)
            return np.array(point)

        n_global = 50
        for i in range(1, n_global + 1):
            if calls >= budget - 50 or time.time() > deadline - 30:
                break
            x = halton(i, n)
            try_eval(x)
            
        for _ in range(10):
            if calls >= budget - 50 or time.time() > deadline - 30:
                break
            x = rng.uniform(-5, 5, n)
            try_eval(x)

        # Phase 2: Local refinement with Powell
        if best_f < float('inf') and calls < budget - 10:
            remaining = budget - calls
            powell_budget = min(remaining - 5, 300)
            if powell_budget >= 10:
                x_best, f_best = powell_method(best_x, best_f, max_iter=powell_budget)
                if f_best < best_f:
                    best_x = x_best
                    best_f = f_best

        # Phase 3: Second restart if budget allows
        if calls < budget - 10 and time.time() < deadline - 10:
            # Find a second candidate
            candidates = []
            for i in range(20, 80):
                if calls >= budget - 10 or time.time() > deadline - 10:
                    break
                x = halton(i, n)
                f = try_eval(x)
                if f is not None:
                    candidates.append((f, x))
            
            if candidates:
                candidates.sort(key=lambda t: t[0])
                for f_c, x_c in candidates:
                    if np.linalg.norm(x_c - best_x) > 1.0:
                        remaining = budget - calls
                        if remaining >= 10:
                            powell_budget = min(remaining - 5, 150)
                            if powell_budget >= 10:
                                x_try, f_try = powell_method(x_c, f_c, max_iter=powell_budget)
                                if f_try < best_f:
                                    best_x = x_try
                                    best_f = f_try
                        break

        return best_x, best_f
