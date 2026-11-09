import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        n_calls = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = float('inf')

        def call_f(x):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget:
                return None
            n_calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Finite Difference Gradient Estimation (Forward Differences)
        def estimate_gradient(x):
            g = np.zeros(dim)
            h = 1e-4
            for i in range(dim):
                x_plus = x.copy()
                x_plus[i] += h
                x_plus = np.clip(x_plus, -5.0, 5.0)
                
                f_plus = call_f(x_plus)
                if f_plus is None:
                    return None, False
                    
                # We need f(x) which is not stored in this function.
                # We should pass f(x) or re-evaluate? 
                # Re-evaluating f(x) is wasteful. 
                # Let's modify run_bfgs to pass f0.
                # But estimate_gradient is called inside run_bfgs where f is known.
                # So we need to return g, and the caller knows f.
                # But estimate_gradient doesn't know f.
                # Let's just calculate g using (f_plus - f_current)/h.
                # We need f_current.
                # Let's change signature: estimate_gradient(x, f_current)
                pass
            return g, True

        # Revised: estimate_gradient takes f_current
        def estimate_gradient(x, f_current):
            g = np.zeros(dim)
            h = 1e-4
            for i in range(dim):
                x_plus = x.copy()
                x_plus[i] += h
                x_plus = np.clip(x_plus, -5.0, 5.0)
                
                f_plus = call_f(x_plus)
                if f_plus is None:
                    return None, False
                    
                g[i] = (f_plus - f_current) / h
            return g, True

        # Line Search using BFGS Direction
        def line_search(x, d, f0, g0):
            alpha = 1.0
            c1 = 1e-4
            max_iter = 10 # Reduced to save calls
            for _ in range(max_iter):
                x_new = np.clip(x + alpha * d, -5.0, 5.0)
                f_new = call_f(x_new)
                if f_new is None:
                    return None, None, None, alpha
                
                if f_new <= f0 + c1 * alpha * np.dot(g0, d):
                    return x_new, f_new, alpha, alpha
                alpha *= 0.5
            return None, None, None, alpha

        # BFGS Optimization Routine
        def run_bfgs(x0, max_steps=50):
            nonlocal n_calls
            x = x0.copy()
            f = call_f(x)
            if f is None:
                return x, f
            
            g, ok = estimate_gradient(x, f)
            if not ok:
                return x, f
                
            H = np.eye(dim)
            steps = 0
            
            while steps < max_steps and n_calls < budget - 10:
                if time.time() > deadline - 5:
                    break
                    
                if np.linalg.norm(g) < 1e-5:
                    break
                    
                d = -H.dot(g)
                
                if np.dot(d, g) > 0:
                    H = np.eye(dim)
                    d = -g
                    
                x_new, f_new, alpha, _ = line_search(x, d, f, g)
                
                if x_new is None:
                    # Fallback: small step
                    x_new = np.clip(x - 0.1 * g, -5.0, 5.0)
                    f_new = call_f(x_new)
                    if f_new is None:
                        break
                    g_new, ok = estimate_gradient(x_new, f_new)
                    if not ok:
                        break
                else:
                    g_new, ok = estimate_gradient(x_new, f_new)
                    if not ok:
                        break

                if f_new < f:
                    s = x_new - x
                    y = g_new - g
                    ys = np.dot(y, s)
                    
                    if ys > 1e-10:
                        Hy = H.dot(y)
                        denom2 = np.dot(y, Hy)
                        if denom2 > 1e-10:
                            H = H + (np.outer(s, y) + np.outer(y, s)) / ys - (np.outer(Hy, Hy)) / denom2
                        else:
                            H = H + np.outer(s, y) / ys
                    else:
                        H = np.eye(dim)

                    x = x_new
                    f = f_new
                    g = g_new
                    steps += 1
                else:
                    break
                    
            return x, f

        # Multi-Start Strategy
        initial_points = [np.zeros(dim)]
        for _ in range(5):
            initial_points.append(rng.uniform(-5.0, 5.0, dim))

        for p in initial_points:
            if n_calls >= budget - 20:
                break
            if time.time() > deadline - 5:
                break
            x_res, f_res = run_bfgs(p, max_steps=15)
            if f_res < best_f:
                best_f = f_res
                best_x = x_res.copy()

        # Final polish
        if n_calls < budget - 5 and time.time() < deadline - 2:
            x_res, f_res = run_bfgs(best_x, max_steps=10)
            if f_res < best_f:
                best_f = f_res
                best_x = x_res.copy()

        return best_x, best_f
