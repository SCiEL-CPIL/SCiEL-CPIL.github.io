import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = float('inf')

        def evaluate(pt):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget or time.time() > deadline:
                return best_f
            pt = np.asarray(pt, dtype=np.float64)
            if pt.shape != (dim,):
                pt = pt.reshape(dim)
            val = func(pt)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = pt.copy()
            return val

        # Phase 1: Exact quadratic model fitting
        # f(x) = a*||x||^2 + q^T x + c
        # We use a moderate step size d for the initial fit to avoid numerical issues with too small steps
        # but large enough to be robust.
        d = 2.0
        points = [np.zeros(dim)]
        for i in range(dim):
            p1 = np.zeros(dim); p1[i] = d
            p2 = np.zeros(dim); p2[i] = -d
            points.append(p1)
            points.append(p2)

        values = np.array([evaluate(p) for p in points])

        c = values[0]
        a_vals = []
        q_vec = np.zeros(dim)

        for i in range(dim):
            val_pos = values[1 + 2*i]
            val_neg = values[2 + 2*i]
            a_i = (val_pos + val_neg - 2*c) / (2 * d * d)
            q_i = (val_pos - val_neg) / (2 * d)
            a_vals.append(a_i)
            q_vec[i] = q_i

        a = np.mean(a_vals)

        if abs(a) < 1e-12:
            a = 1e-12 if a >= 0 else -1e-12

        # Estimated minimum
        x_opt = -q_vec / (2 * a)
        x_opt = np.clip(x_opt, -5, 5)

        f_opt_est = evaluate(x_opt)

        # Phase 2: Polish with Finite-Difference Newton Method
        # The function is quadratic, so Newton's method should converge in 1-2 steps.
        # We use the estimated Hessian H = 2*a*I.
        # Step: x_new = x - H^{-1} grad(x) = x - (1/(2*a)) * grad(x)
        
        # Use a smaller step size for gradient estimation to be more accurate
        h_grad = 1e-5
        
        x = best_x.copy()
        fx = best_f
        
        # Perform a few Newton steps
        max_newton_steps = 20
        for _ in range(max_newton_steps):
            if calls_made >= budget or time.time() > deadline:
                break
            
            # Compute gradient using central differences
            grad = np.zeros(dim)
            for i in range(dim):
                if calls_made >= budget or time.time() > deadline:
                    break
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += h_grad
                x_minus[i] -= h_grad
                x_plus = np.clip(x_plus, -5, 5)
                x_minus = np.clip(x_minus, -5, 5)
                
                f_plus = evaluate(x_plus)
                f_minus = evaluate(x_minus)
                grad[i] = (f_plus - f_minus) / (2 * h_grad)
            
            if calls_made >= budget or time.time() > deadline:
                break

            # Check if gradient is small enough
            if np.linalg.norm(grad) < 1e-10:
                break
            
            # Newton step
            # H is approx 2*a*I, so H_inv is (1/(2*a))*I
            step_size = 1.0 / (2.0 * a)
            x_new = x - step_size * grad
            x_new = np.clip(x_new, -5, 5)
            
            f_new = evaluate(x_new)
            
            if f_new < fx:
                x = x_new
                fx = f_new
            else:
                # If it didn't improve, we might be at the minimum or the model is wrong.
                # For a convex quadratic, it should always improve unless we are at the min.
                break

        # Phase 3: Final fine-tuning random search with very small steps
        # To catch any tiny residuals
        remaining = budget - calls_made
        if remaining > 0 and time.time() < deadline:
            rng = np.random.default_rng(42)
            sigma = 1e-4
            x = best_x.copy()
            fx = best_f
            
            while calls_made < budget and time.time() < deadline:
                z = rng.standard_normal(dim)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                fy = evaluate(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    # sigma *= 1.1 # Slightly increase if improving
                else:
                    sigma *= 0.5
                    if sigma < 1e-8:
                        break

        return best_x, best_f
