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

        # Phase 2: Refine using the quadratic model more directly
        # Since the function is quadratic, the fitted model should be very accurate.
        # We'll do a few iterations of: estimate gradient at current best, take Newton step.
        # But first, let's try to refine the quadratic fit using points near the estimated minimum.
        
        x = best_x.copy()
        fx = best_f
        
        # Refine the quadratic model around the current best point
        # Use a smaller step size for better local accuracy
        d_refine = 0.1
        points_refine = [x]
        for i in range(dim):
            p1 = x.copy(); p1[i] += d_refine
            p2 = x.copy(); p2[i] -= d_refine
            points_refine.append(p1)
            points_refine.append(p2)
        
        # Only evaluate if we have budget
        if calls_made + len(points_refine) <= budget and time.time() < deadline:
            values_refine = np.array([evaluate(p) for p in points_refine])
            c_ref = values_refine[0]
            a_vals_ref = []
            q_vec_ref = np.zeros(dim)
            
            for i in range(dim):
                val_pos = values_refine[1 + 2*i]
                val_neg = values_refine[2 + 2*i]
                a_i = (val_pos + val_neg - 2*c_ref) / (2 * d_refine * d_refine)
                q_i = (val_pos - val_neg) / (2 * d_refine)
                a_vals_ref.append(a_i)
                q_vec_ref[i] = q_i
            
            a_ref = np.mean(a_vals_ref)
            
            if abs(a_ref) > 1e-12:
                x_opt_ref = x - q_vec_ref / (2 * a_ref)
                x_opt_ref = np.clip(x_opt_ref, -5, 5)
                f_ref = evaluate(x_opt_ref)
                if f_ref < fx:
                    x = x_opt_ref
                    fx = f_ref

        # Phase 3: Targeted local search using the quadratic model
        # Use remaining budget for a more efficient local optimization
        if calls_made < budget and time.time() < deadline:
            rng = np.random.default_rng(42)
            x = best_x.copy()
            fx = best_f
            
            # Use a geometrically decreasing step size for local search
            sigma = 0.01
            no_improve_count = 0
            max_no_improve = 20
            
            while calls_made < budget and time.time() < deadline and no_improve_count < max_no_improve:
                z = rng.standard_normal(dim)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                fy = evaluate(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    no_improve_count = 0
                    # Slightly increase step if improving
                    sigma *= 1.2
                else:
                    sigma *= 0.7
                    no_improve_count += 1
                
                if sigma < 1e-10:
                    break

        return best_x, best_f
