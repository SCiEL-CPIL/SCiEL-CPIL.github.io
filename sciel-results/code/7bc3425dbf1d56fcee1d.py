import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 50s safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        # State tracking
        calls = 0
        best_x = None
        best_f = np.inf
        all_points = []
        all_fvals = []

        def evaluate(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline - 10:
                raise Exception("Budget or time limit exceeded")
            x = np.asarray(x, dtype=float)
            if x.shape != (dim,):
                x = x.reshape(dim)
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls += 1
            all_points.append(x.copy())
            all_fvals.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # 1. Initialization: Halton sequence + structured points
        def halton_sequence(index, base):
            q = 1.0
            r = 0.0
            while index > 0:
                q = q / base
                r = r + q * (index % base)
                index = index // base
            return r

        points = []
        n_init = 20
        for i in range(1, n_init + 1):
            p = np.array([halton_sequence(i, base) for base in range(2, 2 + dim)])
            p = p * 10.0 - 5.0
            points.append(p)

        # Add some structured points
        points.append(np.zeros(dim))
        points.append(np.ones(dim) * 3.0)
        points.append(np.ones(dim) * -3.0)

        # Evaluate initial points
        for p in points:
            if calls >= budget or time.time() > deadline - 10:
                break
            try:
                evaluate(p)
            except Exception:
                break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls += 1

        # 2. Ridge-Axis Tracking Strategy
        # The sharp ridge (f13) is narrow along one direction and flat along others.
        # We estimate the ridge direction from the top few points and perform
        # 1-D minimization along that axis, combined with local NM for the rest.
        
        # Sort points by function value
        valid_indices = sorted(range(len(all_fvals)), key=lambda i: all_fvals[i])
        
        # Estimate ridge direction from top K points
        top_k = min(10, len(valid_indices))
        top_points = np.array([all_points[i] for i in valid_indices[:top_k]])
        top_fvals = np.array([all_fvals[i] for i in valid_indices[:top_k]])
        
        # Weighted PCA: weight by inverse function value (better points weigh more)
        # Use a simple approach: project onto the direction of steepest descent
        # between the best and second-best point, then refine with more points
        
        if top_k >= 2:
            # Direction from worst to best in top set
            d_best = top_points[0]
            d_worst = top_points[-1]
            ridge_dir = d_best - d_worst
            ridge_norm = np.linalg.norm(ridge_dir)
            if ridge_norm > 1e-6:
                ridge_dir = ridge_dir / ridge_norm
            else:
                ridge_dir = np.zeros(dim)
                ridge_dir[0] = 1.0
        else:
            ridge_dir = np.zeros(dim)
            ridge_dir[0] = 1.0

        # 3. 1-D Line Search Along Ridge Axis
        # Perform golden-section or ternary search along the ridge direction
        # from the current best point
        
        if calls < budget - 20 and time.time() < deadline - 15:
            # Define a 1-D function along the ridge axis
            def func_1d(t):
                x_trial = best_x + t * ridge_dir
                x_trial = np.clip(x_trial, -5.0, 5.0)
                return evaluate(x_trial)
            
            # Golden section search on [-5, 5] along the ridge direction
            # But we need to be careful: the ridge might not extend that far
            # Use a bounded search around the current best
            
            # First, find a good interval by expanding from best_x
            step = 1.0
            t_min = 0.0
            t_max = 0.0
            
            # Try to find an interval where f(t) > f(0) on both sides
            # Start with small steps and expand
            for _ in range(10):
                if calls >= budget or time.time() > deadline - 10:
                    break
                try:
                    f_neg = func_1d(-step)
                    f_pos = func_1d(step)
                except Exception:
                    break
                
                if f_neg > best_f and f_pos > best_f:
                    t_min = -step
                    t_max = step
                    break
                elif f_neg > best_f:
                    # Only negative side is worse, so ridge goes positive
                    t_min = -step
                    t_max = step
                    break
                elif f_pos > best_f:
                    # Only positive side is worse, so ridge goes negative
                    t_min = -step
                    t_max = step
                    break
                else:
                    # Both sides are better or equal, expand
                    step *= 2.0
                    if step > 10.0:
                        break
            
            if t_min < t_max and t_max - t_min > 1e-6:
                # Golden section search
                gr = (np.sqrt(5.0) - 1.0) / 2.0
                a, b = t_min, t_max
                c = b - gr * (b - a)
                d = a + gr * (b - a)
                
                try:
                    fc = func_1d(c)
                    fd = func_1d(d)
                except Exception:
                    fc, fd = best_f, best_f
                
                max_iter_1d = 50
                for _ in range(max_iter_1d):
                    if calls >= budget or time.time() > deadline - 10:
                        break
                    if fd < fc:
                        a = c
                        c = d
                        fc = fd
                        d = a + gr * (b - a)
                        try:
                            fd = func_1d(d)
                        except Exception:
                            break
                    else:
                        b = d
                        d = c
                        fd = fc
                        c = b - gr * (b - a)
                        try:
                            fc = func_1d(c)
                        except Exception:
                            break
                    if abs(b - a) < 1e-6:
                        break

        # 4. Local Optimization using Nelder-Mead from best point
        if calls < budget - 20 and time.time() < deadline - 15:
            try:
                minimize(evaluate, best_x, method='Nelder-Mead', 
                         options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            except Exception:
                pass

        # 5. Coordinate Descent Polish
        if calls < budget - 5 and time.time() < deadline - 15:
            x_current = best_x.copy()
            f_current = best_f
            
            step_sizes = [0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.001]
            
            for step in step_sizes:
                if calls >= budget or time.time() > deadline - 10:
                    break
                
                improved = True
                while improved and calls < budget and time.time() < deadline - 10:
                    improved = False
                    for d in range(dim):
                        if calls >= budget or time.time() > deadline - 10:
                            break
                        
                        for direction in [1, -1]:
                            x_trial = x_current.copy()
                            x_trial[d] += direction * step
                            x_trial = np.clip(x_trial, -5.0, 5.0)
                            
                            skip = False
                            for prev_p in all_points:
                                if np.allclose(prev_p, x_trial, atol=1e-10):
                                    skip = True
                                    break
                            if skip:
                                continue
                            
                            try:
                                f_trial = evaluate(x_trial)
                            except Exception:
                                break
                            
                            if f_trial < f_current:
                                f_current = f_trial
                                x_current = x_trial.copy()
                                improved = True
                                break

        # 6. Final high-precision polish with Powell if budget remains
        if calls < budget - 10 and time.time() < deadline - 15:
            try:
                minimize(evaluate, best_x, method='Powell',
                         options={'maxiter': 50, 'xtol': 1e-10, 'ftol': 1e-10})
            except Exception:
                pass

        return best_x, best_f
