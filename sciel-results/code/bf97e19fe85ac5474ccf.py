import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 330s limit
        
        eval_count = 0
        best_x = None
        best_f = np.inf
        
        def evaluate(x):
            nonlocal eval_count, best_x, best_f
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx
        
        # Phase 1: (1+1)-ES with conservative sigma adaptation
        x = np.zeros(d)
        fx = evaluate(x)
        sigma = 2.0  # Start moderate for better exploration
        success_count = 0
        target_success = 0.2
        
        # Use 50% of budget for global exploration
        phase1_budget = int(budget * 0.5)
        
        for i in range(phase1_budget - 1):
            if time.time() > deadline or eval_count >= budget - 1:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = evaluate(y)
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
                success_count += 1
            else:
                sigma *= 0.9
                success_count -= 1
            
            # Adapt sigma based on success rate
            if i % 10 == 9:
                actual_success = success_count / 10
                if actual_success > target_success:
                    sigma *= 1.3
                elif actual_success < target_success:
                    sigma *= 0.8
                success_count = 0

        # Phase 2: Nelder-Mead refinement with strict eval tracking
        # Initialize simplex
        center = best_x.copy() if best_x is not None else np.zeros(d)
        simplex = [(center.copy(), best_f)]
        
        # Use adaptive step size based on sigma
        step = max(0.05, sigma * 0.3)
        
        for i in range(d):
            if time.time() > deadline or eval_count >= budget - 1:
                break
            pt = center.copy()
            pt[i] += step
            fval = evaluate(pt)
            simplex.append((pt.copy(), fval))
        
        # Sort simplex
        simplex.sort(key=lambda p: p[1])
        
        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        
        # Run NM for remaining budget
        while eval_count < budget - 1:
            if time.time() > deadline:
                break
            
            # Get best, worst, second worst
            x_worst, f_worst = simplex[-1]
            x_second, f_second = simplex[-2]
            
            # Centroid of all but worst
            x_centroid = np.mean([p[0] for p in simplex[:-1]], axis=0)
            
            # Reflection
            x_ref = x_centroid + alpha * (x_centroid - x_worst)
            f_ref = evaluate(x_ref)
            
            if f_ref < simplex[0][1]:  # Better than best
                # Expansion
                if eval_count >= budget - 1:
                    break
                x_exp = x_centroid + gamma * (x_centroid - x_worst)
                f_exp = evaluate(x_exp)
                if f_exp < f_ref:
                    simplex[-1] = (x_exp, f_exp)
                else:
                    simplex[-1] = (x_ref, f_ref)
            elif f_ref < f_second:  # Between best and second worst
                simplex[-1] = (x_ref, f_ref)
            else:  # Worse than second worst
                # Contraction
                if f_ref < f_worst:
                    # Outer contraction
                    x_con = x_centroid + 0.5 * (x_ref - x_centroid)
                else:
                    # Inner contraction
                    x_con = x_centroid - 0.5 * (x_worst - x_centroid)
                
                if eval_count >= budget - 1:
                    break
                f_con = evaluate(x_con)
                
                if f_con < min(f_ref, f_worst):
                    simplex[-1] = (x_con, f_con)
                else:
                    # Shrink
                    x_best, f_best_cur = simplex[0]
                    new_simplex = []
                    for i, (x_i, f_i) in enumerate(simplex):
                        if i == 0:
                            new_simplex.append((x_i, f_i))
                        else:
                            if eval_count >= budget - 1:
                                break
                            x_shrink = x_best + rho * (x_i - x_best)
                            f_shrink = evaluate(x_shrink)
                            new_simplex.append((x_shrink, f_shrink))
                    if len(new_simplex) == len(simplex):
                        simplex = new_simplex
                    break  # If we couldn't complete shrink, stop
        
            # Re-sort
            simplex.sort(key=lambda p: p[1])
            
            # Convergence check
            x_min = np.min([p[0] for p in simplex], axis=0)
            x_max = np.max([p[0] for p in simplex], axis=0)
            if np.max(x_max - x_min) < 1e-10:
                break
        
        return best_x, best_f
