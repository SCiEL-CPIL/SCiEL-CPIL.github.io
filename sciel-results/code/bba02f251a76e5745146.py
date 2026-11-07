import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_used = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls_used
            if time.time() > deadline or calls_used >= budget:
                return np.inf
            calls_used += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def nelder_mead(x0, max_calls):
            nonlocal calls_used
            n = dim
            # Initial simplex
            alpha = 0.5
            simplex = [x0.copy()]
            for i in range(n):
                x_i = x0.copy()
                x_i[i] += alpha
                simplex.append(x_i)
            
            # Evaluate initial simplex
            f_vals = []
            for x in simplex:
                f_vals.append(evaluate(x))
                if calls_used >= max_calls:
                    break
            
            if calls_used >= max_calls:
                return

            # Nelder-Mead parameters
            r = 1.0   # reflection
            e = 2.0   # expansion
            c = 0.5   # contraction
            s = 0.5   # shrink
            
            while calls_used < max_calls and time.time() < deadline:
                # Sort simplex by function value
                indices = np.argsort(f_vals)
                simplex = [simplex[i] for i in indices]
                f_vals = [f_vals[i] for i in indices]
                
                x_worst = simplex[-1]
                x_best = simplex[0]
                f_worst = f_vals[-1]
                f_best = f_vals[0]
                
                # Check convergence
                if np.max(f_vals) - np.min(f_vals) < 1e-8:
                    break
                if np.linalg.norm(simplex[-1] - simplex[0]) < 1e-8:
                    break

                # Centroid of best n points
                centroid = np.sum(simplex[:-1], axis=0) / n
                
                # Reflection
                x_r = centroid + r * (centroid - x_worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = evaluate(x_r)
                if calls_used >= max_calls:
                    break
                
                if f_r < f_best:
                    # Expansion
                    x_e = centroid + e * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = evaluate(x_e)
                    if calls_used >= max_calls:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_vals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_vals[-1] = f_r
                elif f_r < f_vals[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_worst:
                        # Outside contraction
                        x_c = centroid + c * (x_r - centroid)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = evaluate(x_c)
                        if calls_used >= max_calls:
                            break
                        if f_c <= f_r:
                            simplex[-1] = x_c
                            f_vals[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = x_best + s * (simplex[i] - x_best)
                                f_vals[i] = evaluate(simplex[i])
                                if calls_used >= max_calls:
                                    return
                    else:
                        # Inside contraction
                        x_c = centroid - c * (centroid - x_worst)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = evaluate(x_c)
                        if calls_used >= max_calls:
                            break
                        if f_c < f_worst:
                            simplex[-1] = x_c
                            f_vals[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = x_best + s * (simplex[i] - x_best)
                                f_vals[i] = evaluate(simplex[i])
                                if calls_used >= max_calls:
                                    return

        def run_1plus1_es(x0, sigma0, max_calls):
            nonlocal best_x, best_f, calls_used
            x = x0.copy()
            fx = evaluate(x)
            if calls_used >= max_calls:
                return
            
            sigma = sigma0
            no_improve = 0
            max_no_improve = 30
            
            for _ in range(max_calls):
                if time.time() > deadline:
                    break
                if calls_used >= max_calls:
                    break
                
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = evaluate(y)
                
                if fy < fx:
                    x, fx = y, fy
                    no_improve = 0
                    sigma *= 1.1 # Increase step on success
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    no_improve += 1
                    sigma *= 0.9 # Decrease step on failure
                    if no_improve >= max_no_improve:
                        break

        # Strategy: Multi-Start Nelder-Mead with ES Polish
        # 1. Origin
        # 2. 4 Random points
        # Allocate budget: 5 starts * 60 calls = 300 calls.
        # Remaining 200 calls for final polish on best using (1+1)-ES.
        
        starts = [np.zeros(dim)]
        for _ in range(4):
            starts.append(rng.uniform(-4, 4, dim))
            
        calls_per_start = 60
        
        for x0 in starts:
            if calls_used >= budget:
                break
            nelder_mead(x0, calls_per_start)
            
        # Final polish
        if best_x is not None and calls_used < budget:
            remaining = budget - calls_used
            # Use (1+1)-ES for fine-tuning the best point found by Nelder-Mead
            # Start with a moderate sigma to allow some exploration, but focus on exploitation
            run_1plus1_es(best_x, 0.5, remaining)

        return best_x, best_f
