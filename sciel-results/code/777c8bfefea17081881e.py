import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        calls_used = 0
        best_x = None
        best_f = np.inf
        
        def safe_call(x):
            nonlocal calls_used, best_x, best_f
            if calls_used >= self.budget:
                return None
            if time.time() > deadline:
                return None
            
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            val = func(x)
            calls_used += 1
            
            if val < best_f:
                best_f = val
                best_x = x.copy()
                
            return val

        def nelder_mead(x0, max_iter=100):
            """
            Perform Nelder-Mead optimization starting from x0.
            Returns the best x found in this run.
            """
            nonlocal calls_used, best_x, best_f
            
            n = self.dim
            # Initial simplex
            simplex = [x0.copy()]
            for i in range(n):
                xi = x0.copy()
                xi[i] += 0.5 # Initial step size
                simplex.append(xi)
            
            # Evaluate initial simplex
            for i in range(n + 1):
                f_val = safe_call(simplex[i])
                if f_val is None:
                    return best_x, best_f
                simplex[i] = (f_val, simplex[i])
            
            # Sort simplex by function value
            simplex.sort(key=lambda item: item[0])
            
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrinkage
            
            for _ in range(max_iter):
                if calls_used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                # Check convergence
                f_best = simplex[0][0]
                f_worst = simplex[-1][0]
                if abs(f_worst - f_best) < 1e-10:
                    break
                
                # Centroid of all points except the worst
                x_worst = simplex[-1][1]
                centroid = np.sum([p[1] for p in simplex[:-1]], axis=0) / n
                
                # 1. Reflection
                x_r = centroid + alpha * (centroid - x_worst)
                f_r = safe_call(x_r)
                if f_r is None:
                    break
                
                if f_r < simplex[0][0]:
                    # Try expansion
                    x_e = centroid + gamma * (centroid - x_worst)
                    f_e = safe_call(x_e)
                    if f_e is None:
                        break
                    if f_e < f_r:
                        simplex[-1] = (f_e, x_e)
                    else:
                        simplex[-1] = (f_r, x_r)
                elif f_r < simplex[-2][0]:
                    # Accept reflection
                    simplex[-1] = (f_r, x_r)
                else:
                    # Try contraction
                    x_c = centroid + rho * (x_worst - centroid)
                    f_c = safe_call(x_c)
                    if f_c is None:
                        break
                    if f_c < f_r:
                        simplex[-1] = (f_c, x_c)
                    else:
                        # Shrinkage
                        for i in range(1, n + 1):
                            x_s = simplex[0][1] + sigma * (simplex[i][1] - simplex[0][1])
                            f_s = safe_call(x_s)
                            if f_s is None:
                                break
                            simplex[i] = (f_s, x_s)
                        
                        if f_s is None:
                            break
                        
                        # Re-sort
                        simplex.sort(key=lambda item: item[0])
                        continue # Skip re-sort at end of loop if we just sorted? 
                        # Actually, standard NM sorts after every step.
                
                # Re-sort simplex
                simplex.sort(key=lambda item: item[0])
            
            return best_x, best_f

        # Phase 1: Initial Sampling
        # Sample a few points to get a good starting point for NM
        rng = np.random.default_rng(42)
        init_points = [np.zeros(self.dim)]
        # Add corners and mid-points
        for i in range(self.dim):
            p1 = np.zeros(self.dim); p1[i] = -4.5
            p2 = np.zeros(self.dim); p2[i] = 4.5
            init_points.extend([p1, p2])
        
        # Random points
        while len(init_points) < 20:
            init_points.append(rng.uniform(-4.5, 4.5, self.dim))
            
        # Evaluate initial points
        for p in init_points:
            if calls_used >= self.budget:
                break
            safe_call(p)
            
        if best_x is None:
            return None, np.inf
            
        # Phase 2: Nelder-Mead Refinement
        # Run NM from the current best
        # We can run multiple restarts if we have budget left
        # NM is deterministic given the start, so we restart from random perturbations if it converges too early or we have budget left
        
        # Strategy: Run NM. If it stops early (convergence), restart from a random point if budget remains.
        
        while calls_used < self.budget and time.time() < deadline:
            # Start NM from best_x
            # Add a small random jitter to avoid getting stuck if we are exactly at a vertex and NM degenerates
            start_point = best_x + 0.1 * rng.standard_normal(self.dim)
            start_point = np.clip(start_point, -5.0, 5.0)
            
            # Run NM with a limit on iterations to allow restarts
            # 500 budget / 6 points per iter ~ 80 iters max. Let's allow 100 iters.
            prev_best_f = best_f
            nelder_mead(start_point, max_iter=100)
            
            # If no improvement in this NM run, and we used some budget, 
            # maybe the start was bad or we are at the optimum.
            if best_f == prev_best_f:
                # Try a completely new random start if we have enough budget
                if self.budget - calls_used > 50:
                    new_start = rng.uniform(-4.5, 4.5, self.dim)
                    f_new = safe_call(new_start)
                    if f_new is not None and f_new < best_f:
                        best_x = new_start
                        best_f = f_new
                        # Continue to NM from here
                    else:
                        # If random point didn't help, and NM didn't help, 
                        # we might be stuck. Break to save time/budget? 
                        # Or just break out of the while loop if we've tried enough.
                        # For now, break if no improvement after a full NM cycle.
                        break
                else:
                    break
            else:
                # Improvement made, continue loop to try and refine further or restart
                # If we improved, we can try NM again from the new best
                # But to avoid infinite loop of tiny improvements, we can check if improvement was significant
                # For linear functions, NM should converge fast.
                pass

        return best_x, best_f
