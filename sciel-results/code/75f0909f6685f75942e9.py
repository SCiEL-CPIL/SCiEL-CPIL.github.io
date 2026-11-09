import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0
        calls_made = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Initial Sampling using Sobol-like quasi-random points
        n_init = 30
        
        # Generate quasi-random points using a simple Halton sequence
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        points = []
        # Add center
        points.append(np.zeros(dim))
        # Add Halton sequence points
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        for i in range(1, n_init):
            p = np.array([halton(i + 1, primes[j % len(primes)]) * 10 - 5 for j in range(dim)])
            points.append(p)
        
        init_results = []
        for p in points:
            if calls_made >= budget or time.time() > deadline:
                break
            f = evaluate(p)
            init_results.append((p, f))
            
        if not init_results:
            return best_x, best_f

        init_results.sort(key=lambda x: x[1])
        
        # Phase 2: Multi-start (1+1)-ES with 1/5 success rule
        def run_11es(x0, f0, max_calls):
            nonlocal calls_made, best_x, best_f
            
            x = x0.copy()
            f = f0
            sigma = 0.5  # Initial step size
            
            # 1/5 success rule parameters
            window = 10
            successes = 0
            total = 0
            
            for _ in range(max_calls):
                if calls_made >= budget or time.time() > deadline:
                    break
                
                # Generate trial point
                d = rng.standard_normal(dim)
                x_trial = x + sigma * d
                x_trial = np.clip(x_trial, -5.0, 5.0)
                f_trial = evaluate(x_trial)
                
                # Update best
                if f_trial < f:
                    x = x_trial.copy()
                    f = f_trial
                    successes += 1
                total += 1
                
                # Adapt sigma every 'window' iterations
                if total >= window:
                    success_rate = successes / total
                    if success_rate > 0.5:
                        sigma *= 1.2  # Increase step size
                    elif success_rate < 0.2:
                        sigma *= 0.8  # Decrease step size
                    successes = 0
                    total = 0
                    
            return f

        # Phase 3: Nelder-Mead polish for precise convergence
        def nelder_mead(x0, f0, max_calls):
            nonlocal calls_made, best_x, best_f
            
            # Initialize simplex
            simplex = [x0.copy()]
            f_simplex = [f0]
            
            # Create other vertices by perturbing each dimension
            step = 0.5
            for i in range(dim):
                x_new = x0.copy()
                x_new[i] += step
                x_new = np.clip(x_new, -5.0, 5.0)
                f_new = evaluate(x_new)
                simplex.append(x_new)
                f_simplex.append(f_new)
                
            # Nelder-Mead parameters
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma_nm = 0.5  # Shrinkage
            
            for _ in range(max_calls):
                if calls_made >= budget or time.time() > deadline:
                    break
                
                # Sort simplex by function value
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]
                
                # Check convergence
                if abs(f_simplex[0] - f_simplex[-1]) < 1e-10:
                    break
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflect
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = evaluate(x_r)
                
                if f_r < f_simplex[0]:
                    # Try expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = evaluate(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                elif f_r < f_simplex[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                else:
                    # Try contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = evaluate(x_c)
                    if f_c < f_simplex[-1]:
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            f_simplex[i] = evaluate(simplex[i])
                        
                        # Update best
                        min_idx = np.argmin(f_simplex)
                        if f_simplex[min_idx] < f_simplex[0]:
                            pass  # Already tracked by evaluate()
                        
                        break  # Shrinkage often means we're converged
                
            return f_simplex[0]

        # Allocate remaining budget
        remaining = budget - calls_made
        
        # Strategy: Use (1+1)-ES on top candidates, then NM polish on the best
        # Allocate 70% to (1+1)-ES, 30% to NM polish
        es_budget = int(remaining * 0.7)
        nm_budget = int(remaining * 0.3)
        
        # Run (1+1)-ES on top 3 candidates
        for i in range(min(3, len(init_results))):
            if time.time() > deadline or calls_made >= budget:
                break
            x0, f0 = init_results[i]
            # Allocate more budget to better initial points
            if i == 0:
                alloc = es_budget
            elif i == 1:
                alloc = es_budget // 2
            else:
                alloc = es_budget // 4
            run_11es(x0, f0, alloc)
        
        # NM polish on the current best
        if best_x is not None and calls_made < budget and time.time() < deadline:
            nelder_mead(best_x.copy(), best_f, nm_budget)

        return best_x, best_f
