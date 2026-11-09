"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Nelder-Mead Optimizer for BBOB f12 (Bent Cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.calls = 0
        self.start_time = time.time()
        self.time_limit = 290.0
        
        d = self.dim
        N = self.budget
        
        # Global best tracking
        best_x = np.zeros(d)
        best_fx = float('inf')
        
        def eval_func(x):
            nonlocal best_x, best_fx
            if self.calls >= N:
                raise StopIteration("Budget exceeded")
            if time.time() - self.start_time > self.time_limit:
                raise StopIteration("Time limit exceeded")
            
            self.calls += 1
            fx = func(x)
            
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
                
            return fx

        def nelder_mead(start_x, max_evals=400):
            nonlocal self
            calls_at_start = self.calls
            
            def eval_func_limited(x):
                if (self.calls - calls_at_start) >= max_evals:
                    raise StopIteration("NM budget exceeded")
                if time.time() - self.start_time > self.time_limit:
                    raise StopIteration("Time limit exceeded")
                return eval_func(x)

            # Initialize simplex
            simplex = [start_x.copy()]
            f_vals = [eval_func_limited(start_x)]
            
            # Add vertices
            for i in range(d):
                v = start_x.copy()
                v[i] += 0.5 if v[i] + 0.5 <= 5.0 else -0.5
                v = np.clip(v, -5.0, 5.0)
                try:
                    fv = eval_func_limited(v)
                except StopIteration:
                    return
                simplex.append(v)
                f_vals.append(fv)
            
            # Nelder-Mead iterations
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink
            
            while self.calls - calls_at_start < max_evals and time.time() - self.start_time < self.time_limit:
                # Sort simplex
                order = np.argsort(f_vals)
                simplex = [simplex[i].copy() for i in order]
                f_vals = [f_vals[i] for i in order]
                
                # Check convergence
                if len(set(f_vals)) < 2 or (max(f_vals) - min(f_vals)) < 1e-10:
                    break
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                try:
                    fr = eval_func_limited(xr)
                except StopIteration:
                    break
                
                if fr < f_vals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5.0, 5.0)
                    try:
                        fe = eval_func_limited(xe)
                    except StopIteration:
                        break
                    
                    if fe < fr:
                        simplex[-1] = xe
                        f_vals[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_vals[-1] = fr
                elif fr < f_vals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    f_vals[-1] = fr
                else:
                    # Contraction
                    if fr < f_vals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                    
                    xc = np.clip(xc, -5.0, 5.0)
                    try:
                        fc = eval_func_limited(xc)
                    except StopIteration:
                        break
                    
                    if fc < min(fr, f_vals[-1]):
                        simplex[-1] = xc
                        f_vals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            try:
                                f_vals[i] = eval_func_limited(simplex[i])
                            except StopIteration:
                                return

        # 1. Initial sampling
        rng = np.random.default_rng(42)
        
        # Start at origin
        eval_func(np.zeros(d))
        
        # Add a few random points to escape local minima
        for _ in range(5):
            if self.calls >= N or time.time() - self.start_time > self.time_limit:
                break
            p = rng.uniform(-4.0, 4.0, size=d)
            try:
                eval_func(p)
            except StopIteration:
                break
        
        # 2. Nelder-Mead optimization from best point
        if self.calls < N and time.time() - self.start_time < self.time_limit:
            remaining = N - self.calls
            nm_budget = min(remaining, 400)
            nelder_mead(best_x, max_evals=nm_budget)

        # 3. Final coordinate polish if budget remains
        if self.calls < N and time.time() - self.start_time < self.time_limit:
            x = best_x.copy()
            fx = best_fx
            step = 0.1
            for _ in range(30):
                if self.calls >= N or time.time() - self.start_time > self.time_limit:
                    break
                improved = False
                for i in range(d):
                    if self.calls >= N or time.time() - self.start_time > self.time_limit:
                        break
                    for direction in [1, -1]:
                        x_trial = x.copy()
                        x_trial[i] += direction * step
                        x_trial = np.clip(x_trial, -5.0, 5.0)
                        try:
                            fx_trial = eval_func(x_trial)
                        except StopIteration:
                            return x, best_fx
                        except Exception:
                            continue
                        
                        if fx_trial < fx:
                            x = x_trial
                            fx = fx_trial
                            improved = True
                            break
                    if improved:
                        break
                
                if not improved:
                    step *= 0.5
                    if step < 1e-8:
                        break

        return best_x, best_fx
# EVOLVE-BLOCK-END
