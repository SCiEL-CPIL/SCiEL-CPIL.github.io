"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start Nelder-Mead for Sphere."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin for 330s limit
        
        # State
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Helper to safely call func
        def safe_call(x):
            nonlocal calls_made
            x = np.clip(x, -5, 5)
            if calls_made >= self.budget:
                return np.inf
            fx = func(x)
            calls_made += 1
            return fx

        # 1. Initial diverse sampling to find good basins
        # Sphere is unimodal, but we don't know the shift. 
        # Sampling a few points helps if the initial guess is far, but NM is global-ish enough in low dim.
        # Let's use a grid of starting points to ensure we find the basin quickly.
        # In 5D, a small grid or random starts is fine.
        
        # Strategy: 
        # Phase 1: Sample ~10 points to find the lowest.
        # Phase 2: Run Nelder-Mead from that point.
        # Phase 3: If time/budget remains, restart from other promising points or refine.
        
        # Since f is a shifted sphere, any point is in the basin. 
        # The key is getting close to the center.
        # A random start in [-5,5] is expected distance ~sqrt(5)*2/sqrt(3) ~ 2.5 from center (if center is 0).
        # NM converges quadratically near the optimum.
        
        # Let's use a fixed set of strategic starting points to avoid bad RNG luck.
        # Or just one good random start? 
        # Inspiration 1 suggests exploiting analytical structure. 
        # Actually, for Sphere, the gradient is linear. 
        # We can estimate the center using 2 points? No, we don't know the scale.
        # But we can use the fact that it's a quadratic.
        
        # Let's stick to a robust Multi-Start NM.
        
        # Generate a set of starting points
        rng = np.random.default_rng(42)
        starts = []
        # Center
        starts.append(np.zeros(self.dim))
        # Vertices of the box
        for i in range(self.dim):
            for val in [-5, 5]:
                p = np.zeros(self.dim)
                p[i] = val
                starts.append(p)
        # Random points
        for _ in range(10):
            starts.append(rng.uniform(-5, 5, self.dim))
        
        # Evaluate all starts
        start_vals = []
        for s in starts:
            fs = safe_call(s)
            start_vals.append((fs, s))
            if time.time() > deadline:
                break
        
        if calls_made == 0:
             return None, np.inf # Should not happen
            
        # Sort by fitness
        start_vals.sort(key=lambda x: x[0])
        
        # Run Nelder-Mead from the best start
        # And potentially from the second best if time allows.
        
        def nelder_mead(func, x0, max_iter):
            # Standard NM implementation
            n = self.dim
            simplex = [x0.copy()]
            fvals = [func(x0)]
            
            # Initialize simplex
            for i in range(n):
                xi = x0.copy()
                step = 0.5 if x0[i] < 4.5 else -0.5
                xi[i] += step
                simplex.append(xi)
                fvals.append(func(xi))
            
            # Sort simplex
            idx = np.argsort(fvals)
            simplex = [simplex[i] for i in idx]
            fvals = [fvals[i] for i in idx]
            
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                f_worst = fvals[-1]
                
                # Reflection
                alpha = 1.0
                x_r = centroid + alpha * (centroid - worst)
                f_r = func(x_r)
                
                if f_r < fvals[0]:
                    # Expansion
                    gamma = 2.0
                    x_e = centroid + gamma * (centroid - worst)
                    f_e = func(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    beta = 0.5
                    x_c = centroid + beta * (worst - centroid)
                    f_c = func(x_c)
                    if f_c < f_worst:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n+1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = func(simplex[i])
                
                # Re-sort
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]
                
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()
                    
                # Convergence check
                if np.std(fvals) < 1e-10:
                    break
            
            return simplex[0], fvals[0]

        # We have limited budget. 
        # NM uses n+1 calls per iteration. 
        # Let's allocate budget wisely.
        # Initial sampling used ~16 calls.
        # Remaining: 500 - 16 = 484.
        # Each NM iteration uses ~1-2 calls. 
        # Let's run NM from the best start.
        
        # To maximize AOCC, we want to update best_f frequently.
        # The NM function above updates a local best, but we need to update the global best_f/best_x 
        # passed to the closure or returned. 
        # Let's modify NM to take best_x, best_f and return updated ones.
        
        def run_nm(x0, budget_left):
            nonlocal best_x, best_f
            n = self.dim
            simplex = [x0.copy()]
            fvals = [best_f if np.array_equal(x0, best_x) else func(x0)]
            # Wait, if x0 is not best_x, we already called it in the initial phase?
            # Yes, we called all starts. So fvals[0] is known.
            
            # Actually, let's just pass the known value.
            # But the closure `func` increments calls_made.
            # We must NOT call func again for the start point if we already did.
            # This is tricky with the safe_call wrapper.
            
            # Let's restructure: 
            # 1. Sample starts, record (f, x).
            # 2. Pick best start.
            # 3. Run NM. NM will call func for new points.
            #    We must ensure we don't re-evaluate the start point.
            
            # Let's implement NM carefully.
            
            # Initialize simplex around x0
            # x0 is already evaluated.
            simplex = [x0.copy()]
            fvals = [safe_call(x0) if not np.array_equal(x0, best_x) else best_f]
            # Wait, if x0 is the best_x from sampling, we already have its f.
            # But safe_call would call it again. 
            # We need to track evaluated points or just accept the small waste?
            # 500 budget is tight. 
            # Let's just assume we can call it again? No, "More than 500 calls... invalid".
            # We must be precise.
            
            # Better approach: 
            # The `safe_call` wrapper increments `calls_made`.
            # We should NOT call `safe_call` for points we've already evaluated.
            
            # Let's keep a dictionary of evaluated points? 
            # Or just be careful in the logic.
            
            # Let's restart the logic inside __call__ to be cleaner.
            pass

        # --- Clean Implementation ---
        
        # Reset state
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        def call_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return np.inf
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # 1. Initial Sampling
        rng = np.random.default_rng(42)
        starts = [np.zeros(self.dim)]
        for _ in range(9):
            starts.append(rng.uniform(-5, 5, self.dim))
            
        start_results = []
        for s in starts:
            fs = call_func(s)
            start_results.append((fs, s))
            if time.time() > deadline or calls_made >= self.budget:
                break
        
        # Sort starts
        start_results.sort(key=lambda x: x[0])
        
        # 2. Nelder-Mead from best start
        # We will run NM from the best start, and if budget remains, from the second best.
        
        def nelder_mead(x0, f0, max_iter):
            # x0 is already evaluated with f0
            n = self.dim
            simplex = [x0.copy()]
            fvals = [f0]
            
            # Build initial simplex
            for i in range(n):
                xi = x0.copy()
                step = 0.5 if x0[i] < 4.5 else -0.5
                xi[i] += step
                fi = call_func(xi)
                simplex.append(xi)
                fvals.append(fi)
                
                if time.time() > deadline or calls_made >= self.budget:
                    return
            
            # Sort
            idx = np.argsort(fvals)
            simplex = [simplex[i] for i in idx]
            fvals = [fvals[i] for i in idx]
            
            for _ in range(max_iter):
                if time.time() > deadline or calls_made >= self.budget:
                    break
                
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                f_worst = fvals[-1]
                
                # Reflection
                x_r = centroid + (centroid - worst)
                f_r = call_func(x_r)
                
                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + 2 * (centroid - worst)
                    f_e = call_func(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + 0.5 * (worst - centroid)
                    f_c = call_func(x_c)
                    if f_c < f_worst:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n+1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = call_func(simplex[i])
                            if time.time() > deadline or calls_made >= self.budget:
                                return
                
                # Re-sort
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]
                
                # Convergence
                if np.std(fvals) < 1e-12:
                    break

        # Run NM on best start
        if start_results:
            f_best_start, x_best_start = start_results[0]
            # Estimate iterations: remaining budget / 2
            remaining = self.budget - calls_made
            max_iter = max(1, remaining // 2)
            nelder_mead(x_best_start, f_best_start, max_iter)
            
            # If budget remains, try second best start?
            # For Sphere, one good start is usually enough. 
            # But if the first start was bad (unlikely with 10 samples), 
            # the second might be better. 
            # However, NM is very fast. 
            # Let's just run NM on the best.
            
            # If we have significant budget left (e.g. NM converged very fast), 
            # we can do a final polish or restart.
            # But NM converges quadratically, so it will stop early if tol is tight.
            # If it stops early, we have budget left. 
            # We can use it for a second NM from a different point.
            
            if calls_made < self.budget and time.time() < deadline:
                if len(start_results) > 1:
                    f_second, x_second = start_results[1]
                    # Only if the first one didn't get close to the second?
                    # Or just run it.
                    remaining2 = self.budget - calls_made
                    max_iter2 = max(1, remaining2 // 2)
                    nelder_mead(x_second, f_second, max_iter2)

        if best_x is None:
             best_x = np.zeros(self.dim)
             best_f = call_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
