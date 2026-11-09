import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Time guard
        start_time = time.time()
        deadline = start_time + 280  # 20s margin before 300s limit
        
        calls = 0
        best_x = None
        best_f = float('inf')
        
        def safe_func(x):
            nonlocal calls
            if calls >= budget:
                return float('inf')
            calls += 1
            return func(x)

        # Nelder-Mead Implementation
        # Initial simplex
        # Use a moderate initial step size
        # Domain is [-5, 5]. 
        # Start at 0? Or random?
        # Since f_opt is in [-100, 100] and minimizer in [-4, 4], starting at 0 is a good guess.
        # But hidden shift might move it. 
        # Let's start with a few random points to pick a good center? 
        # Or just start NM from a reasonable point.
        # Given the convex nature, any start works, but closer is faster.
        # Let's try 3 random starts and pick the best one to initialize NM?
        # That costs 3 calls.
        
        rng = np.random.default_rng(42)
        
        # Try a few initial points to find a good region
        candidates = []
        for _ in range(3):
            x_init = rng.uniform(-5, 5, dim)
            f_init = safe_func(x_init)
            candidates.append((x_init, f_init))
            if time.time() > deadline:
                break
        
        if not candidates:
            return np.zeros(dim), float('inf')
            
        # Sort candidates
        candidates.sort(key=lambda x: x[1])
        x0, f0 = candidates[0]
        
        if best_f == float('inf'):
            best_x, best_f = x0, f0
            
        # If we have enough budget, initialize simplex around the best candidate
        # Simplex size: 1.0 is a good default for [-5,5]
        alpha = 1.0 
        simplex = [x0.copy()]
        for i in range(dim):
            x_i = x0.copy()
            x_i[i] += alpha
            # Clip to box? NM doesn't strictly respect bounds, but func clips.
            # It's better to keep simplex within bounds to avoid wasted calls on clipped points?
            # The func clips. So evaluating outside is "safe" but might be less informative.
            # Let's keep it within bounds if possible.
            if x_i[i] > 5: x_i[i] = 5
            if x_i[i] < -5: x_i[i] = -5
            f_i = safe_func(x_i)
            simplex.append(x_i)
            if f_i < best_f:
                best_x, best_f = x_i, f_i
                
        # Track simplex function values
        simplex_f = [f0] + [safe_func(s) for s in simplex[1:]] # Wait, we already evaluated simplex[1:] above?
        # No, I evaluated them in the loop. I should store them.
        
        # Let's re-structure to avoid re-evaluation
        simplex = []
        simplex_f = []
        
        # Re-initialize properly
        # We already have x0, f0
        simplex.append(x0)
        simplex_f.append(f0)
        
        for i in range(dim):
            x_i = x0.copy()
            x_i[i] += alpha
            x_i = np.clip(x_i, -5, 5)
            f_i = safe_func(x_i)
            simplex.append(x_i)
            simplex_f.append(f_i)
            if f_i < best_f:
                best_x, best_f = x_i, f_i
                
        # Sort simplex by function value
        indices = np.argsort(simplex_f)
        simplex = [simplex[i] for i in indices]
        simplex_f = [simplex_f[i] for i in indices]
        
        # NM Parameters
        # Standard:
        # Reflection: 1.0
        # Expansion: 2.0
        # Outside Contraction: 0.5
        # Inside Contraction: 0.5
        # Shrink: 0.5
        
        rho = 1.0
        chi = 2.0
        gamma = 0.5
        sigma = 0.5
        
        max_iter = 200 # Safety cap on iterations, time guard is primary
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if calls >= budget:
                break
            
            # Sort is already done from previous iteration?
            # We need to ensure simplex is sorted at start of loop
            indices = np.argsort(simplex_f)
            simplex = [simplex[i] for i in indices]
            simplex_f = [simplex_f[i] for i in indices]
            
            # Check convergence
            if simplex_f[-1] - simplex_f[0] < 1e-10:
                break
                
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + rho * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = safe_func(xr)
            
            if fr < simplex_f[0]:
                # Reflection is better than best
                # Try Expansion
                xe = centroid + chi * (xr - centroid)
                xe = np.clip(xe, -5, 5)
                fe = safe_func(xe)
                if fe < fr:
                    simplex[-1] = xe
                    simplex_f[-1] = fe
                    if fe < best_f:
                        best_x, best_f = xe, fe
                else:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                    if fr < best_f:
                        best_x, best_f = xr, fr
            elif fr < simplex_f[-2]:
                # Reflection is better than worst but not best
                simplex[-1] = xr
                simplex_f[-1] = fr
                if fr < best_f:
                    best_x, best_f = xr, fr
            else:
                # Reflection is worse than worst
                # Try Contraction
                # Inside contraction
                xc = centroid + gamma * (simplex[-1] - centroid)
                xc = np.clip(xc, -5, 5)
                fc = safe_func(xc)
                
                if fc < simplex_f[-1]:
                    simplex[-1] = xc
                    simplex_f[-1] = fc
                    if fc < best_f:
                        best_x, best_f = xc, fc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        simplex_f[i] = safe_func(simplex[i])
                        if simplex_f[i] < best_f:
                            best_x, best_f = simplex[i], simplex_f[i]
                    # Re-sort after shrink
                    indices = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in indices]
                    simplex_f = [simplex_f[i] for i in indices]
                    
        return best_x, best_f
