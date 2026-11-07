import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 330s limit
        
        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)
        
        best_x = np.zeros(self.dim)
        best_fx = np.inf
        call_count = 0

        # Tracked function to count calls and update best
        def tracked_func(x):
            nonlocal call_count, best_x, best_fx
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            call_count += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: Sparse Global Lattice Search
        # We want to identify the basin of attraction early.
        # A grid search is inefficient in high dimensions, but in 5D with a 
        # small number of points per dimension, we can cover the space.
        # Let's try a 3x3x3x3x3 grid = 243 points.
        # This uses 243 calls.
        # If we have 500 budget, we can afford this.
        # However, 3 points per dimension might be too coarse.
        # Let's try 2 points per dimension: 2^5 = 32 points.
        # Or a mix: some random, some grid.
        
        # Strategy: 
        # 1. Evaluate a small set of strategic points (center, corners, random).
        # 2. Pick the best few and run Nelder-Mead on them.
        # 3. Use remaining budget for further refinement.

        # Let's allocate:
        # - 50 calls for initial diverse sampling (random + structured)
        # - 300 calls for Nelder-Mead on top 3 candidates (100 each)
        # - 150 calls for final polish / more NM restarts

        # Initial Sampling
        candidates = []
        
        # Center
        candidates.append(np.zeros(self.dim))
        
        # Random points
        for _ in range(49):
            candidates.append(rng.uniform(-4.0, 4.0, self.dim))
        
        # Evaluate all candidates
        for x in candidates:
            if call_count >= self.budget - 10:
                break
            if not time_ok():
                break
            fx = tracked_func(x)
        
        # Sort candidates by function value
        # We need to keep track of which x gave which fx
        # Since tracked_func updates best, we can just use best_x, but we want multiple starts.
        # Let's re-evaluate or store results. Storing is safer to avoid extra calls.
        # Actually, we can just keep a list of (fx, x) for the ones we evaluated.
        # Let's redo the sampling with explicit storage to avoid relying on global best for sorting.
        
        # Reset for clean tracking of candidates
        sampled = []
        # We already used some calls. Let's just use the ones we have.
        # To be precise, let's restart the sampling logic to be clean.
        
        # --- Revised Phase 1: Clean Sampling ---
        # We will evaluate N points and store them.
        N_sample = 30
        sample_points = []
        sample_points.append(np.zeros(self.dim))
        for _ in range(N_sample - 1):
            sample_points.append(rng.uniform(-4.0, 4.0, self.dim))
        
        # We need to clear the "best" state if we want to be sure, but we can just 
        # evaluate and collect.
        
        # Let's clear the global best tracking for this phase to ensure we pick the best 
        # from THIS set, or just use the global best which will be the min of all calls.
        # Since we are minimizing, the global best is always the best found so far.
        # So we can just evaluate, and the best_x/best_fx will be the best of these.
        
        for x in sample_points:
            if call_count >= self.budget - 50 or not time_ok():
                break
            tracked_func(x)
        
        # Now best_x is the best of the first 30 points.
        # Let's run Nelder-Mead on this best point.
        
        # Phase 2: Nelder-Mead Refinement
        # Nelder-Mead is derivative-free and often faster than CMA-ES for low-dim smooth functions.
        # It uses a simplex.
        
        def nelder_mead(func, x0, max_iter=100, initial_simplex_size=0.5):
            """
            Implement Nelder-Mead manually to have full control over budget and time.
            """
            nonlocal call_count, best_x, best_fx
            dim = len(x0)
            
            # Initialize simplex
            simplex = [x0.copy()]
            for i in range(dim):
                x_i = x0.copy()
                x_i[i] += initial_simplex_size
                simplex.append(x_i)
            
            # Evaluate initial simplex
            f_values = [func(x) for x in simplex]
            
            # Sort simplex by function value
            order = np.argsort(f_values)
            simplex = [simplex[i] for i in order]
            f_values = [f_values[i] for i in order]
            
            alpha = 1.0   # Reflection
            gamma = 2.0   # Expansion
            rho = 0.5     # Contraction
            sigma = 0.5   # Shrink
            
            for it in range(max_iter):
                if not time_ok() or call_count >= self.budget - 5:
                    break
                
                # 1. Reflection
                x_centroid = np.mean(simplex[:-1], axis=0)
                x_ref = x_centroid + alpha * (x_centroid - simplex[-1])
                f_ref = func(x_ref)
                
                if f_ref < f_values[0]:
                    # 2. Expansion
                    x_exp = x_centroid + gamma * (x_ref - x_centroid)
                    f_exp = func(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_values[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_values[-1] = f_ref
                elif f_ref < f_values[-2]:
                    # Reflection is better than second worst
                    simplex[-1] = x_ref
                    f_values[-1] = f_ref
                else:
                    # 3. Contraction
                    x_con = x_centroid + rho * (simplex[-1] - x_centroid)
                    f_con = func(x_con)
                    if f_con < f_values[-1]:
                        simplex[-1] = x_con
                        f_values[-1] = f_con
                    else:
                        # 4. Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_values[i] = func(simplex[i])
                
                # Sort simplex
                order = np.argsort(f_values)
                simplex = [simplex[i] for i in order]
                f_values = [f_values[i] for i in order]
                
                # Check convergence
                if np.std(f_values) < 1e-10:
                    break
            
            return simplex[0], f_values[0]

        # Run Nelder-Mead from the best point found in Phase 1
        if time_ok() and call_count < self.budget - 20:
            try:
                x_nm, fx_nm = nelder_mead(tracked_func, best_x, max_iter=200, initial_simplex_size=1.0)
                # tracked_func already updates best_x/best_fx
            except Exception:
                pass

        # Phase 3: Additional Restart / Polish
        # If we have budget left, try another NM from a slightly perturbed best point
        # or just continue polishing.
        
        if time_ok() and call_count < self.budget - 10:
            # Perturb best_x slightly and run NM again
            x_pert = best_x + 0.1 * rng.standard_normal(self.dim)
            x_pert = np.clip(x_pert, -5.0, 5.0)
            try:
                x_nm2, fx_nm2 = nelder_mead(tracked_func, x_pert, max_iter=100, initial_simplex_size=0.5)
            except Exception:
                pass

        # Phase 4: Final 1+1-ES Polish
        # Use remaining calls for a fine-grained local search
        if time_ok() and call_count < self.budget:
            x = best_x.copy()
            fx = best_fx
            sigma = 0.05
            remaining = self.budget - call_count
            
            for _ in range(remaining):
                if not time_ok() or call_count >= self.budget:
                    break
                    
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5.0, 5.0)
                fy = tracked_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                else:
                    sigma *= 0.95
                    
                if sigma < 1e-8:
                    break

        return best_x, best_fx
