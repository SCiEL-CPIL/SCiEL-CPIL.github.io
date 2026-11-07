import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.start_time = time.time()
        # Deadline guard: 300s limit, keep 10s margin for safety
        self.deadline = self.start_time + 290.0

    def _check_time(self):
        return time.time() < self.deadline

    def __call__(self, func):
        # Wrap func to track calls and handle safety
        def safe_func(x):
            if self.calls >= self.budget:
                return self.best_f
            if not self._check_time():
                return self.best_f
            
            x = np.asarray(x, dtype=np.float64)
            # Clip to box to ensure valid input
            x_clipped = np.clip(x, -5, 5)
            
            try:
                f = float(func(x_clipped))
            except Exception:
                return 1e100
                
            self.calls += 1
            if f < self.best_f:
                self.best_f = f
                self.best_x = x_clipped.copy()
                
            return f

        # Initialize with center point to ensure best_x is not None
        # This consumes 1 call
        x0 = np.zeros(self.dim)
        safe_func(x0)
        
        if self.calls >= self.budget or not self._check_time():
            return self.best_x, self.best_f

        # Strategy: Deterministic Multi-Resolution Coordinate Descent
        # The Lunacek function is bi-modal. Standard random search might miss one basin.
        # Coordinate descent is deterministic and can be guided by a multi-resolution grid.
        
        # We define a set of "start points" that cover the domain systematically.
        # Instead of random starts, we use a stratified grid of starting points.
        # For dim=5, a full grid is expensive, so we use a coarse set of anchors.
        
        # Anchors: 
        # 1. Center
        # 2. Vertices of a hypercube scaled to fit inside [-4, 4] to stay away from boundaries initially
        # 3. Mid-points of edges? 
        # Let's use a simpler approach: 
        # We will run coordinate descent from a set of diverse starting points.
        # The "diversity" comes from a deterministic quasi-random sequence (Sobol-like or just grid).
        
        # Given budget 500, we can afford roughly 5-10 restarts with ~50-100 steps each.
        # Each step in coordinate descent costs 1 eval (if we do forward difference) or 2 (if central).
        # Let's use a "Line Search" per dimension which is more efficient.
        
        # Algorithm for one restart from x_start:
        # For step in range(max_steps):
        #   For d in range(dim):
        #     Find best point along dimension d by scanning a small interval around x[d].
        #     Update x[d].
        
        # To handle bi-modality, we need starts far apart.
        # Let's generate starts based on a low-discrepancy sequence (Halton) or a grid.
        
        # Simple Grid of Starts:
        # In 5D, a 3x3x3x3x3 grid is 243 points. Too many to evaluate all as starts.
        # We will pick a subset of starts.
        
        # Let's try 5 distinct regions:
        # 1. Center (0,0,0,0,0)
        # 2. (2,2,2,2,2)
        # 3. (-2,-2,-2,-2,-2)
        # 4. (2,-2,2,-2,2)
        # 5. (-2,2,-2,2,-2)
        
        starts = [
            np.zeros(self.dim),
            np.array([2.0]*self.dim),
            np.array([-2.0]*self.dim),
            np.array([2.0, -2.0, 2.0, -2.0, 2.0])[:self.dim],
            np.array([-2.0, 2.0, -2.0, 2.0, -2.0])[:self.dim]
        ]
        
        # Ensure we don't exceed budget with too many restarts
        # Let's allocate budget dynamically.
        
        for x_start in starts:
            if self.calls >= self.budget or not self._check_time():
                break
            
            x_curr = x_start.copy()
            f_curr = safe_func(x_curr)
            
            # If this start is already better than global best, update
            if f_curr < self.best_f:
                self.best_f = f_curr
                self.best_x = x_curr.copy()
            
            # Coordinate Descent with Line Search
            # We will perform a fixed number of passes over dimensions.
            # Each pass tries to minimize along each dimension.
            
            # To make it robust, we use a "golden section" or "ternary search" like approach
            # but since we are black-box, we just sample points.
            # A simpler, robust method for 5D:
            # For each dimension, try x + delta, x - delta for a few delta sizes.
            
            # Let's use a multi-step refinement.
            # Initial step size large, then shrink.
            
            steps_per_pass = 50
            step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]
            
            # We have limited budget. Let's estimate.
            # 5 restarts * 50 steps * 5 dims = 1250 evals? Too many.
            # We must be efficient.
            
            # Revised Plan:
            # Use a "Pattern Search" style.
            # For each restart, do a limited number of coordinate moves.
            
            # Let's do 10 full passes over dimensions for each restart.
            # 5 restarts * 10 passes * 5 dims = 250 evals.
            # Plus initial 5 evals for starts. Total ~255.
            # Remaining ~245 for final polish.
            
            # Actually, let's just run a loop until budget is low.
            
            # For this restart, run coordinate descent
            while self.calls < self.budget - 50 and self._check_time(): # Keep 50 for final polish
                improved = False
                # Try to improve along each dimension
                for d in range(self.dim):
                    if self.calls >= self.budget - 50 or not self._check_time():
                        break
                    
                    # Try a few step sizes
                    # We can try a small grid of steps for this dimension
                    # To save calls, let's just try +/- 0.5, +/- 0.1
                    # This is crude. Let's use a more adaptive step.
                    
                    # Let's stick to a simple fixed step size that shrinks over the restart
                    # We need a state variable for step size.
                    
                    # Since we are in a nested loop, let's manage step size outside.
                    # This structure is getting complex.
                    
                    # Alternative: Use Nelder-Mead? 
                    # Nelder-Mead is simple, robust, and handles non-smoothness well.
                    # It uses a simplex. In 5D, simplex has 6 points.
                    # Each iteration is ~6 evals.
                    # 500 budget / 6 = ~80 iterations.
                    # 80 iterations of Nelder-Mead is often enough for 5D if initialized well.
                    # And we can restart it.
                    
                    # Let's switch to Multi-Start Nelder-Mead.
                    # It's a different family from (1+1)-ES (simplex vs gradient-free stochastic).
                    
                    break # Break out of while loop to restart logic with Nelder-Mead
                break # Break out of for loop
                
            # If we broke out early, we should probably just implement Nelder-Mead properly below.
            # The above while loop was a placeholder for CD. Let's discard and do NM.
            
            # Reset control flow: We will implement a helper for NM.
            # But we are already inside the for loop over starts.
            # Let's just call a local function for NM optimization.
            
            # We need to be careful with the budget.
            # Let's define a function to run NM from x_start.
            
            def run_nelder_mead(x_start, max_iter):
                # Simple Nelder-Mead implementation
                # Initial simplex
                x0 = x_start.copy()
                f0 = safe_func(x0)
                
                # Create simplex
                simplex = [x0.copy()]
                f_simplex = [f0]
                
                # Expand in each dimension
                for d in range(self.dim):
                    x_new = x0.copy()
                    # Step size for initial simplex
                    # If x0 is 0, use 0.1. If non-zero, use 0.1 * max(1, |x0[d]|)
                    step = 0.1 if abs(x0[d]) < 1.0 else 0.1 * abs(x0[d])
                    x_new[d] += step
                    x_new = np.clip(x_new, -5, 5)
                    f_new = safe_func(x_new)
                    simplex.append(x_new)
                    f_simplex.append(f_new)
                    
                    if self.calls >= self.budget - 50 or not self._check_time():
                        return x0, f0
                
                # Nelder-Mead iterations
                alpha = 1.0  # Reflection
                gamma = 2.0  # Expansion
                rho = 0.5    # Contraction
                sigma = 0.5  # Shrink
                
                for i in range(max_iter):
                    if self.calls >= self.budget - 50 or not self._check_time():
                        break
                    
                    # Sort simplex by f
                    idx = np.argsort(f_simplex)
                    simplex = [simplex[i] for i in idx]
                    f_simplex = [f_simplex[i] for i in idx]
                    
                    # Centroid of all but worst
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Worst point
                    x_worst = simplex[-1]
                    f_worst = f_simplex[-1]
                    
                    # Reflection
                    x_r = centroid + alpha * (centroid - x_worst)
                    x_r = np.clip(x_r, -5, 5)
                    f_r = safe_func(x_r)
                    
                    if f_simplex[0] <= f_r < f_worst:
                        # Replace worst with reflection
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                    elif f_r < f_simplex[0]:
                        # Expansion
                        x_e = centroid + gamma * (x_r - centroid)
                        x_e = np.clip(x_e, -5, 5)
                        f_e = safe_func(x_e)
                        
                        if f_e < f_r:
                            simplex[-1] = x_e
                            f_simplex[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            f_simplex[-1] = f_r
                    else:
                        # Contraction
                        # Outside or inside contraction
                        if f_r < f_worst:
                            # Inside contraction
                            x_c = centroid + rho * (x_r - centroid)
                        else:
                            # Outside contraction
                            x_c = centroid - rho * (x_worst - centroid)
                        
                        x_c = np.clip(x_c, -5, 5)
                        f_c = safe_func(x_c)
                        
                        if f_c < min(f_r, f_worst):
                            simplex[-1] = x_c
                            f_simplex[-1] = f_c
                        else:
                            # Shrink
                            for j in range(1, len(simplex)):
                                simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                                simplex[j] = np.clip(simplex[j], -5, 5)
                                f_simplex[j] = safe_func(simplex[j])
                            
                            if self.calls >= self.budget - 50 or not self._check_time():
                                break
                    
                    # Update global best
                    best_in_simplex_idx = np.argmin(f_simplex)
                    if f_simplex[best_in_simplex_idx] < self.best_f:
                        self.best_f = f_simplex[best_in_simplex_idx]
                        self.best_x = simplex[best_in_simplex_idx].copy()
                    
                    # Check for convergence
                    if np.std(f_simplex) < 1e-8:
                        break
            
            # Run NM for this start
            # Estimate max_iter based on remaining budget
            # Each iter costs ~1-3 evals.
            # Let's allow up to 20 iterations per start to be safe.
            x_opt, f_opt = run_nelder_mead(x_start, max_iter=20)
            
            if f_opt < self.best_f:
                self.best_f = f_opt
                self.best_x = x_opt.copy()

        # Final Polish: Very fine local search around global best
        # Use a small step size coordinate descent or NM with tight bounds
        if self.best_x is not None and self.calls < self.budget and self._check_time():
            x_curr = self.best_x.copy()
            f_curr = self.best_f
            
            # Use a simple random hill climbing with very small steps
            # Or just run NM again with smaller initial simplex
            # Let's do a fine grid scan around best_x
            # In 5D, a 3^5 grid is 243 points.
            # We have ~50 calls left.
            # Let's do a random search in a small box.
            
            for _ in range(self.budget - self.calls - 1):
                if not self._check_time():
                    break
                # Perturb best_x
                x_prop = self.best_x + rng.normal(0, 0.05, self.dim)
                x_prop = np.clip(x_prop, -5, 5)
                f_prop = safe_func(x_prop)
                if f_prop < self.best_f:
                    self.best_f = f_prop
                    self.best_x = x_prop.copy()

        return self.best_x, self.best_f
