import numpy as np
import time
import cma

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
                if not np.isfinite(f):
                    f = 1e100
            except Exception:
                f = 1e100
                
            self.calls += 1
            if f < self.best_f:
                self.best_f = f
                self.best_x = x_clipped.copy()
                
            return f

        # 1. Initial Sampling Phase
        # Lunacek's function is multimodal. We need to find good starting points for local optimizers.
        # Use 20% of budget for random sampling to identify promising regions.
        n_samples = max(10, int(self.budget * 0.2))
        
        # Generate random points in [-5, 5]^d
        # Use a fixed seed for reproducibility within the instance if possible, 
        # but instances are different. Randomness helps explore.
        # We'll just use np.random which is seeded by the environment or default.
        # To be safe and deterministic-ish per run, we could seed, but 
        # the task says "hidden seeded shift", implying the function is fixed per instance.
        # Using random exploration is standard.
        
        samples = []
        for _ in range(n_samples):
            if not self._check_time():
                break
            x = np.random.uniform(-5, 5, self.dim)
            f = safe_func(x)
            samples.append((f, x))
            
        if not samples:
            # Fallback if no samples made
            x0 = np.zeros(self.dim)
            safe_func(x0)
            return self.best_x, self.best_f

        # Sort samples by fitness
        samples.sort(key=lambda item: item[0])
        
        # Select top K distinct points to start CMA-ES
        # "Distinct" means they are far enough apart to likely represent different basins.
        # In 5D, distance > 1.0 is a reasonable heuristic for distinctness in [-5,5].
        K = 3 
        starts = []
        min_dist = 1.5
        
        for f, x in samples:
            is_distinct = True
            for _, xs in starts:
                if np.linalg.norm(x - xs) < min_dist:
                    is_distinct = False
                    break
            if is_distinct:
                starts.append((f, x))
            if len(starts) >= K:
                break
        
        # If we didn't find enough distinct points, just take the best ones
        if len(starts) < K:
            for f, x in samples:
                if (f, x) not in starts:
                    starts.append((f, x))
                if len(starts) >= K:
                    break
        
        # 2. CMA-ES Phase
        # Distribute remaining budget among the starts.
        remaining_budget = self.budget - self.calls
        
        # Allocate more budget to better starting points?
        # Or equal split? Equal split is safer for robustness.
        # Let's give the best start 50% of remaining, and split the rest among others.
        # Actually, simpler: just run them sequentially or in parallel? 
        # We can't parallelize easily in a single process without threads, 
        # and function calls are likely not thread-safe or just overhead.
        # Sequential is fine. 
        # We will run CMA-ES for each start, limiting maxfev for each.
        
        # Strategy: 
        # Start 1 (best): Get 60% of remaining budget
        # Start 2: Get 25%
        # Start 3: Get 15%
        # If time runs out, stop.
        
        budgets = []
        total_alloc = 0
        for i, (f, x) in enumerate(starts):
            if i == 0:
                b = int(remaining_budget * 0.6)
            elif i == 1:
                b = int(remaining_budget * 0.25)
            else:
                b = int(remaining_budget * 0.15)
            
            # Ensure at least 10 evaluations for CMA-ES to be meaningful
            b = max(10, b)
            # Ensure we don't exceed total remaining
            if total_alloc + b > remaining_budget:
                b = max(10, remaining_budget - total_alloc)
            
            budgets.append(b)
            total_alloc += b
            
        # Run CMA-ES for each start
        bounds = [(-5, 5)] * self.dim
        
        for i, (f_start, x_start) in enumerate(starts):
            if not self._check_time():
                break
            if self.calls >= self.budget:
                break
                
            maxfev = budgets[i]
            
            # CMA-ES settings
            # sigma0: initial step size. 
            # For Lunacek, the scale of the function varies. 
            # A sigma of 0.5 is reasonable for [-5,5] box.
            sigma0 = 0.5
            
            opts = {
                'maxfev': maxfev,
                'bounds': bounds,
                'seed': 42 + i, # Different seeds for different starts
                'verbose': -9, # Suppress output
                'tolfun': 1e-12,
                'tolfunint': 1e-12,
                'tolfx': 1e-12,
            }
            
            try:
                # Create CMA-ES object
                es = cma.CMAEvolutionStrategy(x_start, sigma0, opts)
                
                # Main loop
                while not es.stop():
                    if not self._check_time():
                        break
                    if self.calls >= self.budget:
                        break
                    
                    # Ask for next points
                    X = es.ask()
                    
                    # Evaluate
                    Y = []
                    for x in X:
                        f_val = safe_func(x)
                        Y.append(f_val)
                        
                    # Tell
                    es.tell(X, Y)
                    
                    # Update global best
                    # safe_func already updates self.best_x/f
                    
            except Exception:
                # If CMA-ES fails, continue to next start
                pass
        
        # 3. Final Polish (Optional)
        # If we have budget left, do a small random walk around the best point found
        if self.calls < self.budget and self._check_time() and self.best_x is not None:
             x = self.best_x.copy()
             f = self.best_f
             sigma = 0.1
             while self.calls < self.budget and self._check_time():
                 z = np.random.standard_normal(self.dim)
                 x_new = x + sigma * z
                 x_new = np.clip(x_new, -5, 5)
                 f_new = safe_func(x_new)
                 if f_new < f:
                     x = x_new
                     f = f_new
                     sigma *= 1.1
                 else:
                     sigma *= 0.9
                 if sigma < 1e-6:
                     break

        return self.best_x, self.best_f
