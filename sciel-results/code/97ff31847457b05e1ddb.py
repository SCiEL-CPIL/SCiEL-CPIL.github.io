import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = None
        self.best_x = None
        self.best_f = float('inf')

    def __call__(self, func):
        self.start_time = time.time()
        deadline = self.start_time + 280.0  # Safety margin
        rng = np.random.default_rng(123)
        n = self.dim
        budget = self.budget

        def evaluate(x):
            if self.calls >= budget or time.time() > deadline:
                return None
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            self.calls += 1
            if val < self.best_f:
                self.best_f = val
                self.best_x = x_clipped.copy()
            return val

        # 1. Initial exploration: Sample a diverse set of points to find a good basin
        # We use a combination of center, random, and structured points
        initial_candidates = []
        # Center
        initial_candidates.append(np.zeros(n))
        # Random points
        for _ in range(20):
            initial_candidates.append(rng.uniform(-4, 4, n))
        # Some points near boundaries (Rosenbrock minima are often near 1 in each dim before rotation)
        for _ in range(10):
            initial_candidates.append(rng.uniform(0.5, 1.5, n))
            initial_candidates.append(rng.uniform(-1.5, -0.5, n))
        
        # Evaluate initial candidates
        for x in initial_candidates:
            if self.calls >= budget or time.time() > deadline:
                break
            evaluate(x)
        
        if self.best_x is None:
            # Fallback if we couldn't even init
            self.best_x = np.zeros(n)
            evaluate(self.best_x)
            return self.best_x, self.best_f

        # 2. Hybrid Local Search: Nelder-Mead with restarts and adaptive step sizes
        # We will run a series of local optimizations. 
        # Strategy: 
        # - Start from current best.
        # - If stuck (no improvement in last K calls) or time is running out, 
        #   perturb the best point and try again.
        # - Use Nelder-Mead as it is robust for Rosenbrock-like functions.
        
        # We define a local search function that uses Nelder-Mead
        def local_search(x_start, max_iter):
            """
            Perform local search using Nelder-Mead.
            Returns the best x and f found during this local search.
            """
            if self.calls >= budget or time.time() > deadline:
                return x_start, self.best_f
            
            # We need a wrapper that stops if we exceed local budget or global budget
            local_calls = [0]
            local_best_x = [x_start.copy()]
            local_best_f = [self.best_f]
            
            def wrapper(z):
                if self.calls >= budget or time.time() > deadline or local_calls[0] >= max_iter * n * 2:
                    return local_best_f[0]
                
                z_clipped = np.clip(z, -5, 5)
                val = func(z_clipped)
                self.calls += 1
                local_calls[0] += 1
                
                if val < local_best_f[0]:
                    local_best_f[0] = val
                    local_best_x[0] = z_clipped.copy()
                    
                # Update global best
                if val < self.best_f:
                    self.best_f = val
                    self.best_x = z_clipped.copy()
                    
                return val

            try:
                # Nelder-Mead options
                # xatol and fatol control convergence
                result = scipy_optimize_minimize(
                    wrapper, 
                    x_start, 
                    method='Nelder-Mead',
                    options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
            except Exception:
                pass
            
            return self.best_x, self.best_f

        # Since we can't import scipy.optimize inside the method easily without top-level import, 
        # and the prompt says scipy is available, let's import it at the top of the file.
        # But wait, the current program imports scipy.optimize at the top. I should do the same.
        
        # Let's implement a simple coordinate descent or Nelder-Mead manually if scipy is not desired, 
        # but scipy is allowed. Let's use scipy.optimize.minimize.
        
        # We will alternate between local refinement and global jumps.
        # Phase 1: Aggressive local refinement from the best point found in initial exploration.
        # Phase 2: If stagnation, jump to a new random point and refine.
        
        # Let's run a loop of "attempts". Each attempt is a local search.
        # If the local search improves the global best, we continue refining from there.
        # If it stagnates, we perturb the best point or jump to a new random location.
        
        attempts = 0
        max_attempts = 50
        last_best_f = self.best_f
        stagnation_counter = 0
        stagnation_limit = 30  # If no improvement in 30 calls, consider a jump
        
        while self.calls < budget and time.time() < deadline and attempts < max_attempts:
            attempts += 1
            
            # Determine starting point for this attempt
            if attempts == 1:
                x_start = self.best_x.copy()
            else:
                # If we stagnated in the previous attempt, we might want to jump
                if stagnation_counter > stagnation_limit:
                    # Jump to a new random point, but bias towards the region of the best point
                    # or just a fresh random point to escape local minima
                    if rng.random() < 0.5:
                        # Local jump
                        x_start = self.best_x + rng.normal(0, 0.5, n)
                    else:
                        # Global jump
                        x_start = rng.uniform(-4, 4, n)
                    stagnation_counter = 0
                else:
                    # Continue refining from current best
                    x_start = self.best_x.copy()
            
            # Perform local search
            # Allocate a dynamic budget for local search
            # Early on, use more budget. Later, use less to allow for more jumps if needed.
            if self.calls < budget * 0.5:
                local_max_iter = 200
            else:
                local_max_iter = 50
                
            x_best_local, f_best_local = local_search(x_start, local_max_iter)
            
            # Check for stagnation
            if f_best_local == last_best_f:
                stagnation_counter += self.calls - (self.calls - local_max_iter * n * 2) # Approximate
                # Actually, let's just track calls since last improvement
                # We need to track calls_since_improvement
            else:
                last_best_f = f_best_local
                stagnation_counter = 0

        return self.best_x, self.best_f

# We need to import scipy.optimize at the top
import scipy.optimize

# Redefine Optimizer to use the top-level import
class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = None
        self.best_x = None
        self.best_f = float('inf')

    def __call__(self, func):
        self.start_time = time.time()
        deadline = self.start_time + 280.0
        rng = np.random.default_rng(123)
        n = self.dim
        budget = self.budget

        def evaluate(x):
            if self.calls >= budget or time.time() > deadline:
                return None
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            self.calls += 1
            if val < self.best_f:
                self.best_f = val
                self.best_x = x_clipped.copy()
            return val

        # 1. Initial exploration
        initial_candidates = []
        initial_candidates.append(np.zeros(n))
        for _ in range(20):
            initial_candidates.append(rng.uniform(-4, 4, n))
        for _ in range(10):
            initial_candidates.append(rng.uniform(0.5, 1.5, n))
            initial_candidates.append(rng.uniform(-1.5, -0.5, n))
        
        for x in initial_candidates:
            if self.calls >= budget or time.time() > deadline:
                break
            evaluate(x)
        
        if self.best_x is None:
            self.best_x = np.zeros(n)
            evaluate(self.best_x)
            return self.best_x, self.best_f

        # 2. Hybrid Local Search with restarts
        def local_search(x_start, max_iter):
            if self.calls >= budget or time.time() > deadline:
                return x_start, self.best_f
            
            local_calls = [0]
            local_best_x = [x_start.copy()]
            local_best_f = [self.best_f]
            
            def wrapper(z):
                if self.calls >= budget or time.time() > deadline or local_calls[0] >= max_iter * n * 2:
                    return local_best_f[0]
                
                z_clipped = np.clip(z, -5, 5)
                val = func(z_clipped)
                self.calls += 1
                local_calls[0] += 1
                
                if val < local_best_f[0]:
                    local_best_f[0] = val
                    local_best_x[0] = z_clipped.copy()
                    
                if val < self.best_f:
                    self.best_f = val
                    self.best_x = z_clipped.copy()
                    
                return val

            try:
                result = scipy.optimize.minimize(
                    wrapper, 
                    x_start, 
                    method='Nelder-Mead',
                    options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
            except Exception:
                pass
            
            return self.best_x, self.best_f

        attempts = 0
        max_attempts = 50
        last_best_f = self.best_f
        stagnation_counter = 0
        stagnation_limit = 30
        
        while self.calls < budget and time.time() < deadline and attempts < max_attempts:
            attempts += 1
            
            if attempts == 1:
                x_start = self.best_x.copy()
            else:
                if stagnation_counter > stagnation_limit:
                    if rng.random() < 0.5:
                        x_start = self.best_x + rng.normal(0, 0.5, n)
                    else:
                        x_start = rng.uniform(-4, 4, n)
                    stagnation_counter = 0
                else:
                    x_start = self.best_x.copy()
            
            if self.calls < budget * 0.5:
                local_max_iter = 200
            else:
                local_max_iter = 50
                
            calls_before = self.calls
            x_best_local, f_best_local = local_search(x_start, local_max_iter)
            calls_after = self.calls
            
            if f_best_local < last_best_f:
                last_best_f = f_best_local
                stagnation_counter = 0
            else:
                stagnation_counter += calls_after - calls_before

        return self.best_x, self.best_f
