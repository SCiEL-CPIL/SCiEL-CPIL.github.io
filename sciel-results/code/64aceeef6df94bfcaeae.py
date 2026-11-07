import numpy as np
import time
from scipy.optimize import cma

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin
        
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(42)

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if time.time() > deadline:
                return np.inf
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            try:
                val = func(x)
                if val < best_f:
                    best_f = val
                    best_x = x.copy()
                return val
            except:
                return np.inf

        # 1. Initial Diversified Sampling
        # We use a small set of diverse starting points to kick off CMA-ES runs.
        # This is crucial for f13 (Sharp Ridge) because the global minimum might be
        # in a specific direction or shifted.
        
        n_starts = 4
        starts = []
        
        # Start 1: Origin
        starts.append(np.zeros(dim))
        
        # Start 2-4: Random points with varying magnitudes to explore different basins
        # We want to cover the space [-5, 5]^5 reasonably well.
        for i in range(n_starts - 1):
            # Mix of small and large steps
            scale = 1.0 + i * 1.5
            p = rng.normal(0, scale, dim)
            p = np.clip(p, -4.9, 4.9)
            starts.append(p)
        
        # Evaluate all starts
        for s in starts:
            safe_call(s)
        
        if best_x is None:
            return np.zeros(dim), 0.0

        # 2. Adaptive Multi-Start CMA-ES Loop
        # We will run CMA-ES on the best current point.
        # If it stalls (no improvement for X calls) or time is up, we restart 
        # from a new random point or the current best with a perturbation.
        
        # CMA-ES parameters
        # Initial sigma should be large enough to explore but small enough to converge.
        # For dim=5, sigma=1.0 is a reasonable default, but we can adapt it.
        
        remaining_budget = budget - calls_made
        if remaining_budget < 10:
            return best_x, best_f

        # We'll manage a list of "active" search states. 
        # For simplicity and robustness within the budget, we do sequential restarts 
        # of CMA-ES, but we "warm start" with the best point found so far 
        # and perturb it slightly to escape local minima/flat areas.
        
        # Strategy:
        # - Run CMA-ES for a chunk of budget (e.g., 50-100 calls).
        # - If it improves best_f, continue or restart from new best with smaller sigma.
        # - If it stalls, restart from a random point or best_x + noise.
        
        # Let's define a helper to run one CMA-ES block
        def run_cma_block(x0, sigma, max_calls):
            nonlocal calls_made, best_f, best_x, deadline
            if time.time() > deadline or calls_made >= budget:
                return False
            
            # Create CMA-ES instance
            # We need to wrap the function to stop when max_calls is reached or time is up
            # cma.fmin doesn't have a strict "max calls" option that stops immediately on count,
            # but maxiter controls iterations. Each iteration uses dim+1 evaluations roughly.
            # To be precise, we use a custom wrapper or check inside the loop if possible.
            # However, cma.fmin is a black box. We can use cma.CMA class for more control.
            
            es = cma.CMA(x0, sigma, {'maxiter': 1000, 'bounds': [-5.0]*dim, 'seed': int(time.time()*1000) % 10000})
            
            # We need to manually loop to control the call count precisely
            # es.ask() returns points, es.tell(points, fvals) updates.
            
            # Estimate max iterations allowed
            # Each iteration consumes dim+1 calls on average (actually more due to restarts/termination logic)
            # Let's be conservative: max_iter = max_calls // (dim + 1)
            max_iter = max(1, max_calls // (dim + 1))
            
            improved = False
            prev_best = best_f
            
            for i in range(max_iter):
                if time.time() > deadline or calls_made >= budget:
                    break
                
                # Ask for points
                points = es.ask()
                
                # Evaluate points
                fvals = []
                for p in points:
                    fvals.append(safe_call(p))
                
                # Check if we ran out of budget or time during evaluation
                if calls_made >= budget or time.time() > deadline:
                    break
                    
                # Tell CMA-ES the results
                # Note: If safe_call returned inf due to timeout, we should stop.
                if any(np.isinf(f) for f in fvals):
                    break
                    
                es.tell(points, fvals)
                
                # Check for convergence
                if es.stop():
                    break
                    
            # Check if we improved
            if best_f < prev_best:
                improved = True
                
            return improved

        # Main Loop: Sequential CMA-ES runs with adaptive restarts
        # We allocate chunks of budget.
        # Chunk size: start with ~50 calls, decrease as we get closer to budget end?
        # Or keep it fixed? 
        # Let's try a fixed chunk of 60 calls for the first few, then smaller.
        
        # Initial sigma
        sigma = 1.0
        
        # We will run up to 5 restarts if we stall, or until budget is exhausted.
        max_restarts = 5
        restart_count = 0
        
        current_x = best_x.copy()
        
        while calls_made < budget and time.time() < deadline:
            # Determine chunk size
            remaining = budget - calls_made
            if remaining < 20:
                break
                
            # Adaptive chunk size: larger when we have lots of budget, smaller when tight
            chunk_size = min(remaining, 80)
            
            # If we are in a later restart, maybe reduce sigma to polish?
            # Or increase sigma if we stalled?
            # If the previous run improved, we are likely in the right basin. Keep sigma or shrink slightly.
            # If it stalled, we might be stuck. Perturb x0 and maybe increase sigma.
            
            if restart_count > 0 and not improved:
                # Stalled: Perturb start point and increase sigma to explore more
                current_x = best_x + rng.normal(0, 0.5, dim)
                current_x = np.clip(current_x, -4.9, 4.9)
                sigma = min(sigma * 1.5, 3.0)
            else:
                # Improved or first run: Start from best, maybe shrink sigma if very close
                # For f13, we don't want to shrink too fast or we miss the ridge direction.
                # Keep sigma stable or shrink slightly.
                current_x = best_x.copy()
                sigma = max(sigma * 0.9, 0.1)

            improved = run_cma_block(current_x, sigma, chunk_size)
            
            if not improved:
                restart_count += 1
                if restart_count > max_restarts:
                    # Try one final random search if CMA-ES keeps failing
                    # This is a safety net
                    for _ in range(min(remaining, 20)):
                        if time.time() > deadline or calls_made >= budget:
                            break
                        dx = rng.normal(0, 0.5, dim)
                        new_x = np.clip(best_x + dx, -5, 5)
                        safe_call(new_x)
                    break
            else:
                # If improved, reset restart counter? 
                # Or just continue. CMA-ES is good at continuing.
                # But our run_cma_block stops after chunk_size. 
                # So we restart the CMA-ES instance with the new best point.
                # This effectively continues the search from the new location.
                pass

        return best_x, best_f
