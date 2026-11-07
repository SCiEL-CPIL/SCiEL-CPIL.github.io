import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310  # 20s safety margin from 330s limit
        calls = 0
        max_calls = self.budget
        dim = self.dim
        
        best_x = np.zeros(dim)
        best_f = np.inf
        
        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= max_calls or time.time() > deadline:
                return np.inf
            try:
                x_clipped = np.clip(x, -5, 5)
                val = func(x_clipped)
                calls += 1
                if val < best_f:
                    best_f = val
                    best_x = x_clipped.copy()
                return val
            except Exception:
                return np.inf

        # Initialize global best with a few random points to ensure we have a valid start
        rng = np.random.default_rng(42)
        
        # Initial random sampling to find a decent starting point
        # Weierstrass is multimodal, so random starts help.
        # We spend ~10% of budget on initial exploration if possible, 
        # but Nelder-Mead is fast, so we can do a few quick starts.
        
        # Strategy: 
        # 1. Run Nelder-Mead from a random point.
        # 2. If it stalls or finishes, restart from the best found point 
        #    with a smaller perturbation, or a new random point if we suspect 
        #    we're in a local minimum far from global.
        
        # Given budget 500, we can afford ~10-20 Nelder-Mead runs 
        # if each takes ~25-50 function evaluations to converge or stall.
        
        nm_max_iter = 50 # Max iterations per NM run (each iter ~dim+1 evals)
        # Actually, standard NM takes O(dim * iterations) evaluations.
        # Let's limit total evals per run to ~30-50.
        
        # We will use a fixed budget per restart phase.
        # Phase 1: Broad exploration with random starts.
        # Phase 2: Polishing the best point found.
        
        # Let's implement a robust Nelder-Mead manually to avoid scipy overhead/dependency issues 
        # in the tight loop, although scipy is available. 
        # Using scipy.optimize.minimize is safer for correctness.
        
        from scipy.optimize import minimize
        
        # We will perform a series of Nelder-Mead optimizations.
        # To maximize AOCC (Area Over Convergence Curve), we want to improve 
        # the best value *early*. 
        # Therefore, we should not spend too much time on the first few random 
        # starts if they are bad. We should interleave or use a strategy 
        # that quickly finds a good basin.
        
        # Heuristic: 
        # 1. Evaluate center (0,0,0,0,0). 
        # 2. Run NM from center.
        # 3. If no significant improvement, try random starts.
        
        # However, the minimum is shifted. Center might be far.
        # Let's use a "Hill Climbing with Adaptive Simplex" approach.
        
        # Better approach for this specific contract:
        # Use scipy's Nelder-Mead but wrap it in a restart loop.
        
        # Since we have 500 calls, and NM is fast, we can do multiple restarts.
        # Let's divide the budget into chunks.
        # But AOCC rewards early progress. 
        # So, we should start with a very cheap, broad search, then refine.
        
        # Actually, Nelder-Mead from a random point in 5D often converges 
        # within 20-30 evaluations to a local minimum.
        
        # Let's try 5 restarts of ~100 evals each? 
        # Or 10 restarts of 50 evals each?
        
        # To get early progress, we should probably just run NM sequentially 
        # and update the global best as we go.
        
        # Let's use a fixed number of restarts based on budget.
        # num_restarts = 10
        # budget_per_restart = max_calls // num_restarts
        
        # But if the first one finds the global min, we waste the rest.
        # We should stop early if we detect convergence?
        # NM doesn't have a perfect "converged" flag that's cheap to check 
        # without looking at step size.
        
        # Let's just run NM with a maxiter limit that corresponds to 
        # ~40 function evaluations.
        # scipy's Nelder-Mead maxiter is number of iterations, 
        # where each iteration is roughly dim+1 function calls.
        # So maxiter=5 -> ~30 calls.
        
        # Let's do 15 restarts of ~30 calls each = 450 calls.
        
        num_restarts = 15
        max_iter_nm = 5 # ~30 function calls per run
        
        for i in range(num_restarts):
            if time.time() > deadline or calls >= max_calls:
                break
            
            # Determine starting point
            if i == 0:
                x0 = np.zeros(dim) # Try center first
            else:
                # Random restart, but biased towards the current best?
                # No, pure random is better for exploration in multimodal.
                # But if we are stuck in a local min, restarting from best 
                # with noise might help polish.
                # Let's mix: 50% random, 50% near best.
                if rng.random() < 0.5:
                    x0 = rng.uniform(-5, 5, dim)
                else:
                    # Perturb best
                    noise_scale = 0.5
                    x0 = best_x + noise_scale * rng.standard_normal(dim)
                    x0 = np.clip(x0, -5, 5)
            
            # Run Nelder-Mead
            # We need to ensure we don't exceed budget.
            # scipy's minimize will stop if maxiter is reached.
            # We can't easily pass a "budget remaining" check into the function 
            # to make it stop early if we run out of time, but maxiter is small.
            
            try:
                res = minimize(
                    func, 
                    x0, 
                    method='Nelder-Mead', 
                    options={'maxiter': max_iter_nm, 'xatol': 1e-4, 'fatol': 1e-4}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                # If NM fails, just continue to next restart
                pass
        
        # Final polish: If we have remaining budget, run one more NM from best_x
        if time.time() < deadline and calls < max_calls:
            try:
                res = minimize(
                    func, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={'maxiter': 20, 'xatol': 1e-6, 'fatol': 1e-6}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
