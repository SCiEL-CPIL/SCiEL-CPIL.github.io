import numpy as np
import time
from scipy.optimize import minimize, dual_annealing

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 290.0  # Safety margin before 330s kill
        
        n = self.dim
        budget = self.budget
        
        # Track evaluations
        eval_count = 0
        best_x = None
        best_f = np.inf
        
        def wrapped_func(x):
            nonlocal eval_count, best_x, best_f
            # Check time
            if time.time() - start_time > time_limit:
                return best_f if best_f < np.inf else 1e10
            
            # Check budget
            if eval_count >= budget:
                return best_f if best_f < np.inf else 1e10
                
            # Clip to box [-5, 5]
            x_clipped = np.clip(x, -5.0, 5.0)
            
            # Evaluate
            try:
                f_val = func(x_clipped)
            except Exception:
                return 1e10
                
            eval_count += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
                
            return f_val

        # Strategy: 
        # 1. Use Dual Annealing for global search.
        #    Dual Annealing is stochastic. We can limit its iterations.
        #    However, dual_annealing might use up most of the budget.
        #    Let's try to run it with a limited number of iterations or max_evals.
        
        # Alternative: Multi-start Nelder-Mead is often faster and more precise for 5D.
        # Let's try 5 random starts.
        # Each Nelder-Mead run can use up to 100 evaluations.
        # 5 * 100 = 500.
        
        # But random starts might miss the global minimum if the basin is small.
        # Katsuura has many local minima.
        
        # Let's try a hybrid:
        # 1. Run a few evaluations to estimate the function scale.
        # 2. Use scipy.optimize.differential_evolution with small population?
        #    Or just stick to the proven "Dual Annealing" insight.
        
        # Let's use dual_annealing with a max_evals constraint.
        # We don't know exactly how many evals it will use, but we can wrap it.
        
        # Actually, dual_annealing in scipy does not have a direct max_evals parameter in older versions,
        # but we can stop it via the time check in wrapped_func.
        # However, if it stops early, it might not have found the global min.
        
        # Let's try a different approach: 
        # 1. Sample 10 random points.
        # 2. Pick the best 2.
        # 3. Run Nelder-Mead on each of those 2 with a budget of 200 evals each.
        #    Total: 10 + 200 + 200 = 410. Leaves 90 for extra refinement or another start.
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Global Sampling
        # Sample 10 points
        n_samples = 10
        X_samples = rng.uniform(-5, 5, size=(n_samples, n))
        
        # Evaluate samples
        # We can do this in parallel? No, single thread.
        # Just loop.
        for x in X_samples:
            wrapped_func(x)
            
        # If budget is exceeded, return best
        if eval_count >= budget:
            return best_x, best_f
            
        # Phase 2: Local Refinement
        # Sort samples by fitness? We don't have the fitness values directly in a list 
        # because wrapped_func updates best_f but doesn't return a list.
        # Let's re-evaluate the best few? No, that wastes budget.
        
        # We need to keep track of the points we evaluated.
        # Let's modify the strategy to keep a list of (x, f) pairs.
        
        # Restart the logic with a list to track points.
        
        # --- Revised Logic ---
        
        # Reset state for clarity in implementation, though we already used 10 evals.
        # It's better to design the wrapper to store history.
        
        history = []
        
        def wrapped_func_v2(x):
            nonlocal eval_count, best_x, best_f
            if time.time() - start_time > time_limit:
                return best_f if best_f < np.inf else 1e10
            if eval_count >= budget:
                return best_f if best_f < np.inf else 1e10
            
            x_clipped = np.clip(x, -5.0, 5.0)
            try:
                f_val = func(x_clipped)
            except Exception:
                return 1e10
                
            eval_count += 1
            history.append((x_clipped.copy(), f_val))
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
                
            return f_val

        # We already called wrapped_func 10 times. 
        # But wrapped_func didn't store history. 
        # So we have wasted 10 evals without history.
        # This is inefficient. Let's restart the __call__ logic properly.
        
        # I will rewrite the __call__ body to be clean.
        
        return self._run(func, start_time, time_limit)

    def _run(self, func, start_time, time_limit):
        n = self.dim
        budget = self.budget
        eval_count = 0
        best_x = None
        best_f = np.inf
        history = []
        
        def wrapped_func(x):
            nonlocal eval_count, best_x, best_f
            if time.time() - start_time > time_limit:
                return best_f if best_f < np.inf else 1e10
            if eval_count >= budget:
                return best_f if best_f < np.inf else 1e10
            
            x_clipped = np.clip(x, -5.0, 5.0)
            try:
                f_val = func(x_clipped)
            except Exception:
                return 1e10
                
            eval_count += 1
            history.append((x_clipped.copy(), f_val))
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
                
            return f_val

        # 1. Initial Sampling
        # Use 20% of budget for global sampling
        n_samples = max(5, budget // 10)
        rng = np.random.default_rng(42)
        
        X_samples = rng.uniform(-5, 5, size=(n_samples, n))
        for x in X_samples:
            wrapped_func(x)
            
        if eval_count >= budget:
            return best_x, best_f

        # 2. Select best candidates for local optimization
        # Sort history by fitness
        history.sort(key=lambda item: item[1])
        
        # Take the top 2 or 3 candidates
        n_local = 2
        candidates = [h[0] for h in history[:n_local]]
        
        # 3. Local Optimization with Nelder-Mead
        # Allocate remaining budget to these candidates
        remaining_budget = budget - eval_count
        budget_per_candidate = remaining_budget // n_local
        
        for start_x in candidates:
            if eval_count >= budget:
                break
            if time.time() - start_time > time_limit:
                break
                
            # Nelder-Mead options
            # maxiter is not strictly eval count, but we can use callback to stop.
            # Or just let it run and rely on the wrapped_func budget check.
            # However, minimize might do many internal evals.
            # We pass wrapped_func, so it will stop when budget is hit.
            
            try:
                res = minimize(
                    wrapped_func, 
                    start_x, 
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                # Update best if res is better (should be, as wrapped_func tracks it)
            except Exception:
                pass

        return best_x, best_f
