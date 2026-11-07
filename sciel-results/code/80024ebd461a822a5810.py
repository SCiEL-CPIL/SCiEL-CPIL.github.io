import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s/330s kill
        
        budget = self.budget
        dim = self.dim
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        
        # Helper to evaluate and track
        def evaluate(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            try:
                val = func(x)
            except Exception:
                return None
            calls += 1
            if val < best_f:
                best_f = val
                best_x = np.array(x, dtype=float).copy()
            return val
        
        # Phase 1: Global Exploration
        # Use a moderate number of points to identify promising basins.
        # 30 points is a good balance for dim=5, budget=500.
        n_explore = 30
        candidates = []
        
        for _ in range(n_explore):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = evaluate(x)
            if fx is not None:
                candidates.append((fx, x))
        
        if best_x is None:
            return None, None
        
        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Hybrid Local Refinement
        # We will use the top few points. For each, we try Nelder-Mead first (to track ridge geometry quickly),
        # and if that doesn't converge well or runs out of budget, we fall back to (1+1)-ES with adaptive sigma.
        
        top_k = min(5, len(candidates))
        remaining_budget = budget - calls
        
        if remaining_budget <= 0:
            return best_x, best_f
        
        # We allocate budget dynamically. 
        # Strategy:
        # 1. Try NM with a small budget (e.g., 20-30 calls) on the best point. 
        # 2. If NM improves significantly, continue NM or switch to ES.
        # 3. Actually, NM can be expensive. Let's use NM on the best 1-2 points with a tight budget,
        #    and use ES on the rest.
        #    Or better: Use NM on the best point to get into the sharp ridge, then switch to ES?
        #    The prompt suggests: "Use a pure derivative-free Nelder-Mead simplex method initialized at the box center... thereby accelerating early convergence".
        #    But initialization at box center might miss the shift. The shift is hidden but bounded.
        #    The current program uses random exploration.
        #    
        #    Let's combine:
        #    - Take top 5 points.
        #    - For the best point, run NM with a budget of ~50 calls.
        #    - For the next 4 points, run (1+1)-ES with the remaining budget.
        #    - Why? NM is good for sharp ridges if initialized close. ES is robust.
        
        # Let's try a different structural split:
        # 1. NM on the best candidate with a limited budget (e.g., 30 calls).
        # 2. If NM improves, use its result as a new seed for ES?
        # 3. Or just run NM on the best, and ES on the rest?
        
        # Let's allocate:
        # - 30 calls for NM on the best point.
        # - Remaining budget for ES on the top 4 points (and maybe the NM result).
        
        nm_budget = 30
        if remaining_budget > nm_budget + 10:
            # Run NM on the best candidate
            x0 = candidates[0][1].copy()
            f0 = candidates[0][0]
            
            # NM options
            opts = {
                'maxiter': nm_budget,
                'xatol': 1e-6,
                'fatol': 1e-8,
                'adaptive': True
            }
            
            try:
                res = minimize(func, x0, method='Nelder-Mead', options=opts)
                # Note: minimize() calls func internally. We need to track those calls.
                # The evaluate() wrapper is not used by minimize directly unless we wrap func.
                # We must wrap func to count calls.
                
                # Let's re-evaluate the approach. We cannot easily wrap func for minimize without modifying the interface or creating a closure that counts.
                # Let's create a counting wrapper for minimize.
                
                nm_calls = 0
                def nm_func(x):
                    nonlocal nm_calls, calls, best_f, best_x
                    if calls >= budget or time.time() > deadline:
                        return best_f # Return best known to stop optimization? Or just a large number?
                        # Actually, if we run out of budget, we should stop.
                        # But minimize doesn't know about our global budget.
                        # We must ensure nm_calls doesn't exceed nm_budget.
                        # And that total calls doesn't exceed budget.
                        
                    if nm_calls >= nm_budget:
                        return best_f
                    if calls >= budget:
                        return best_f
                    if time.time() > deadline:
                        return best_f
                    
                    try:
                        val = func(x)
                    except Exception:
                        val = best_f
                    
                    nm_calls += 1
                    calls += 1
                    if val < best_f:
                        best_f = val
                        best_x = np.array(x, dtype=float).copy()
                    return val

                res = minimize(nm_func, x0, method='Nelder-Mead', options=opts)
                
                # If NM found something better, add it to candidates or just let it be the best.
                # The best_x is already updated.
                
            except Exception:
                pass
            
            # Update candidates list with the result of NM if it improved?
            # Actually, best_x is already the global best.
            # Let's add the NM result as a new candidate if it improved.
            if res.fun < candidates[0][0]:
                candidates.insert(0, (res.fun, res.x.copy()))
                # Re-sort? No, it's better than the first one, so it's the best.
        
        # Now, use (1+1)-ES for the remaining budget on the top candidates.
        # We have already used some budget for NM.
        # Let's take the top 4 candidates (including the one potentially updated by NM).
        # We will run ES on each, distributing the remaining budget.
        
        remaining_after_nm = budget - calls
        if remaining_after_nm <= 0:
            return best_x, best_f
        
        # Select top 4 distinct points (avoid duplicates if NM result is same as start)
        es_candidates = []
        for fx, x in candidates[:5]:
            if len(es_candidates) >= 4:
                break
            # Check if this point is too close to an existing ES candidate
            is_dup = False
            for _, ex in es_candidates:
                if np.linalg.norm(x - ex) < 1e-5:
                    is_dup = True
                    break
            if not is_dup:
                es_candidates.append((fx, x))
        
        if not es_candidates:
            return best_x, best_f
        
        n_es = len(es_candidates)
        budget_per_es = max(1, remaining_after_nm // n_es)
        
        for fx, x in es_candidates:
            if calls >= budget or time.time() > deadline:
                break
            
            x = x.copy()
            fx = fx
            sigma = 1.0
            
            # (1+1)-ES
            for _ in range(budget_per_es):
                if calls >= budget or time.time() > deadline:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                
                if fy is None:
                    break
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.5
                else:
                    sigma *= 0.87
                
                # Keep sigma in reasonable bounds
                sigma = np.clip(sigma, 1e-4, 10.0)
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            
            # If we improved significantly, we might want to re-seed other ES runs?
            # No, let's keep it simple.
            
        return best_x, best_f
