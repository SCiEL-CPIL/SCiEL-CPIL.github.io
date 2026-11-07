import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """L-BFGS-B optimizer for BBOB-style functions with global initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        
        # State for best solution
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # We need to track calls to ensure we don't exceed budget
        # and to handle the "return value ignored" but "calls recorded" constraint.
        
        # Helper to call func and track budget
        def tracked_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                # If we exceed budget, we should ideally not call, 
                # but scipy might call it. 
                # The task says "More than 500 calls ... makes the whole result invalid".
                # So we must strictly prevent the (budget+1)th call.
                # However, scipy's minimize usually respects maxiter/feval limits.
                # To be safe, if we hit the limit, we return a large value to stop progress?
                # No, better to structure the loop so we don't exceed.
                # If we are exactly at budget, we cannot call again.
                # But scipy might try. 
                # Strategy: Use a wrapper that raises an exception or returns inf?
                # If we return inf, scipy might handle it, but the call was made.
                # The task says "More than 500 calls ... invalid".
                # So we MUST ensure the 501st call never happens.
                
                # This implies we need to control the number of calls strictly.
                # L-BFGS-B in scipy allows maxiter and ftol, but feval is not a direct stop condition 
                # that prevents the call itself if it's the last one allowed.
                
                # Safer approach: 
                # 1. Estimate how many calls a minimize step will take? Hard.
                # 2. Use a custom loop? L-BFGS-B is internal.
                # 3. Use a wrapper that counts, and if count == budget, we can't call.
                #    But if scipy calls it, it's too late.
                
                # Let's look at the constraint: "More than 500 calls ... invalid".
                # This means calls_made must be <= 500.
                # If I call func when calls_made == 500, it becomes 501. Invalid.
                
                # Therefore, I must ensure that when calls_made == 500, no further calls are made.
                # This is tricky with scipy because it decides when to stop.
                
                # Alternative: Use a smaller budget for scipy, leaving a margin?
                # Or, catch the exception? No, exception later keeps calls made.
                
                # Let's use a strict counter. If calls_made >= budget, we return best_f (or large value) 
                # WITHOUT calling func? 
                # But scipy expects a float. If we return a cached value, we don't call func.
                # This is a "fake" evaluation. Does it violate "Every call is recorded"?
                # "Every call [to func] is recorded". If I don't call func, it's not recorded.
                # So I can return a dummy value to stop scipy from exploring further, 
                # as long as I don't actually call func.
                
                # However, if I return a dummy value, scipy might think it found a minimum or failed.
                # If I return best_f, it might converge.
                
                # Let's implement this "soft" stop.
                return best_f if best_f != np.inf else 1e100
            
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            calls_made += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
                
            return f_val

        # Gradient estimation via central differences
        # This is expensive: 2*dim calls per gradient.
        # dim=5, so 10 calls per gradient.
        # L-BFGS-B typically needs ~10-20 iterations for Rosenbrock.
        # 20 * (1 func + 10 grad) = 220 calls.
        # Budget 500 allows for multiple restarts or a more thorough search.
        
        def gradient(x):
            # Check budget before computing gradient
            # We need 2*dim calls for central diff.
            if calls_made + 2 * dim > budget:
                # Not enough budget for a full gradient step.
                # Return zero gradient to stop? Or use forward diff (dim calls)?
                # If calls_made + dim > budget, we can't even do forward diff.
                # If we return zeros, L-BFGS-B might stop or behave oddly.
                # Let's try to be as efficient as possible.
                # If we have at least dim calls left, use forward diff.
                if calls_made + dim > budget:
                    return np.zeros(dim)
                
                # Forward difference
                eps = 1e-5
                grad = np.zeros(dim)
                for i in range(dim):
                    x_pert = x.copy()
                    x_pert[i] += eps
                    f_pert = tracked_func(x_pert)
                    # We already have f(x) from the previous call? 
                    # No, tracked_func doesn't cache. 
                    # We should reuse f(x) if possible.
                    # But tracked_func is a black box wrapper.
                    # Let's just call it. It's 1 + dim calls per gradient with forward diff.
                    grad[i] = (f_pert - f_val_current) / eps
                return grad

        # To make gradient efficient, we need f(x) value.
        # L-BFGS-B calls func(x) then grad(x).
        # We can cache the last x and f(x).
        
        last_x = None
        last_f = None

        def tracked_func_with_cache(x):
            nonlocal calls_made, best_f, best_x, last_x, last_f
            if calls_made >= budget:
                return best_f if best_f != np.inf else 1e100
            
            # Check if x is same as last_x (within tolerance)
            # This helps if scipy calls func then grad immediately.
            # But we don't know if grad will use the same x.
            # Standard practice: func(x) is called, then grad(x) is called.
            # If we cache f(x) in func, we can use it in grad.
            
            x_clipped = np.clip(x, -5, 5)
            
            # Simple cache check: if x is very close to last_x, reuse last_f?
            # Risky if x is slightly different.
            # Let's just call it. The overhead of 1 extra call per gradient step is acceptable.
            # Actually, scipy's L-BFGS-B calls func(x) and then grad(x).
            # If we implement grad using forward differences, we need f(x).
            # If we call func(x) again inside grad, that's 1 (from scipy) + 1 (in grad) + dim (perturbations) = dim+2 calls per step.
            # dim=5 -> 7 calls per step.
            # 500 / 7 = ~70 steps. Plenty.
            
            f_val = func(x_clipped)
            calls_made += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
                
            last_x = x_clipped.copy()
            last_f = f_val
            return f_val

        def gradient_with_cache(x):
            nonlocal calls_made, last_x, last_f
            # If we have budget for forward diff (dim calls)
            if calls_made + dim > budget:
                return np.zeros(dim)
                
            eps = 1e-5
            grad = np.zeros(dim)
            # Use last_f if x is close to last_x?
            # To be safe, let's just assume the last call to tracked_func_with_cache was for this x.
            # But what if it wasn't?
            # Let's just call tracked_func_with_cache(x) again? No, that wastes a call.
            # Let's store the f value from the func call.
            # But gradient is a separate method.
            # We can pass x to gradient, and inside gradient, if we don't have f(x), we must compute it.
            # But we already computed it in func(x).
            # How to share state? Global/nonlocal variables.
            
            # If last_x is close to x, use last_f.
            if last_x is not None and np.allclose(last_x, x, rtol=1e-05, atol=1e-08):
                f0 = last_f
            else:
                # We need f(x). But we can't call func again if we are tight on budget?
                # Actually, if we are here, we have at least dim calls left for perturbations.
                # Do we have 1 more for f(x)?
                if calls_made + 1 + dim > budget:
                    return np.zeros(dim)
                f0 = tracked_func_with_cache(x)
                
            for i in range(dim):
                x_pert = x.copy()
                x_pert[i] += eps
                f_pert = tracked_func_with_cache(x_pert)
                grad[i] = (f_pert - f0) / eps
                
            return grad

        # Global search strategy:
        # 1. Start at center (0,0,0,0,0) or random points.
        # 2. Run L-BFGS-B.
        # 3. If budget remains, restart from new random points or best found so far with perturbation.
        
        rng = np.random.default_rng(42)
        
        # Initial guess: Center of the box is often good for Rosenbrock (min at 1,1..1)
        # But the problem says minimizer lies in [-4, 4]^5.
        # Rosenbrock min is at (1,1,1,1,1) usually, but rotated/shifted.
        # Let's try a few starting points.
        
        # Strategy: 
        # Use ~80% of budget for the first run from a good guess.
        # Use remaining for restarts if needed.
        
        # Actually, L-BFGS-B is very efficient. 
        # Let's just run it once with maxiter high, and let the budget cap handle it.
        # But we need to ensure we don't exceed budget.
        
        # We will use a loop of restarts.
        # Each restart uses a fraction of the remaining budget.
        
        # Starting points:
        # 1. (1, 1, 1, 1, 1) - standard Rosenbrock min location
        # 2. (0, 0, 0, 0, 0)
        # 3. Random points
        
        start_points = [
            np.ones(dim),
            np.zeros(dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim)
        ]
        
        # We will try to fit as many "full" L-BFGS-B runs as possible.
        # A "full" run might take 50-100 calls.
        
        for x0 in start_points:
            if time.time() > deadline - 5:
                break
            if calls_made >= budget:
                break
                
            # Reset cache for new start
            last_x = None
            last_f = None
            
            # Run L-BFGS-B
            # maxiter: set high, we rely on budget cap in tracked_func
            # But tracked_func returns best_f when budget is hit, which might confuse L-BFGS-B.
            # Better to set a maxiter that is likely to stay within budget.
            # Or just let it run and catch the "soft stop".
            
            try:
                res = minimize(
                    tracked_func_with_cache, 
                    x0, 
                    method='L-BFGS-B', 
                    jac=gradient_with_cache,
                    bounds=[(-5, 5)] * dim,
                    options={
                        'maxiter': 100, # Limit iterations to avoid infinite loops
                        'ftol': 1e-12,
                        'gtol': 1e-8
                    }
                )
            except Exception:
                pass

            # After each run, check if we improved.
            # If we have budget left, we could restart from res.x with a small perturbation?
            # Or just try the next start point.
            
            # If we are very close to the end of budget, break.
            if calls_made > budget * 0.8:
                # Do one more local refinement from best_x if budget allows
                if calls_made + 10 < budget and best_x is not None:
                    last_x = None
                    last_f = None
                    try:
                        res = minimize(
                            tracked_func_with_cache, 
                            best_x, 
                            method='L-BFGS-B', 
                            jac=gradient_with_cache,
                            bounds=[(-5, 5)] * dim,
                            options={
                                'maxiter': 20,
                                'ftol': 1e-14,
                                'gtol': 1e-10
                            }
                        )
                    except Exception:
                        pass
                break

        # Final check: if we have some budget left, do a few random restarts?
        # Probably not needed if L-BFGS-B converged.
        
        # Return best found
        if best_x is None:
            # Fallback
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1
            
        return best_x, best_f
