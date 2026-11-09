import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        n_calls = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        best_x = np.zeros(dim)
        best_f = float('inf')

        def call_f(x):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget:
                return None
            n_calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def estimate_gradient(x, h=1e-4):
            """
            Estimate gradient using central finite differences.
            Returns gradient vector or None if budget exceeded.
            """
            nonlocal n_calls
            grad = np.zeros(dim)
            for i in range(dim):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += h
                x_minus[i] -= h
                
                # Clip to box
                x_plus = np.clip(x_plus, -5.0, 5.0)
                x_minus = np.clip(x_minus, -5.0, 5.0)
                
                # Check if we have enough budget for 2 more calls
                if n_calls + 2 > budget:
                    return None
                
                fp = call_f(x_plus)
                if fp is None: return None
                fm = call_f(x_minus)
                if fm is None: return None
                
                # If clipping changed the point, the difference might be 0.
                # We handle this by checking if the step was effectively taken.
                # If x_plus[i] == x_minus[i] (due to clipping at boundary), grad is 0 or undefined.
                # For simplicity, if the effective step is 0, we set grad component to 0.
                effective_step = x_plus[i] - x_minus[i]
                if effective_step > 1e-10:
                    grad[i] = (fp - fm) / effective_step
                else:
                    # At boundary, use one-sided if possible, or 0.
                    # If x[i] is at max, we can't go up. Use down.
                    if x[i] >= 5.0 - 1e-8:
                         x_down = x.copy()
                         x_down[i] -= h
                         x_down = np.clip(x_down, -5.0, 5.0)
                         if n_calls + 1 > budget: return None
                         fd = call_f(x_down)
                         if fd is None: return None
                         eff_step = x[i] - x_down[i]
                         if eff_step > 1e-10:
                             grad[i] = (f_val - fd) / eff_step # approx
                         else:
                             grad[i] = 0
                    elif x[i] <= -5.0 + 1e-8:
                         x_up = x.copy()
                         x_up[i] += h
                         x_up = np.clip(x_up, -5.0, 5.0)
                         if n_calls + 1 > budget: return None
                         fu = call_f(x_up)
                         if fu is None: return None
                         eff_step = x_up[i] - x[i]
                         if eff_step > 1e-10:
                             grad[i] = (fu - f_val) / eff_step
                         else:
                             grad[i] = 0
                    else:
                         grad[i] = 0
            return grad

        def df_bfgs(x0, max_calls=100):
            """
            Derivative-Free BFGS optimization.
            """
            nonlocal n_calls, best_x, best_f
            x = x0.copy()
            fx = call_f(x)
            if fx is None: return best_x, best_f
            
            # Initial inverse Hessian approximation H_k = I
            H = np.eye(dim)
            
            # Initial step
            p = -H @ estimate_gradient(x)
            if p is None: return best_x, best_f
            
            # Backtracking line search parameters
            c1 = 1e-4
            rho = 0.5
            max_ls_iter = 10
            ls_step = 1.0
            
            prev_x = x.copy()
            prev_fx = fx
            prev_p = p.copy()
            prev_grad = None
            
            for k in range(max_calls):
                if time.time() > deadline - 5:
                    break
                if n_calls >= budget:
                    break
                
                # Line search to find alpha such that f(x + alpha*p) < f(x) + c1 * alpha * grad^T p
                # We need gradient at x. We just estimated it? No, we estimated it to get p.
                # Let's re-estimate or store it. For efficiency, we assume the gradient used for p is close enough
                # for the Armijo condition in the first step, but for BFGS update we need s and y.
                
                # Actually, standard BFGS needs grad at x. We have 'p' which was -H @ grad_prev.
                # So grad_prev = -H^{-1} @ p? No, H is inverse Hessian. p = -H @ grad.
                # So grad = H^{-1} @ (-p). This is expensive.
                # Let's just re-estimate gradient at current x for the line search condition? 
                # That costs dim*2 calls.
                # Alternative: Use a simpler line search that just checks improvement.
                
                found_alpha = False
                alpha = ls_step
                for _ in range(max_ls_iter):
                    x_new = x + alpha * p
                    x_new = np.clip(x_new, -5.0, 5.0)
                    fx_new = call_f(x_new)
                    if fx_new is None:
                        return best_x, best_f
                    
                    # Simple improvement check
                    if fx_new < fx:
                        found_alpha = True
                        break
                    else:
                        alpha *= rho
                        if alpha < 1e-6:
                            break
                
                if not found_alpha:
                    # No improvement, try a random small perturbation or break
                    # In DF-BFGS, if line search fails, we might restart or shrink step
                    # For simplicity, break and rely on multi-start
                    break
                
                # Update variables
                s = x_new - x
                y = None # We need gradient difference. 
                # To get y = grad_new - grad_old, we need grad_new.
                # This is the cost of DF-BFGS: 2 * dim * 2 calls per iteration (2 grads, 1 func eval per line search step)
                # This is very expensive. 500 calls / (10 * 2) = 25 iterations max.
                
                # Optimization: We can skip the BFGS update if we don't have enough budget for the next gradient.
                # But without the update, it's just gradient descent with fixed step, which is slow.
                
                # Let's estimate grad at x_new
                if n_calls + 2 * dim <= budget:
                    grad_new = estimate_gradient(x_new)
                    if grad_new is None:
                        return best_x, best_f
                    # We also need grad_old. We didn't store it explicitly, but we used it to compute p.
                    # Let's store grad_old before computing p.
                    # Refactoring: 
                    # 1. Estimate grad at x (2*dim calls)
                    # 2. Compute p = -H @ grad
                    # 3. Line search (1 call per step)
                    # 4. Estimate grad at x_new (2*dim calls)
                    # 5. Update H
                    # Total per iter: ~4*dim + 1 calls = 21 calls.
                    # 500 / 21 ~ 23 iterations.
                    
                    # We need grad_old. Let's re-estimate it? No, that's wasteful.
                    # Let's modify the loop to store grad.
                    
                    # For now, let's just use a simpler method if BFGS is too complex/costly.
                    # Given the budget, maybe a hybrid is better?
                    # The prompt asks for DF-BFGS.
                    
                    # To make this work within budget, we must be efficient.
                    # We will re-structure:
                    # 1. grad_old = estimate_gradient(x)
                    # 2. p = -H @ grad_old
                    # 3. Line search
                    # 4. grad_new = estimate_gradient(x_new)
                    # 5. s = x_new - x, y = grad_new - grad_old
                    # 6. Update H
                    
                    # Since we already did line search, we have x_new, fx_new.
                    # We need grad_old. We didn't store it. 
                    # This implementation is getting messy. Let's restart the DF-BFGS function logic properly.
                    
                    return best_x, best_f # Abort this complex path if not careful
                    
                else:
                    # Not enough budget for gradient, just accept x_new as new best and stop
                    if fx_new < best_f:
                        best_f = fx_new
                        best_x = x_new.copy()
                    break
            
            return best_x, best_f

        # The above DF-BFGS implementation has a flaw: it doesn't store grad_old.
        # Let's rewrite the DF-BFGS logic correctly inside the call.
        
        def robust_df_bfgs(x0, max_iters=20):
            nonlocal n_calls, best_x, best_f
            x = x0.copy()
            fx = call_f(x)
            if fx is None: return best_x, best_f
            
            H = np.eye(dim)
            
            for k in range(max_iters):
                if time.time() > deadline - 5 or n_calls >= budget:
                    break
                
                # 1. Estimate gradient at x
                if n_calls + 2 * dim > budget:
                    break
                grad = estimate_gradient(x)
                if grad is None: break
                
                # 2. Search direction
                p = -H @ grad
                
                # 3. Line search
                alpha = 1.0
                found = False
                for _ in range(5):
                    x_new = x + alpha * p
                    x_new = np.clip(x_new, -5.0, 5.0)
                    if n_calls >= budget: break
                    fx_new = call_f(x_new)
                    if fx_new is None: break
                    if fx_new < fx:
                        found = True
                        break
                    alpha *= 0.5
                    if alpha < 1e-6: break
                
                if not found:
                    break
                
                # 4. Estimate gradient at x_new
                if n_calls + 2 * dim > budget:
                    # Accept x_new as best but can't update H
                    if fx_new < best_f:
                        best_f = fx_new
                        best_x = x_new.copy()
                    break
                
                grad_new = estimate_gradient(x_new)
                if grad_new is None: break
                
                # 5. BFGS Update
                s = x_new - x
                y = grad_new - grad
                sy = s @ y
                if sy > 1e-8:
                    Hy = H @ y
                    H = H + ((sy + y @ Hy) / (sy**2)) * np.outer(s, s) - (1/sy) * np.outer(Hy, s)
                    # Symmetrize
                    H = (H + H.T) / 2.0
                    
                # 6. Update state
                x = x_new
                fx = fx_new
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                    
            return best_x, best_f

        # Strategy:
        # 1. Start at 0.
        # 2. Run robust_df_bfgs.
        # 3. If budget remains, try a few random starts with DF-BFGS.
        
        # Initial start
        robust_df_bfgs(np.zeros(dim), max_iters=15)
        
        # Restarts
        # DF-BFGS is expensive (~20-30 calls per iter). 
        # With 500 calls, we get ~15-20 iterations per run.
        # We can afford 1-2 restarts if the first one consumes most of the budget.
        
        if n_calls < budget - 50 and time.time() < deadline - 10:
             # Try one random restart
             x_rand = rng.uniform(-4, 4, dim)
             robust_df_bfgs(x_rand, max_iters=10)
             
        # Final polish with (1+1)-ES if budget remains
        remaining = budget - n_calls
        if remaining > 20 and time.time() < deadline - 2:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for _ in range(remaining):
                if time.time() > deadline - 1:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = call_f(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                    if fx < best_f:
                        best_x = x.copy()
                        best_f = fx
                else:
                    sigma *= 0.9
                    sigma = max(sigma, 1e-4)

        return best_x, best_f
