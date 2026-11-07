"""Combined root-finder: Broyden's method with finite-difference Jacobian,
backtracking line search, multi-start strategy, and Nelder-Mead fallback.
Uses a call counter to stay within budget and a time deadline for safety."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Homotopy/Continuation: Solve F_t(x) = t*F(x) + (1-t)*(x-x0) = 0 for t from 0.1 to 1.0.
    2. Local solver: Broyden's method with backtracking line search.
    3. Fallback: Multi-start Broyden on original F(x) if homotopy fails.
    
    Call budget: stay well under 500 calls.
    Time budget: ~140 seconds.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    x0_arr = np.array(x0, dtype=float).copy()
    best_x = x0_arr.copy()
    best_norm = float('inf')
    
    calls = 0
    max_calls = 490
    
    def safe_F(x):
        nonlocal calls
        if time.time() > deadline:
            raise TimeoutError
        if calls >= max_calls:
            raise RuntimeError("Call limit exceeded")
        try:
            Fx = F(x)
            Fx = np.asarray(Fx, dtype=float)
            if Fx.shape != (2,):
                Fx = Fx.reshape(2)
            if not np.all(np.isfinite(Fx)):
                return None
            calls += 1
            return Fx
        except Exception:
            return None

    def is_valid(x):
        return np.all(np.isfinite(x)) and x.shape == (2,)

    # Wrapper for homotopy
    def F_t(x, t):
        # G(x) = t * F(x) + (1-t) * (x - x0)
        Fx = safe_F(x)
        if Fx is None:
            return None
        return t * Fx + (1 - t) * (x - x0_arr)

    def compute_jacobian(x, Fx, h=1e-6):
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_j = safe_F(x + e)
            if Fx_j is None:
                return None
            J[:, j] = (Fx_j - Fx) / h
        return J

    def try_broyden_general(x_start, F_func, max_iter=100, tol=1e-10):
        """
        Solves F_func(x) = 0 using Broyden's method.
        F_func(x) should return F(x) or None.
        """
        nonlocal best_x, best_norm, calls
        x = x_start.copy()
        
        # Initial evaluation
        Fx = F_func(x)
        if Fx is None:
            return x, float('inf'), False
        norm = np.linalg.norm(Fx)
        
        if norm < best_norm:
            best_norm = norm
            best_x = x.copy()
            
        if norm <= tol:
            return x, norm, True
        
        # Compute initial Jacobian
        # Note: F_func might be F_t, so we need to compute J of F_func.
        # But safe_F only calls F. 
        # For homotopy, J of F_t = t * J_F + (1-t) * I.
        # It is safer to finite difference F_func directly if we can.
        # But safe_F counts calls. 
        # If we finite difference F_func, we call safe_F 3 times.
        # This is consistent.
        
        J = compute_jacobian(x, Fx) # This calls safe_F, which is F(x). 
        # Wait, compute_jacobian uses safe_F, which calls F(x).
        # But F_func is t*F + (1-t)*(x-x0).
        # The Jacobian of F_func is t*J_F + (1-t)*I.
        # If we use compute_jacobian on F_func, it will finite difference F_func.
        # But my compute_jacobian helper calls safe_F(x+e) which returns F(x+e), not F_func(x+e).
        # So I need a generic finite difference for F_func.
        
        def compute_jacobian_generic(x, F_func):
            J = np.zeros((2, 2))
            h = 1e-6
            Fx_center = F_func(x)
            if Fx_center is None: return None
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fx_e = F_func(x + e)
                if Fx_e is None: return None
                J[:, j] = (Fx_e - Fx_center) / h
            return J

        J = compute_jacobian_generic(x, F_func)
        if J is None:
            return x, norm, False

        for _ in range(max_iter):
            if time.time() > deadline or calls >= max_calls - 5:
                break
            
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except Exception:
                    break
            
            if not np.all(np.isfinite(dx)):
                break
            
            # Backtracking line search
            alpha = 1.0
            Fx_old_norm = norm
            step_accepted = False
            
            for _ in range(15):
                x_try = x + alpha * dx
                if not is_valid(x_try):
                    alpha *= 0.5
                    continue
                Fx_try = F_func(x_try)
                if Fx_try is None:
                    alpha *= 0.5
                    continue
                
                norm_try = np.linalg.norm(Fx_try)
                
                if norm_try < best_norm:
                    best_norm = norm_try
                    best_x = x_try.copy()
                
                if norm_try < Fx_old_norm * 0.9:
                    # Update Jacobian
                    s = alpha * dx
                    y = Fx_try - Fx
                    J_s = J @ s
                    diff = y - J_s
                    denom = s @ s
                    if denom > 1e-15:
                        J = J + np.outer(diff, s) / denom
                    else:
                        J_new = compute_jacobian_generic(x_try, F_func)
                        if J_new is None:
                            break
                        J = J_new
                    
                    x = x_try
                    Fx = Fx_try
                    norm = norm_try
                    step_accepted = True
                    break
                else:
                    alpha *= 0.5
            
            if not step_accepted:
                break
            
            if norm <= tol:
                return x, norm, True
        
        return x, norm, norm <= tol

    # --- Homotopy Strategy ---
    
    # Start with t = 0.1
    t = 0.1
    x_current = x0_arr.copy()
    homotopy_success = False
    
    while t < 1.0:
        if time.time() > deadline - 10 or calls >= max_calls - 20:
            break
        
        # Define F_t for this t
        def F_t_func(x, t_val=t):
            return F_t(x, t_val)
            
        x_sol, norm_sol, success = try_broyden_general(x_current, F_t_func, max_iter=50, tol=1e-10)
        
        if success:
            x_current = x_sol
            # Increase t
            # Geometric increase is often better
            t = min(1.0, t * 1.5) 
            if t >= 1.0:
                homotopy_success = True
        else:
            # If this step fails, try smaller increment or break
            # If we can't even solve for small t, homotopy is unlikely to work
            break

    if homotopy_success:
        # Final check on original F
        Fx_final = safe_F(x_current)
        if Fx_final is not None:
            norm_final = np.linalg.norm(Fx_final)
            if norm_final <= 1e-8:
                return x_current
            # If it's close, try a few more Newton steps on original F
            # Or just return if it's the best we have
            if norm_final < best_norm:
                best_norm = norm_final
                best_x = x_current.copy()
            # Polish with Broyden on original F
            def F_orig_func(x):
                return safe_F(x)
            x_polish, norm_polish, success_polish = try_broyden_general(x_current, F_orig_func, max_iter=50, tol=1e-10)
            if success_polish:
                return x_polish
            if norm_polish < best_norm:
                best_norm = norm_polish
                best_x = x_polish.copy()
            return best_x

    # --- Fallback: Original Multi-start Broyden ---
    
    # Initial evaluation of original F
    Fx = safe_F(x0_arr)
    if Fx is None:
        return best_x
    norm = np.linalg.norm(Fx)
    best_norm = norm
    best_x = x0_arr.copy()
    if norm <= 1e-8:
        return best_x

    # Try Broyden from initial point
    def F_orig_func(x):
        return safe_F(x)
        
    x_result, norm_result, success = try_broyden_general(x0_arr, F_orig_func, max_iter=150, tol=1e-10)
    if success:
        return x_result
    
    # Multi-start strategy
    candidates = []
    for scale in [0.5, 2.0, 5.0, 10.0]:
        for sign in [-1, 1]:
            for axis in range(2):
                x_cand = x0_arr.copy()
                x_cand[axis] += sign * scale * (abs(x0_arr[axis]) + 1.0)
                candidates.append(x_cand)
    
    if norm > 0:
        grad_dir = -Fx / norm
        for scale in [1.0, 2.0, 5.0]:
            x_cand = x0_arr + scale * grad_dir
            candidates.append(x_cand)
    
    rng = np.random.RandomState(42)
    for _ in range(10):
        x_cand = x0_arr + rng.randn(2) * (np.linalg.norm(x0_arr) + 1.0)
        candidates.append(x_cand)
    
    for x_cand in candidates:
        if time.time() > deadline - 10 or calls >= max_calls - 10:
            break
        if not is_valid(x_cand):
            continue
        
        Fx_cand = safe_F(x_cand)
        if Fx_cand is None:
            continue
        
        norm_cand = np.linalg.norm(Fx_cand)
        if norm_cand < best_norm:
            best_norm = norm_cand
            best_x = x_cand.copy()
        
        if norm_cand <= 1e-8:
            return x_cand
        
        x_result, norm_result, success = try_broyden_general(x_cand, F_orig_func, max_iter=50, tol=1e-10)
        if success:
            return x_result
            
    return best_x
# EVOLVE-BLOCK-END
