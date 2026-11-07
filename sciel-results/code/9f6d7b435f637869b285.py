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
    1. Broyden's method with backtracking line search (efficient, ~2 calls/iter).
    2. Multi-start: if initial Broyden fails, try several candidate starting points.
    3. Nelder-Mead on ||F||^2 as a last resort.
    
    Call budget: stay well under 500 calls.
    Time budget: ~140 seconds (leaving margin for the evaluator's 150s limit).
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    x = np.array(x0, dtype=float).copy()
    best_x = x.copy()
    best_norm = float('inf')
    
    calls = 0
    max_calls = 490  # safety margin below 500
    
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
    
    # Initial evaluation
    Fx = safe_F(x)
    if Fx is None:
        return best_x
    norm = np.linalg.norm(Fx)
    best_norm = norm
    best_x = x.copy()
    if norm <= 1e-8:
        return best_x
    
    # Helper to compute Jacobian via finite differences
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
    
    def try_broyden(x_start, Fx_start, max_iter=100):
        nonlocal best_x, best_norm, calls
        x = x_start.copy()
        Fx = Fx_start.copy()
        norm = np.linalg.norm(Fx)
        
        if norm <= 1e-8:
            return x, norm, True
        
        # Compute initial Jacobian
        J = compute_jacobian(x, Fx)
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
                Fx_try = safe_F(x_try)
                if Fx_try is None:
                    alpha *= 0.5
                    continue
                
                norm_try = np.linalg.norm(Fx_try)
                
                if norm_try < best_norm:
                    best_norm = norm_try
                    best_x = x_try.copy()
                
                if norm_try < Fx_old_norm * 0.9:  # Sufficient decrease
                    # Update Jacobian (Broyden)
                    s = alpha * dx
                    y = Fx_try - Fx
                    J_s = J @ s
                    diff = y - J_s
                    denom = s @ s
                    if denom > 1e-15:
                        J = J + np.outer(diff, s) / denom
                    else:
                        J_new = compute_jacobian(x_try, Fx_try)
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
            
            if norm <= 1e-8:
                return x, norm, True
        
        return x, norm, norm <= 1e-8
    
    def nelder_mead(x, max_iters=200):
        nonlocal best_x, best_norm, calls
        def objective(x):
            try:
                Fx = safe_F(x)
                if Fx is None:
                    return 1e10
                return np.sum(Fx ** 2)
            except Exception:
                return 1e10
        
        x = np.array(x, dtype=float).copy()
        # Initial simplex
        delta = 0.1
        simplex = [x.copy()]
        for i in range(2):
            xi = x.copy()
            xi[i] += delta
            simplex.append(xi)
        
        f_vals = [objective(v) for v in simplex]
        
        alpha, gamma, rho = 1.0, 2.0, 0.5
        
        for it in range(max_iters):
            if time.time() > deadline or calls >= max_calls - 10:
                break
            if min(f_vals) <= 1e-16:  # norm^2 <= 1e-16 means norm <= 1e-8
                idx = np.argmin(f_vals)
                return simplex[idx], np.sqrt(f_vals[idx]), True
            
            # Sort simplex
            order = np.argsort(f_vals)
            simplex = [simplex[i] for i in order]
            f_vals = [f_vals[i] for i in order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_ref = centroid + alpha * (centroid - simplex[-1])
            f_ref = objective(x_ref)
            
            if f_vals[0] <= f_ref < f_vals[-2]:
                simplex[-1] = x_ref
                f_vals[-1] = f_ref
            elif f_ref < f_vals[0]:
                # Expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                f_exp = objective(x_exp)
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    f_vals[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    f_vals[-1] = f_ref
            else:
                # Contraction
                x_con = centroid + rho * (simplex[-1] - centroid)
                f_con = objective(x_con)
                if f_con < f_vals[-1]:
                    simplex[-1] = x_con
                    f_vals[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, 3):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        f_vals[i] = objective(simplex[i])
        
        idx = np.argmin(f_vals)
        return simplex[idx], np.sqrt(f_vals[idx]), f_vals[idx] <= 1e-16
    
    # Try Broyden from initial point
    x_result, norm_result, success = try_broyden(x, Fx, max_iter=150)
    if success:
        return x_result
    
    # Multi-start strategy: try different initial points
    candidates = []
    
    # Points along coordinate axes from x0
    for scale in [0.5, 2.0, 5.0, 10.0]:
        for sign in [-1, 1]:
            for axis in range(2):
                x_cand = x0.copy()
                x_cand[axis] += sign * scale * (abs(x0[axis]) + 1.0)
                candidates.append(x_cand)
    
    # Points based on gradient direction (approximate)
    if norm > 0:
        grad_dir = -Fx / norm
        for scale in [1.0, 2.0, 5.0]:
            x_cand = x + scale * grad_dir
            candidates.append(x_cand)
    
    # Random points in a neighborhood
    rng = np.random.RandomState(42)
    for _ in range(10):
        x_cand = x + rng.randn(2) * (np.linalg.norm(x) + 1.0)
        candidates.append(x_cand)
    
    # Try each candidate with limited Broyden iterations
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
        
        # Try Broyden from this candidate with fewer iterations
        x_result, norm_result, success = try_broyden(x_cand, Fx_cand, max_iter=50)
        if success:
            return x_result
    
    # Try Nelder-Mead as last resort
    if time.time() < deadline - 5 and calls < max_calls - 20:
        x_result, norm_result, success = nelder_mead(x, max_iters=150)
        if success:
            return x_result
    
    # Return the best result found
    return best_x
# EVOLVE-BLOCK-END
