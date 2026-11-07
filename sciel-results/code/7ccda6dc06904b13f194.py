"""
Root finder for 3 fixed 2D systems using a global-to-local strategy.
1. Global phase: Coarse grid search to find a point near a root.
2. Local phase: Broyden's method with backtracking line search to converge.
3. Fallback: Nelder-Mead on ||F||^2 if local solver fails.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    x0 = np.array(x0, dtype=float)
    best_x = x0.copy()
    best_norm = float('inf')
    
    calls = 0
    max_calls = 490  # Safety margin below 500
    
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

    def update_best(x, norm):
        nonlocal best_x, best_norm
        if norm < best_norm:
            best_norm = norm
            best_x = x.copy()

    # 1. Initial Evaluation
    Fx0 = safe_F(x0)
    if Fx0 is None:
        return best_x
    norm0 = np.linalg.norm(Fx0)
    update_best(x0, norm0)
    if norm0 <= 1e-8:
        return best_x

    # 2. Global Search Phase
    # Determine a suitable scale for the grid
    # We check scales: 1, 10, 100, 1000, 10000
    # We want to find a region where F varies.
    
    scales = [1.0, 10.0, 100.0, 1000.0, 10000.0]
    best_scale = 1.0
    min_norm_found = norm0
    
    # Quick scale detection
    for s in scales:
        if time.time() > deadline or calls >= max_calls - 50:
            break
        # Check a few points at this scale
        # Just check x0 + s * [1, 0] and x0 + s * [0, 1] to see if F changes
        # If F is similar, the scale might be too small or too large.
        # Actually, we just want to find a point with lower norm.
        
        # Let's do a small ring search at scale s
        # 4 points: N, S, E, W
        for dx, dy in [(s, 0), (-s, 0), (0, s), (0, -s)]:
            x_try = x0 + np.array([dx, dy])
            if not is_valid(x_try):
                continue
            Fx_try = safe_F(x_try)
            if Fx_try is None:
                continue
            norm_try = np.linalg.norm(Fx_try)
            update_best(x_try, norm_try)
            if norm_try < min_norm_found:
                min_norm_found = norm_try
                best_scale = s
                # If we found a significantly better point, we might want to center the grid there
                # But for simplicity, we keep x0 as center for now, or update center?
                # Let's update the center for the grid search to the best point found so far.
                # Actually, let's just record the best point and scale.
        
        # If we found a much better point, we can break early and use that scale for a denser grid around that point
        if min_norm_found < 1e-4:
            break

    # Perform a denser grid search around the best point found so far
    # Use best_scale to define the grid extent
    # Grid size: 5x5 or 7x7?
    # 7x7 = 49 calls. This is affordable.
    
    center = best_x.copy()
    extent = best_scale * 2.0  # Search 2x the detected scale around the best point
    
    # If best_scale is very large, the grid might be huge. Cap the extent.
    # If best_scale is very small, the grid might be tiny.
    
    # Let's do a 5x5 grid centered at best_x
    # Step size = extent / 4
    step = extent / 4.0
    if step == 0:
        step = 1.0
    
    grid_points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            x_try = center + np.array([i * step, j * step])
            if is_valid(x_try):
                grid_points.append(x_try)
    
    for x_try in grid_points:
        if time.time() > deadline or calls >= max_calls - 20:
            break
        Fx_try = safe_F(x_try)
        if Fx_try is None:
            continue
        norm_try = np.linalg.norm(Fx_try)
        update_best(x_try, norm_try)
        if norm_try <= 1e-8:
            return x_try

    # 3. Local Polish Phase
    # Use Broyden's method starting from best_x
    
    def try_broyden(x_start, Fx_start, max_iter=100):
        nonlocal best_x, best_norm, calls
        x = x_start.copy()
        Fx = Fx_start.copy()
        norm = np.linalg.norm(Fx)
        
        if norm <= 1e-8:
            return x, norm, True
        
        # Compute initial Jacobian
        h = 1e-6
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_j = safe_F(x + e)
            if Fx_j is None:
                return x, norm, False
            J[:, j] = (Fx_j - Fx) / h
        
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
                        # Recompute Jacobian if step is tiny
                        h = 1e-6
                        J_new = np.zeros((2, 2))
                        for j in range(2):
                            e = np.zeros(2)
                            e[j] = h
                            Fx_j = safe_F(x_try + e)
                            if Fx_j is None:
                                break
                            J_new[:, j] = (Fx_j - Fx_try) / h
                        else:
                            J = J_new
                        if Fx_j is None:
                            break
                    
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

    x_result, norm_result, success = try_broyden(best_x, safe_F(best_x))
    if success:
        return x_result

    # 4. Fallback: Nelder-Mead
    # If Broyden failed, try Nelder-Mead on ||F||^2
    
    def nelder_mead(x_start, max_iters=200):
        nonlocal best_x, best_norm, calls
        
        def objective(x):
            try:
                Fx = safe_F(x)
                if Fx is None:
                    return 1e10
                return np.sum(Fx ** 2)
            except Exception:
                return 1e10
        
        x = np.array(x_start, dtype=float).copy()
        # Initial simplex
        delta = 0.1 * (np.linalg.norm(x) + 1.0)
        if delta == 0: delta = 0.1
        simplex = [x.copy()]
        for i in range(2):
            xi = x.copy()
            xi[i] += delta
            simplex.append(xi)
        
        f_vals = [objective(v) for v in simplex]
        update_best(simplex[np.argmin(f_vals)], np.sqrt(min(f_vals)))
        
        alpha, gamma, rho = 1.0, 2.0, 0.5
        
        for it in range(max_iters):
            if time.time() > deadline or calls >= max_calls - 10:
                break
            if min(f_vals) <= 1e-16:
                idx = np.argmin(f_vals)
                return simplex[idx], np.sqrt(f_vals[idx]), True
            
            order = np.argsort(f_vals)
            simplex = [simplex[i] for i in order]
            f_vals = [f_vals[i] for i in order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_ref = centroid + alpha * (centroid - simplex[-1])
            f_ref = objective(x_ref)
            update_best(x_ref, np.sqrt(f_ref))
            
            if f_vals[0] <= f_ref < f_vals[-2]:
                simplex[-1] = x_ref
                f_vals[-1] = f_ref
            elif f_ref < f_vals[0]:
                x_exp = centroid + gamma * (x_ref - centroid)
                f_exp = objective(x_exp)
                update_best(x_exp, np.sqrt(f_exp))
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    f_vals[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    f_vals[-1] = f_ref
            else:
                x_con = centroid + rho * (simplex[-1] - centroid)
                f_con = objective(x_con)
                update_best(x_con, np.sqrt(f_con))
                if f_con < f_vals[-1]:
                    simplex[-1] = x_con
                    f_vals[-1] = f_con
                else:
                    for i in range(1, 3):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        f_vals[i] = objective(simplex[i])
                        update_best(simplex[i], np.sqrt(f_vals[i]))
        
        idx = np.argmin(f_vals)
        return simplex[idx], np.sqrt(f_vals[idx]), f_vals[idx] <= 1e-16

    if time.time() < deadline - 5 and calls < max_calls - 20:
        x_result, norm_result, success = nelder_mead(best_x, max_iters=150)
        if success:
            return x_result

    return best_x
# EVOLVE-BLOCK-END
