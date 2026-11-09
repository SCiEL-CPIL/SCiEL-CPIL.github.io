import numpy as np
import time

def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8 using a hybrid strategy:
    1. Broyden's method (fast, Jacobian-based).
    2. Nelder-Mead (robust, derivative-free, minimizes ||F(x)||^2).
    3. Sparse Grid Search + Polish.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    def time_ok():
        return time.time() < deadline

    x0 = np.array(x0, dtype=float)
    call_count = 0
    MAX_CALLS = 480
    
    def F_counted(x):
        nonlocal call_count
        call_count += 1
        return F(x)
    
    best_x = x0.copy()
    try:
        best_norm = np.linalg.norm(F_counted(x0))
    except:
        best_norm = np.inf
        
    if best_norm <= 1e-8:
        return x0

    def update_best(x):
        nonlocal best_x, best_norm
        if not np.all(np.isfinite(x)):
            return
        try:
            fx = np.asarray(F_counted(x), dtype=float)
            if fx.shape != (2,) or not np.all(np.isfinite(fx)):
                return
            n = np.linalg.norm(fx)
            if np.isfinite(n) and n < best_norm:
                best_norm = n
                best_x = x.copy()
        except:
            pass

    def broyden_method(x_init, max_iter=20):
        """
        Broyden's method with backtracking.
        Returns True if root found.
        """
        if not time_ok() or call_count >= MAX_CALLS - 10:
            return False
            
        x = np.array(x_init, dtype=float)
        h = 1e-6
        
        try:
            Fx = np.asarray(F_counted(x), dtype=float)
            if not np.all(np.isfinite(Fx)):
                return False
            n = np.linalg.norm(Fx)
            update_best(x)
            if n <= 1e-8:
                return True
        except:
            return False

        try:
            Fx1 = np.asarray(F_counted(x + [h, 0]), dtype=float)
            Fx2 = np.asarray(F_counted(x + [0, h]), dtype=float)
            if not (np.all(np.isfinite(Fx1)) and np.all(np.isfinite(Fx2))):
                return False
            J = np.column_stack([(Fx1 - Fx) / h, (Fx2 - Fx) / h])
            
            # Regularize
            try:
                if np.linalg.cond(J) > 1e12:
                    J += np.eye(2) * 1e-8
            except:
                pass
        except:
            return False

        for _ in range(max_iter):
            if not time_ok() or call_count >= MAX_CALLS - 5:
                return False
                
            try:
                try:
                    s = np.linalg.solve(J, -Fx)
                except np.linalg.LinAlgError:
                    s = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                    
                if not np.all(np.isfinite(s)):
                    return False
                    
                alpha = 1.0
                success = False
                for _ in range(10):
                    x_trial = x + alpha * s
                    if not np.all(np.isfinite(x_trial)):
                        alpha *= 0.5
                        continue
                    Fx_trial = np.asarray(F_counted(x_trial), dtype=float)
                    if not np.all(np.isfinite(Fx_trial)):
                        alpha *= 0.5
                        continue
                    n_trial = np.linalg.norm(Fx_trial)
                    if n_trial < n:
                        x = x_trial
                        Fx = Fx_trial
                        n = n_trial
                        success = True
                        update_best(x)
                        break
                    alpha *= 0.5
                
                if not success:
                    return False
                    
                if n <= 1e-8:
                    return True
                    
                # Broyden Update
                y = Fx_trial - Fx # Wait, Fx was updated. Need old Fx?
                # In the loop above, if success, Fx is updated to Fx_trial.
                # We need the difference from the previous step.
                # Let's re-structure to keep old_Fx.
                
                # Re-doing Broyden logic cleanly
                return False # Placeholder to ensure structure is distinct if needed, 
                             # but actually I should implement it correctly.
            except:
                return False
        
        return False

    def newton_broyden(x_init):
        """
        Corrected Broyden implementation.
        """
        if not time_ok() or call_count >= MAX_CALLS - 10:
            return False
            
        x = np.array(x_init, dtype=float)
        h = 1e-5
        
        try:
            Fx = np.asarray(F_counted(x), dtype=float)
            if not np.all(np.isfinite(Fx)):
                return False
            n = np.linalg.norm(Fx)
            update_best(x)
            if n <= 1e-8:
                return True
        except:
            return False

        try:
            Fx1 = np.asarray(F_counted(x + [h, 0]), dtype=float)
            Fx2 = np.asarray(F_counted(x + [0, h]), dtype=float)
            if not (np.all(np.isfinite(Fx1)) and np.all(np.isfinite(Fx2))):
                return False
            J = np.column_stack([(Fx1 - Fx) / h, (Fx2 - Fx) / h])
            
            try:
                if np.linalg.cond(J) > 1e10:
                    J += np.eye(2) * 1e-6
            except:
                pass
        except:
            return False

        for i in range(30):
            if not time_ok() or call_count >= MAX_CALLS - 5:
                return False
                
            try:
                try:
                    s = np.linalg.solve(J, -Fx)
                except np.linalg.LinAlgError:
                    s = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                    
                if not np.all(np.isfinite(s)):
                    return False
                    
                alpha = 1.0
                success = False
                x_old = x.copy()
                Fx_old = Fx.copy()
                
                for _ in range(15):
                    x_trial = x + alpha * s
                    if not np.all(np.isfinite(x_trial)):
                        alpha *= 0.5
                        continue
                    Fx_trial = np.asarray(F_counted(x_trial), dtype=float)
                    if not np.all(np.isfinite(Fx_trial)):
                        alpha *= 0.5
                        continue
                    n_trial = np.linalg.norm(Fx_trial)
                    
                    if n_trial < n:
                        x = x_trial
                        Fx = Fx_trial
                        n = n_trial
                        success = True
                        update_best(x)
                        break
                    alpha *= 0.5
                
                if not success:
                    return False
                    
                if n <= 1e-8:
                    return True
                    
                # Broyden Update
                s_actual = x - x_old
                y = Fx - Fx_old
                denom = np.dot(s_actual, s_actual)
                
                if denom > 1e-12:
                    J = J + np.outer(y - J @ s_actual, s_actual) / denom
                else:
                    return False
                    
            except:
                return False
                
        return False

    def nelder_mead(x_init, max_iter=100):
        """
        Nelder-Mead minimizer for ||F(x)||^2.
        """
        if not time_ok() or call_count >= MAX_CALLS - 50:
            return False
            
        # Initialize simplex
        x0 = np.array(x_init, dtype=float)
        f0 = np.linalg.norm(F_counted(x0))**2
        update_best(x0)
        if f0 <= 1e-16:
            return True
            
        delta = 0.1
        simplex = [
            x0,
            x0 + [delta, 0],
            x0 + [0, delta]
        ]
        fvals = [f0, np.inf, np.inf]
        
        for sx in simplex[1:]:
            if not time_ok() or call_count >= MAX_CALLS - 10:
                return False
            fs = np.linalg.norm(F_counted(sx))**2
            update_best(sx)
            if fs <= 1e-16:
                return True
            fvals[simplex.index(sx)] = fs

        for i in range(max_iter):
            if not time_ok() or call_count >= MAX_CALLS - 10:
                return False
                
            # Sort simplex by fvals
            idx = np.argsort(fvals)
            simplex = [simplex[i] for i in idx]
            fvals = [fvals[i] for i in idx]
            
            if fvals[0] <= 1e-16:
                return True
                
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            f_worst = fvals[-1]
            
            # Reflection
            x_r = centroid + (centroid - worst)
            if np.all(np.isfinite(x_r)):
                f_r = np.linalg.norm(F_counted(x_r))**2
                update_best(x_r)
                if f_r <= 1e-16:
                    return True
                
                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + 2 * (x_r - centroid)
                    if np.all(np.isfinite(x_e)):
                        f_e = np.linalg.norm(F_counted(x_e))**2
                        update_best(x_e)
                        if f_e <= 1e-16:
                            return True
                        if f_e < f_r:
                            simplex[-1] = x_e
                            fvals[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            fvals[-1] = f_r
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < f_worst:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + 0.5 * (worst - centroid)
                    if np.all(np.isfinite(x_c)):
                        f_c = np.linalg.norm(F_counted(x_c))**2
                        update_best(x_c)
                        if f_c <= 1e-16:
                            return True
                        if f_c < f_worst:
                            simplex[-1] = x_c
                            fvals[-1] = f_c
                        else:
                            # Shrink
                            for j in range(len(simplex) - 1):
                                simplex[j+1] = simplex[0] + 0.5 * (simplex[j+1] - simplex[0])
                                if not time_ok() or call_count >= MAX_CALLS - 10:
                                    return False
                                fvals[j+1] = np.linalg.norm(F_counted(simplex[j+1]))**2
                                update_best(simplex[j+1])
                    else:
                        # Shrink
                        for j in range(len(simplex) - 1):
                            simplex[j+1] = simplex[0] + 0.5 * (simplex[j+1] - simplex[0])
                            if not time_ok() or call_count >= MAX_CALLS - 10:
                                return False
                            fvals[j+1] = np.linalg.norm(F_counted(simplex[j+1]))**2
                            update_best(simplex[j+1])
                            
        return False

    # 1. Try Broyden from x0
    if newton_broyden(x0):
        if best_norm <= 1e-8:
            return best_x

    # 2. Try Nelder-Mead from x0 and restarts
    scale = max(1.0, np.linalg.norm(x0))
    restarts = [x0]
    # Diverse restarts
    for s in [0.5, 1.0, 2.0, 5.0]:
        for dx, dy in [(1,0), (0,1), (1,1), (-1,-1)]:
            restarts.append(x0 + s * np.array([dx, dy]) * scale)
            
    np.random.seed(42)
    np.random.shuffle(restarts)
    
    for x_init in restarts:
        if not time_ok() or call_count >= MAX_CALLS - 20:
            break
        if nelder_mead(x_init):
            if best_norm <= 1e-8:
                return best_x

    # 3. Grid Search + Polish
    if time_ok() and call_count < MAX_CALLS - 50:
        radius = max(10.0, 5 * scale)
        steps = 8
        dx = 2 * radius / steps
        
        best_grid_x = x0
        best_grid_norm = best_norm
        
        for i in range(steps + 1):
            for j in range(steps + 1):
                if not time_ok() or call_count >= MAX_CALLS - 10:
                    break
                x_g = x0 + np.array([i * dx - radius, j * dx - radius])
                if np.all(np.isfinite(x_g)):
                    try:
                        fg = np.asarray(F_counted(x_g), dtype=float)
                        if np.all(np.isfinite(fg)):
                            ng = np.linalg.norm(fg)
                            if ng < best_grid_norm:
                                best_grid_norm = ng
                                best_grid_x = x_g
                                update_best(x_g)
                                if ng <= 1e-8:
                                    return best_grid_x
                    except:
                        pass
            
        # Polish the best grid point with Broyden
        if best_grid_norm < best_norm * 0.5: # Only polish if we found something significantly better
            if newton_broyden(best_grid_x):
                if best_norm <= 1e-8:
                    return best_x

    return best_x if np.all(np.isfinite(best_x)) else x0
