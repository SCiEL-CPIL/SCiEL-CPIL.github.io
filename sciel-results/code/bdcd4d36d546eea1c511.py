"""Root finder for 3 fixed two-dimensional systems using Hybrid Broyden-Newton with Nelder-Mead fallback."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Initial phase: Broyden's method (low cost per iteration) to get close to root.
    2. Polish phase: Newton's method with finite-difference Jacobian (quadratic convergence) to reach high precision.
    3. Fallback: Nelder-Mead simplex if local methods stall.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def time_ok():
        return time.time() < deadline
    
    x0 = np.array(x0, dtype=float)
    call_count = 0
    MAX_CALLS = 480  # Stay well under 500
    
    def F_counted(x):
        nonlocal call_count
        call_count += 1
        return F(x)
    
    best_x = x0.copy()
    best_norm = np.inf
    
    def update_best(x, fx=None):
        """Update best_x and best_norm. If fx is None, it will be computed (costs a call)."""
        nonlocal best_x, best_norm
        try:
            x = np.asarray(x, dtype=float)
            if x.shape != (2,) or not np.all(np.isfinite(x)):
                return
            if fx is None:
                fx = F_counted(x)
            fx = np.asarray(fx, dtype=float)
            if fx.shape != (2,) or not np.all(np.isfinite(fx)):
                return
            n = np.linalg.norm(fx)
            if np.isfinite(n) and n < best_norm:
                best_norm = n
                best_x = x.copy()
        except:
            pass

    def broyden(x_init, max_iter=10):
        """
        Broyden's method for 2D systems.
        Returns (x, fx, success).
        """
        nonlocal call_count, best_norm, best_x
        if call_count >= MAX_CALLS or not time_ok():
            return x_init, None, False
        
        x = np.array(x_init, dtype=float)
        h = 1e-7
        
        # Initial Jacobian estimate: 3 calls
        Fx0 = np.asarray(F_counted(x), dtype=float)
        if not np.all(np.isfinite(Fx0)):
            return x, Fx0, False
        n0 = np.linalg.norm(Fx0)
        update_best(x, Fx0)
        if n0 <= 1e-8:
            return x, Fx0, True
        
        # Estimate Jacobian
        e1 = np.array([h, 0.0])
        e2 = np.array([0.0, h])
        Fx1 = np.asarray(F_counted(x + e1), dtype=float)
        Fx2 = np.asarray(F_counted(x + e2), dtype=float)
        
        if not (np.all(np.isfinite(Fx1)) and np.all(np.isfinite(Fx2))):
            return x, Fx0, False
        
        J = np.column_stack([(Fx1 - Fx0) / h, (Fx2 - Fx0) / h])
        
        # Check if Jacobian is singular
        detJ = np.linalg.det(J)
        if abs(detJ) < 1e-15:
            J += np.eye(2) * 1e-8
            
        for _ in range(max_iter):
            if call_count >= MAX_CALLS or not time_ok():
                break
            
            # Broyden step: solve J * dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx0)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(J, -Fx0, rcond=None)[0]
            
            # Damping
            step_norm = np.linalg.norm(dx)
            if step_norm > 5.0:
                dx = dx * (5.0 / step_norm)
            
            x_new = x + dx
            if not np.all(np.isfinite(x_new)):
                break
            
            Fx_new = np.asarray(F_counted(x_new), dtype=float)
            if not np.all(np.isfinite(Fx_new)):
                break
            
            n_new = np.linalg.norm(Fx_new)
            update_best(x_new, Fx_new)
            
            if n_new <= 1e-8:
                return x_new, Fx_new, True
            
            # Broyden update
            s = x_new - x
            y = Fx_new - Fx0
            denom = np.dot(s, s)
            if denom > 1e-30:
                J = J + np.outer(y - J @ s, s) / denom
            
            x = x_new
            Fx0 = Fx_new
            n0 = n_new
            
            # If we are getting close, we can stop Broyden and switch to Newton
            if n0 < 1e-4:
                break
                
        return x, Fx0, False

    def newton_polish(x_init, Fx_init, max_iter=10):
        """
        Newton's method with finite-difference Jacobian for high precision.
        Assumes Fx_init is F(x_init).
        """
        nonlocal call_count, best_norm, best_x
        if call_count >= MAX_CALLS or not time_ok():
            return x_init, Fx_init, False
            
        x = np.array(x_init, dtype=float)
        Fx = np.array(Fx_init, dtype=float)
        
        for _ in range(max_iter):
            if call_count >= MAX_CALLS or not time_ok():
                break
                
            n = np.linalg.norm(Fx)
            if n <= 1e-8:
                return x, Fx, True
                
            # Adaptive step size for finite differences
            # Smaller h when close to root to reduce truncation error, but not too small to avoid round-off
            h = 1e-6 * max(1.0, np.linalg.norm(x))
            if n < 1e-4:
                h = 1e-8 * max(1.0, np.linalg.norm(x))
            if n < 1e-6:
                h = 1e-10 * max(1.0, np.linalg.norm(x))
                
            e1 = np.array([h, 0.0])
            e2 = np.array([0.0, h])
            
            Fx1 = np.asarray(F_counted(x + e1), dtype=float)
            Fx2 = np.asarray(F_counted(x + e2), dtype=float)
            
            if not (np.all(np.isfinite(Fx1)) and np.all(np.isfinite(Fx2))):
                break
                
            J = np.column_stack([(Fx1 - Fx) / h, (Fx2 - Fx) / h])
            
            detJ = np.linalg.det(J)
            if abs(detJ) < 1e-14:
                J += np.eye(2) * 1e-10
                
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                
            # Line search
            alpha = 1.0
            success = False
            for _ in range(10):
                x_new = x + alpha * dx
                if not np.all(np.isfinite(x_new)):
                    alpha *= 0.5
                    continue
                    
                Fx_new = np.asarray(F_counted(x_new), dtype=float)
                if not np.all(np.isfinite(Fx_new)):
                    alpha *= 0.5
                    continue
                    
                n_new = np.linalg.norm(Fx_new)
                update_best(x_new, Fx_new)
                
                if n_new < n:
                    success = True
                    x = x_new
                    Fx = Fx_new
                    break
                else:
                    alpha *= 0.5
                    
            if not success:
                break
                
        n_final = np.linalg.norm(Fx)
        return x, Fx, n_final <= 1e-8

    def nelder_mead(x_init, max_iter=50):
        """
        Nelder-Mead simplex method for robust global convergence.
        """
        nonlocal call_count, best_norm, best_x
        if call_count >= MAX_CALLS or not time_ok():
            return x_init, None, False
            
        x0 = np.array(x_init, dtype=float)
        h = 0.1 * max(1.0, np.linalg.norm(x0))
        simplex = [x0.copy(), x0 + np.array([h, 0.0]), x0 + np.array([0.0, h])]
        
        fvals = []
        fxs = []
        for xi in simplex:
            if call_count >= MAX_CALLS or not time_ok():
                return x_init, None, False
            try:
                fx = np.asarray(F_counted(xi), dtype=float)
                if not np.all(np.isfinite(fx)):
                    return x_init, None, False
                fvals.append(np.linalg.norm(fx) ** 2)
                fxs.append(fx)
            except:
                return x_init, None, False
                
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        for _ in range(max_iter):
            if call_count >= MAX_CALLS or not time_ok():
                break
                
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            fxs = [fxs[i] for i in order]
            
            if fvals[0] <= 1e-16:
                update_best(simplex[0], fxs[0])
                return simplex[0], fxs[0], True
                
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = centroid + alpha * (centroid - simplex[-1])
            if not np.all(np.isfinite(xr)):
                break
            try:
                fxr = np.asarray(F_counted(xr), dtype=float)
                if not np.all(np.isfinite(fxr)):
                    break
                fr = np.linalg.norm(fxr) ** 2
            except:
                break
                
            if fr < fvals[0]:
                xe = centroid + gamma * (centroid - simplex[-1])
                if not np.all(np.isfinite(xe)):
                    simplex[-1] = xr
                    fvals[-1] = fr
                    fxs[-1] = fxr
                    continue
                try:
                    fxe = np.asarray(F_counted(xe), dtype=float)
                    if not np.all(np.isfinite(fxe)):
                        simplex[-1] = xr
                        fvals[-1] = fr
                        fxs[-1] = fxr
                        continue
                    fe = np.linalg.norm(fxe) ** 2
                except:
                    simplex[-1] = xr
                    fvals[-1] = fr
                    fxs[-1] = fxr
                    continue
                    
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                    fxs[-1] = fxe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
                    fxs[-1] = fxr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
                fxs[-1] = fxr
            else:
                if fr < fvals[-1]:
                    xc = centroid + rho * (xr - centroid)
                else:
                    xc = centroid - rho * (centroid - simplex[-1])
                    
                if not np.all(np.isfinite(xc)):
                    break
                try:
                    fxc = np.asarray(F_counted(xc), dtype=float)
                    if not np.all(np.isfinite(fxc)):
                        break
                    fc = np.linalg.norm(fxc) ** 2
                except:
                    break
                    
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                    fxs[-1] = fxc
                else:
                    for i in range(1, 3):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        if not np.all(np.isfinite(simplex[i])):
                            return simplex[0], fxs[0], fvals[0] <= 1e-16
                        try:
                            fxi = np.asarray(F_counted(simplex[i]), dtype=float)
                            if not np.all(np.isfinite(fxi)):
                                return simplex[0], fxs[0], fvals[0] <= 1e-16
                            fvals[i] = np.linalg.norm(fxi) ** 2
                            fxs[i] = fxi
                        except:
                            return simplex[0], fxs[0], fvals[0] <= 1e-16
                            
            min_idx = np.argmin(fvals)
            update_best(simplex[min_idx], fxs[min_idx])
            
        min_idx = np.argmin(fvals)
        return simplex[min_idx], fxs[min_idx], fvals[min_idx] <= 1e-16

    # 1. Try Broyden from x0
    x_broyden, Fx_broyden, success = broyden(x0, max_iter=10)
    
    if success:
        return best_x

    # 2. If Broyden got close, try Newton polish
    if Fx_broyden is not None and np.linalg.norm(Fx_broyden) < 1e-3:
        x_newton, Fx_newton, success_newton = newton_polish(x_broyden, Fx_broyden, max_iter=10)
        if success_newton:
            return best_x

    # 3. If still not good, try Nelder-Mead from best_x or x0
    if best_norm > 1e-8 and call_count < MAX_CALLS and time_ok():
        x_nm, Fx_nm, success_nm = nelder_mead(best_x if best_norm < np.inf else x0, max_iter=50)
        if success_nm:
            return best_x
            
        # If Nelder-Mead found a better point but not a root, try Broyden/Newton from there
        if Fx_nm is not None and np.linalg.norm(Fx_nm) < best_norm:
            x_broyden2, Fx_broyden2, success2 = broyden(x_nm, max_iter=10)
            if success2:
                return best_x
            if Fx_broyden2 is not None and np.linalg.norm(Fx_broyden2) < 1e-3:
                x_newton2, Fx_newton2, success_newton2 = newton_polish(x_broyden2, Fx_broyden2, max_iter=10)
                if success_newton2:
                    return best_x

    # 4. Final attempt: If we have budget, try a few random restarts with Broyden
    if best_norm > 1e-8 and call_count < MAX_CALLS - 50 and time_ok():
        scale = max(1.0, np.linalg.norm(x0))
        for i in range(5):
            if call_count >= MAX_CALLS or not time_ok():
                break
            # Random perturbation
            dx = np.random.normal(0, scale, 2)
            x_try = x0 + dx
            x_b, Fx_b, s = broyden(x_try, max_iter=5)
            if s:
                return best_x
            if Fx_b is not None and np.linalg.norm(Fx_b) < 1e-3:
                x_n, Fx_n, sn = newton_polish(x_b, Fx_b, max_iter=5)
                if sn:
                    return best_x

    return best_x if np.all(np.isfinite(best_x)) else x0

# EVOLVE-BLOCK-END
