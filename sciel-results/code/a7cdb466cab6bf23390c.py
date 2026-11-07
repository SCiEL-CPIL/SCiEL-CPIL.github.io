"""Root finder for 3 fixed 2D systems using Broyden's method with Newton polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Finds x in R^2 such that ||F(x)|| <= 1e-8 using Broyden's method
    with merit function line search, Newton polish, and multi-start.
    
    Broyden uses ~3 calls for initial Jacobian, then 1 call per step.
    Newton polish with fresh Jacobians ensures high-precision convergence.
    """
    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)
    
    tol = 1e-8
    tol_sq = tol * tol
    start_time = time.time()
    time_limit = 145.0
    
    def time_left():
        return time_limit - (time.time() - start_time)
    
    def F_eval(v):
        """Safely evaluate F and return (value, norm_sq) or (None, inf)."""
        try:
            fv = F(v)
            fv = np.asarray(fv, dtype=float)
            if fv.shape != (2,) or not np.all(np.isfinite(fv)):
                return None, np.inf
            return fv, float(np.dot(fv, fv))
        except:
            return None, np.inf
    
    best_x = x.copy()
    best_norm_sq = np.inf
    
    # Initial evaluation
    F_x, best_norm_sq = F_eval(x)
    if F_x is None:
        return np.zeros(2)
    
    if best_norm_sq <= tol_sq:
        return best_x
    
    def compute_jacobian(x_pt, Fx_pt):
        """Compute Jacobian via forward differences. 2 extra F-calls."""
        h = 1e-7
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_e, _ = F_eval(x_pt + e)
            if Fx_e is None:
                return None
            J[:, j] = (Fx_e - Fx_pt) / h
        return J
    
    def newton_polish(x_start, Fx_start, max_iters=10):
        """
        Newton's method with fresh Jacobians for high-precision convergence.
        Returns (x, Fx, norm_sq).
        """
        x_cur = x_start.copy()
        Fx_cur = Fx_start.copy()
        norm_sq_cur = float(np.dot(Fx_cur, Fx_cur))
        
        if norm_sq_cur <= tol_sq:
            return x_cur, Fx_cur, norm_sq_cur
        
        for it in range(max_iters):
            if time_left() < 2.0:
                break
            
            J = compute_jacobian(x_cur, Fx_cur)
            if J is None:
                break
            
            det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            if abs(det) < 1e-12:
                break
            
            try:
                s = np.linalg.solve(J, -Fx_cur)
            except np.linalg.LinAlgError:
                break
            
            s_norm = np.linalg.norm(s)
            if s_norm < 1e-15:
                break
            
            # Simple line search
            alpha = 1.0
            accepted = False
            for _ls in range(20):
                x_try = x_cur + alpha * s
                Fx_try, n_sq_try = F_eval(x_try)
                if Fx_try is None:
                    alpha *= 0.5
                    continue
                
                if n_sq_try <= tol_sq:
                    return x_try, Fx_try, n_sq_try
                
                if n_sq_try < norm_sq_cur:
                    x_cur = x_try
                    Fx_cur = Fx_try
                    norm_sq_cur = n_sq_try
                    accepted = True
                    break
                
                alpha *= 0.5
            
            if not accepted:
                break
        
        return x_cur, Fx_cur, norm_sq_cur
    
    def broyden_solve(x_start, Fx_start, max_iters=40):
        """
        Broyden's method (good-given) with merit function line search.
        Uses ~3 calls for initial Jacobian, then 1 call per iteration.
        Returns (x, Fx, norm_sq).
        """
        x_cur = x_start.copy()
        Fx_cur = Fx_start.copy()
        norm_sq_cur = float(np.dot(Fx_cur, Fx_cur))
        
        if norm_sq_cur <= tol_sq:
            return x_cur, Fx_cur, norm_sq_cur
        
        # Initial Jacobian approximation
        J = compute_jacobian(x_cur, Fx_cur)
        if J is None:
            return x_cur, Fx_cur, norm_sq_cur
        
        det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if abs(det) < 1e-12:
            # Try a perturbed start
            return x_cur, Fx_cur, norm_sq_cur
        
        for it in range(max_iters):
            if time_left() < 2.0:
                break
            
            # Solve J * s = -Fx
            try:
                s = np.linalg.solve(J, -Fx_cur)
            except np.linalg.LinAlgError:
                break
            
            s_norm = np.linalg.norm(s)
            if s_norm < 1e-15:
                break
            
            # Limit step size
            max_step = 50.0
            if s_norm > max_step:
                s = s * (max_step / s_norm)
                s_norm = max_step
            
            # Merit function line search: phi = 0.5 * ||F||^2
            alpha = 1.0
            phi_prev = 0.5 * norm_sq_cur
            slope = float(np.dot(Fx_cur, J @ s))
            
            # If slope is not negative, use steepest descent
            if slope >= 0:
                s = -J.T @ Fx_cur
                s_norm = np.linalg.norm(s)
                if s_norm < 1e-15:
                    break
                if s_norm > max_step:
                    s = s * (max_step / s_norm)
                slope = -float(np.dot(s, J.T @ Fx_cur))
                if slope >= 0:
                    break
            
            accepted = False
            c1 = 1e-4
            
            for _ls in range(25):
                x_try = x_cur + alpha * s
                Fx_try, n_sq_try = F_eval(x_try)
                if Fx_try is None:
                    alpha *= 0.5
                    continue
                
                if n_sq_try <= tol_sq:
                    return x_try, Fx_try, n_sq_try
                
                phi_try = 0.5 * n_sq_try
                if phi_try <= phi_prev + c1 * alpha * slope:
                    # Broyden update: J_new = J + (y - J*s) * s^T / (s^T*s)
                    dx = x_try - x_cur
                    dF = Fx_try - Fx_cur
                    s_dot_s = float(np.dot(dx, dx))
                    if s_dot_s > 1e-16:
                        J = J + np.outer(dF - J @ dx, dx) / s_dot_s
                    
                    x_cur = x_try
                    Fx_cur = Fx_try
                    norm_sq_cur = n_sq_try
                    accepted = True
                    break
                
                alpha *= 0.5
            
            if not accepted:
                break
        
        return x_cur, Fx_cur, norm_sq_cur
    
    # Multi-start strategy
    num_restarts = 25
    np.random.seed(42)
    
    # First try from x0
    x_res, F_res, n_res = broyden_solve(x, F_x, max_iters=40)
    
    if n_res < best_norm_sq:
        best_norm_sq = n_res
        best_x = x_res.copy()
    
    if n_res <= tol_sq:
        # Newton polish for extra precision
        x_pol, F_pol, n_pol = newton_polish(x_res, F_res, max_iters=8)
        if n_pol <= tol_sq:
            return x_pol
        if n_pol < best_norm_sq:
            best_norm_sq = n_pol
            best_x = x_pol.copy()
        return best_x
    
    # If not converged, try Newton polish on the result
    if n_res < 1e-6:
        x_pol, F_pol, n_pol = newton_polish(x_res, F_res, max_iters=10)
        if n_pol <= tol_sq:
            return x_pol
        if n_pol < best_norm_sq:
            best_norm_sq = n_pol
            best_x = x_pol.copy()
    
    # Multi-start from various points
    for restart in range(num_restarts):
        if time_left() < 3.0:
            break
        
        if restart == 0:
            x_start = x.copy()
        elif restart < 10:
            # Ring of points around best_x
            angle = 2 * np.pi * restart / 10
            radius = 0.5 + 0.3 * restart
            x_start = best_x + radius * np.array([np.cos(angle), np.sin(angle)])
        elif restart < 18:
            # Random points in medium radius
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(1.0, 5.0)
            x_start = best_x + radius * np.array([np.cos(angle), np.sin(angle)])
        else:
            # Broad random search
            x_start = np.random.uniform(-10, 10, size=2)
        
        if not np.all(np.isfinite(x_start)):
            continue
        
        F_start, n_start = F_eval(x_start)
        if F_start is None:
            continue
        
        if n_start < best_norm_sq:
            best_norm_sq = n_start
            best_x = x_start.copy()
        
        if n_start <= tol_sq:
            return best_x
        
        # Run Broyden
        x_res, F_res, n_res = broyden_solve(x_start, F_start, max_iters=30)
        
        if n_res < best_norm_sq:
            best_norm_sq = n_res
            best_x = x_res.copy()
        
        if n_res <= tol_sq:
            x_pol, F_pol, n_pol = newton_polish(x_res, F_res, max_iters=8)
            if n_pol <= tol_sq:
                return x_pol
            if n_pol < best_norm_sq:
                best_norm_sq = n_pol
                best_x = x_pol.copy()
            return best_x
        
        # If close, try Newton polish
        if n_res < 1e-6:
            x_pol, F_pol, n_pol = newton_polish(x_res, F_res, max_iters=10)
            if n_pol <= tol_sq:
                return x_pol
            if n_pol < best_norm_sq:
                best_norm_sq = n_pol
                best_x = x_pol.copy()
    
    # Final polish attempt
    if best_norm_sq < 1e-4 and time_left() > 5.0:
        F_best, _ = F_eval(best_x)
        if F_best is not None:
            x_pol, F_pol, n_pol = newton_polish(best_x, F_best, max_iters=15)
            if n_pol <= tol_sq:
                return x_pol
            if n_pol < best_norm_sq:
                best_norm_sq = n_pol
                best_x = x_pol.copy()
    
    return best_x
# EVOLVE-BLOCK-END
