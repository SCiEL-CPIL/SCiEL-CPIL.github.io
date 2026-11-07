"""Root finder for 3 fixed 2D systems using Newton's method with continuation and merit function."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Finds x in R^2 such that ||F(x)|| <= 1e-8 using Newton's method
    with periodic Jacobian recomputation and global convergence safeguards.
    
    Structural approach: Newton (quadratic convergence) with merit function
    for global convergence, and continuation from multiple starts.
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
    
    def compute_jacobian(x_pt, Fx_pt):
        """Compute Jacobian via forward differences. Returns J and the 2 extra F values."""
        h = 1e-7
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_e = F(x_pt + e)
            J[:, j] = (Fx_e - Fx_pt) / h
        return J
    
    # Track best solution
    best_x = x.copy()
    best_norm_sq = float('inf')
    
    # Initial evaluation
    try:
        Fx0 = F(x)
        best_norm_sq = float(np.dot(Fx0, Fx0))
    except:
        return np.zeros(2)
    
    if best_norm_sq <= tol_sq:
        return best_x
    
    # Newton solver from a given starting point
    def newton_solve(x_start, Fx_start, max_iters=30, jacobian_recompute_interval=2):
        """
        Newton's method with merit function and periodic Jacobian recompute.
        Returns (x, Fx, converged, iters_used).
        """
        x_cur = x_start.copy()
        Fx_cur = Fx_start.copy()
        norm_sq_cur = float(np.dot(Fx_cur, Fx_cur))
        
        if norm_sq_cur <= tol_sq:
            return x_cur, Fx_cur, True, 0
        
        # Initial Jacobian
        J = compute_jacobian(x_cur, Fx_cur)
        det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if abs(det) < 1e-12:
            return x_cur, Fx_cur, False, 0
        
        jacobian_age = 0
        iters = 0
        
        for it in range(max_iters):
            iters = it + 1
            if time_left() < 1.0:
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
            
            # Line search on merit function phi(x) = 0.5 * ||F(x)||^2
            # Use backtracking with Armijo condition
            alpha = 1.0
            phi_prev = 0.5 * norm_sq_cur
            slope = float(np.dot(Fx_cur, J @ s))  # directional derivative of phi
            # slope should be negative for descent; if not, use steepest descent direction
            if slope >= 0:
                # Use negative gradient of phi as direction
                # grad phi = J^T * Fx
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
            c1 = 1e-4  # Armijo parameter
            
            for _ls in range(30):
                x_try = x_cur + alpha * s
                Fx_try = F(x_try)
                norm_sq_try = float(np.dot(Fx_try, Fx_try))
                phi_try = 0.5 * norm_sq_try
                
                if norm_sq_try <= tol_sq:
                    return x_try, Fx_try, True, iters
                
                if phi_try <= phi_prev + c1 * alpha * slope:
                    accepted = True
                    break
                
                alpha *= 0.5
            
            if not accepted:
                break
            
            # Update
            x_cur = x_try
            Fx_cur = Fx_try
            norm_sq_cur = norm_sq_try
            jacobian_age += 1
            
            # Recompute Jacobian periodically
            if jacobian_age >= jacobian_recompute_interval:
                J = compute_jacobian(x_cur, Fx_cur)
                det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
                if abs(det) < 1e-12:
                    break
                jacobian_age = 0
        
        return x_cur, Fx_cur, False, iters
    
    # Multi-start strategy with continuation
    # Try different starting points with increasing radii
    max_restarts = 20
    
    # First, try from x0
    x_sol, Fx_sol, converged, _ = newton_solve(x, Fx0, max_iters=30)
    if converged:
        return x_sol
    
    if float(np.dot(Fx_sol, Fx_sol)) < best_norm_sq:
        best_norm_sq = float(np.dot(Fx_sol, Fx_sol))
        best_x = x_sol.copy()
    
    # If not converged, try from multiple starting points
    # Strategy: ring of points around best_x and x0
    np.random.seed(42)  # Deterministic for reproducibility
    
    for restart in range(max_restarts):
        if time_left() < 3.0:
            break
        if best_norm_sq <= tol_sq:
            return best_x
        
        # Generate starting point
        if restart % 3 == 0:
            # Random in box around x0
            x_start = x + np.random.uniform(-10, 10, size=2)
        elif restart % 3 == 1:
            # Ring around best_x with growing radius
            radius = 1.0 + 2.0 * (restart // 3)
            angle = np.random.uniform(0, 2 * np.pi)
            x_start = best_x + radius * np.array([np.cos(angle), np.sin(angle)])
        else:
            # Small perturbation around best_x
            x_start = best_x + np.random.randn(2) * 0.5
        
        if not np.all(np.isfinite(x_start)):
            continue
        
        try:
            Fx_start = F(x_start)
        except:
            continue
        
        norm_sq_start = float(np.dot(Fx_start, Fx_start))
        if norm_sq_start < best_norm_sq:
            best_norm_sq = norm_sq_start
            best_x = x_start.copy()
        
        if norm_sq_start <= tol_sq:
            return x_start
        
        x_sol, Fx_sol, converged, _ = newton_solve(x_start, Fx_start, max_iters=25)
        if converged:
            return x_sol
        
        norm_sq_sol = float(np.dot(Fx_sol, Fx_sol))
        if norm_sq_sol < best_norm_sq:
            best_norm_sq = norm_sq_sol
            best_x = x_sol.copy()
    
    return best_x
# EVOLVE-BLOCK-END
