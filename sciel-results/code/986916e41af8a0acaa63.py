# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline management
_CALL_DEADLINE = 20.0

def _safe_norm(F, x):
    """Safely compute ||F(x)||, returning inf on error."""
    try:
        Fx = F(x)
        if not np.all(np.isfinite(Fx)):
            return np.inf
        return np.linalg.norm(Fx)
    except Exception:
        return np.inf

def _broyden_lm(F, x0, max_iter=100, tol=1e-12, deadline=None):
    """
    Levenberg-Marquardt with Broyden Jacobian updates.
    This is a structural change from pure Newton (which requires 5 F calls/iter)
    to LM-Broyden (which requires 1 F call/iter after init).
    
    LM provides robustness via damping (mu), while Broyden provides the Jacobian 
    approximation with minimal cost.
    """
    x = np.array(x0, dtype=float)
    n = 2
    
    # Initial F evaluation
    try:
        Fx = F(x)
        if not np.all(np.isfinite(Fx)):
            return x, np.inf, False
        norm_Fx = np.linalg.norm(Fx)
    except Exception:
        return x, np.inf, False
    
    if norm_Fx <= tol:
        return x, norm_Fx, True

    # Initial Jacobian via finite differences (5 calls)
    h = 1e-6
    J = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = h
        try:
            Fx_plus = F(x + e)
            if not np.all(np.isfinite(Fx_plus)):
                return x, norm_Fx, False
            J[:, j] = (Fx_plus - Fx) / h
        except Exception:
            return x, norm_Fx, False
            
    Fx_old = Fx
    x_old = x.copy()
    
    for _ in range(max_iter):
        if deadline and time.time() > deadline:
            return x, norm_Fx, norm_Fx <= tol
            
        # Solve (J^T J + mu * I) dx = -J^T F
        # We use a simple adaptive mu strategy
        mu = 1e-3
        success_step = False
        
        while True:
            try:
                JtJ = J.T @ J
                # Add damping
                A = JtJ + mu * np.eye(n)
                b = -J.T @ Fx
                dx = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                mu *= 10
                if mu > 1e10:
                    break
                continue
            
            x_new = x + dx
            if not np.all(np.isfinite(x_new)):
                mu *= 10
                if mu > 1e10:
                    break
                continue
            
            try:
                Fx_new = F(x_new)
                if not np.all(np.isfinite(Fx_new)):
                    mu *= 10
                    if mu > 1e10:
                        break
                    continue
                norm_new = np.linalg.norm(Fx_new)
            except Exception:
                mu *= 10
                if mu > 1e10:
                    break
                continue
            
            if norm_new < norm_Fx:
                # Accept step
                # Broyden update: J_new = J + (s_new - s_old) * (s_new - J*s_old)^T / ||s_new - J*s_old||^2
                s_new = x_new - x
                s_old = J @ s_new
                diff = s_new - s_old
                if np.linalg.norm(diff) > 1e-12:
                    J = J + np.outer(s_new - s_old, (s_new - s_old).T) / (np.dot(s_new - s_old, s_new - s_old))
                
                x = x_new
                Fx = Fx_new
                norm_Fx = norm_new
                mu = max(mu * 0.1, 1e-10) # Reduce damping on success
                success_step = True
                
                if norm_Fx <= tol:
                    return x, norm_Fx, True
                break
            else:
                # Reject step, increase damping
                mu *= 10
                if mu > 1e10:
                    break
        
        if not success_step:
            # If we couldn't take a step, try a small random perturbation or break
            # For now, break to avoid infinite loops
            break
            
    return x, norm_Fx, norm_Fx <= tol

def _grid_search(F, x0, deadline):
    """
    Structured search in 2D space.
    Uses a spiral/radial pattern to cover local and somewhat global basins.
    Returns the best point found and its norm.
    """
    best_x = x0.copy()
    best_norm = _safe_norm(F, x0)
    
    if best_norm <= 1e-8:
        return best_x, best_norm

    # Define search radii and angles
    # We prioritize local convergence first, then expand
    radii = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    
    # Interleave radii and angles to balance exploration/exploitation
    # But strictly ordered by radius for early termination if close
    for r in radii:
        if time.time() > deadline:
            break
        for theta in angles:
            if time.time() > deadline:
                break
            x_try = x0 + r * np.array([np.cos(theta), np.sin(theta)])
            norm_try = _safe_norm(F, x_try)
            
            if norm_try < best_norm:
                best_norm = norm_try
                best_x = x_try.copy()
                
                if best_norm <= 1e-8:
                    return best_x, best_norm
                    
    return best_x, best_norm

def find_root_system(F, x0):
    """
    Finds a root x such that ||F(x)|| <= 1e-8.
    Strategy:
    1. Quick check at x0.
    2. Grid search to find a good starting point (basin identification).
    3. Levenberg-Marquardt with Broyden updates for fast, robust local convergence.
    4. If LM fails, try a few other grid points with LM.
    """
    start_time = time.time()
    x0 = np.array(x0, dtype=float)
    
    # 1. Check x0
    norm0 = _safe_norm(F, x0)
    if norm0 <= 1e-8:
        return x0

    # 2. Grid search to find the best candidate
    # We allocate 50% of the time budget to finding a good start
    search_deadline = start_time + _CALL_DEADLINE * 0.5
    best_start_x, best_start_norm = _grid_search(F, x0, search_deadline)
    
    # 3. Run LM-Broyden from the best start
    # Allocate remaining time (with margin) for the solver
    solve_deadline = start_time + _CALL_DEADLINE * 0.9
    x_sol, norm_sol, success = _broyden_lm(F, best_start_x, max_iter=50, tol=1e-12, deadline=solve_deadline)
    
    if success and norm_sol <= 1e-8:
        return x_sol
    
    # 4. If the best start failed, try the next best few points from the grid
    # We re-run a smaller grid search to get top 3 candidates, or just retry from neighbors
    # To keep it simple and fast, we try a few specific points around the best_start
    # that are slightly different
    
    # Let's just try a few random-ish perturbations of the best_start if time permits
    if time.time() < solve_deadline:
        perturbations = [
            best_start_x + np.array([0.1, 0.0]),
            best_start_x + np.array([0.0, 0.1]),
            best_start_x + np.array([-0.1, 0.0]),
            best_start_x + np.array([0.0, -0.1]),
            best_start_x + np.array([0.5, 0.5]),
            best_start_x + np.array([-0.5, -0.5]),
        ]
        
        for x_try in perturbations:
            if time.time() > solve_deadline:
                break
            # Quick norm check
            n_try = _safe_norm(F, x_try)
            if n_try < best_start_norm:
                best_start_x = x_try
                best_start_norm = n_try
            
            x_sol2, norm_sol2, success2 = _broyden_lm(F, x_try, max_iter=30, tol=1e-12, deadline=solve_deadline)
            if success2 and norm_sol2 <= 1e-8:
                return x_sol2
            if norm_sol2 < norm_sol:
                x_sol = x_sol2
                norm_sol = norm_sol2

    # 5. Return best found
    return x_sol
# EVOLVE-BLOCK-END
