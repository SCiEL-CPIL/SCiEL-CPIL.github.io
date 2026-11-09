"""Root finder for 3 fixed two-dimensional systems using Broyden's method with restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _safe_F(F, x):
    """Call F safely, returning a finite array of length 2 or None on failure."""
    try:
        y = F(np.asarray(x, dtype=float))
        y = np.asarray(y, dtype=float)
        if y.shape != (2,) or not np.all(np.isfinite(y)):
            return None
        return y
    except Exception:
        return None


def _broyden_update(J, s, y):
    """Rank-1 Broyden update: J += (y - J*s) @ s^T / (s^T s)"""
    denom = np.dot(s, s)
    if denom > 1e-30:
        J = J + np.outer(y - J @ s, s) / denom
    return J


def _try_broyden(F, x0, max_iter=50, tol=1e-10, deadline=None):
    """Broyden's method with line search. Returns (x, Fx, n_calls, success)."""
    x = np.array(x0, dtype=float)
    n_calls = 0
    t_start = time.time()
    
    def _check_deadline():
        if deadline is not None and time.time() - t_start > deadline:
            return True
        return False
    
    # Initial F evaluation
    Fx = _safe_F(F, x)
    n_calls += 1
    if Fx is None:
        return x, None, n_calls, False
    if np.linalg.norm(Fx) <= tol:
        return x, Fx, n_calls, True
    
    # Initial Jacobian via forward differences
    h = 1e-7
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_pert = _safe_F(F, x + e)
        n_calls += 1
        if Fx_pert is None:
            return x, Fx, n_calls, False
        J[:, j] = (Fx_pert - Fx) / h
    
    # Broyden iterations
    for it in range(max_iter):
        if _check_deadline():
            break
        
        # Solve J * dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Fallback: use pseudo-inverse
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except Exception:
                break
        
        # Line search: find alpha such that ||F(x + alpha*dx)|| < ||Fx||
        Fx_norm = np.linalg.norm(Fx)
        alpha = 1.0
        x_new = x + alpha * dx
        Fx_new = _safe_F(F, x_new)
        n_calls += 1
        if Fx_new is None:
            break
        
        # Backtracking line search
        for _ in range(20):
            Fx_new_norm = np.linalg.norm(Fx_new)
            if Fx_new_norm < Fx_norm:
                break
            alpha *= 0.5
            x_new = x + alpha * dx
            Fx_new = _safe_F(F, x_new)
            n_calls += 1
            if Fx_new is None:
                break
            if _check_deadline():
                return x, Fx, n_calls, False
        
        if Fx_new is None:
            break
        
        Fx_new_norm = np.linalg.norm(Fx_new)
        if Fx_new_norm <= tol:
            x = x_new
            Fx = Fx_new
            return x, Fx, n_calls, True
        
        # Broyden update
        s = x_new - x
        y = Fx_new - Fx
        J = _broyden_update(J, s, y)
        x = x_new
        Fx = Fx_new
        
        # Stagnation detection
        if Fx_new_norm >= Fx_norm * (1 - 1e-12):
            # Try a different direction or break
            break
    
    return x, Fx, n_calls, False


def find_root_system(F, x0):
    """Find root of F: R^2 -> R^2 using Broyden's method with restarts."""
    x0 = np.array(x0, dtype=float)
    t_start = time.time()
    deadline = 110.0  # safety margin before 120s limit
    
    best_x = x0.copy()
    best_Fx = None
    best_norm = np.inf
    best_calls = 0
    
    # Generate candidate starting points
    candidates = [x0]
    # Add some perturbations
    for scale in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        candidates.append(x0 + np.array([scale, 0.0]))
        candidates.append(x0 + np.array([0.0, scale]))
        candidates.append(x0 + np.array([scale, scale]))
        candidates.append(x0 + np.array([-scale, scale]))
        candidates.append(x0 + np.array([scale, -scale]))
    # Origin and axis points
    candidates.append(np.array([0.0, 0.0]))
    candidates.append(np.array([1.0, 0.0]))
    candidates.append(np.array([0.0, 1.0]))
    candidates.append(np.array([-1.0, 0.0]))
    candidates.append(np.array([0.0, -1.0]))
    
    # Try each candidate
    for cand in candidates:
        if time.time() - t_start > deadline:
            break
        
        x_res, Fx_res, n_calls, success = _try_broyden(F, cand, max_iter=40, tol=1e-10, deadline=deadline - (time.time() - t_start))
        best_calls = max(best_calls, n_calls)
        
        if Fx_res is not None:
            norm = np.linalg.norm(Fx_res)
            if norm < best_norm:
                best_norm = norm
                best_x = x_res.copy()
                best_Fx = Fx_res.copy()
            
            if success and norm <= 1e-8:
                # Try to polish further
                if time.time() - t_start < deadline - 5:
                    x_pol, Fx_pol, n_pol, _ = _try_broyden(F, x_res, max_iter=20, tol=1e-12, deadline=deadline - (time.time() - t_start))
                    if Fx_pol is not None:
                        norm_pol = np.linalg.norm(Fx_pol)
                        if norm_pol < norm:
                            best_x = x_pol.copy()
                            best_Fx = Fx_pol.copy()
                            best_norm = norm_pol
                return best_x
    
    # Return best result found
    return best_x
# EVOLVE-BLOCK-END
