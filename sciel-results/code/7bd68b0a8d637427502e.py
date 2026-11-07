# EVOLVE-BLOCK-START
import numpy as np
import time

# We use a strategy focused on minimizing call count:
# 1. Broyden's method for local convergence (1 F-call/iter vs 5 for finite diff Newton).
# 2. Homotopy continuation to find a starting point for Broyden if x0 is far from root.
# 3. Deterministic multi-start with Broyden.
# 4. Safety margin for 150s wall-clock limit.

_CALL_DEADLINE = 120.0 # Safety margin from 150s

def _safe_norm(F, x):
    """Safely compute ||F(x)||, returning inf on error."""
    try:
        Fx = F(x)
        if not np.all(np.isfinite(Fx)):
            return np.inf
        return np.linalg.norm(Fx)
    except Exception:
        return np.inf

def _broyden(F, x0, max_iter=50, tol=1e-12, time_limit=None):
    """
    Broyden's method (good step) for finding roots.
    Returns (x, norm_F, success).
    """
    x = np.array(x0, dtype=float)
    n = len(x)
    
    try:
        Fx = F(x)
        if not np.all(np.isfinite(Fx)):
            return x, np.inf, False
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx <= tol:
            return x, norm_Fx, True
        
        # Initial Jacobian approximation (Identity or finite difference if needed? 
        # Identity is often a good start for Broyden if scaled well, but let's do FD once for stability)
        h = 1e-8
        J = np.zeros((n, n))
        for j in range(n):
            e = np.zeros(n)
            e[j] = h
            Fx_plus = F(x + e)
            if not np.all(np.isfinite(Fx_plus)):
                return x, norm_Fx, False
            J[:, j] = (Fx_plus - Fx) / h
        
        for _ in range(max_iter):
            if time_limit and time.time() > time_limit:
                break
                
            try:
                # Solve J dx = -Fx
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                # If singular, add damping to J
                J += 1e-10 * np.eye(n)
                try:
                    dx = np.linalg.solve(J, -Fx)
                except np.linalg.LinAlgError:
                    return x, norm_Fx, False
            
            x_new = x + dx
            Fx_new = F(x_new)
            
            if not np.all(np.isfinite(Fx_new)):
                # Step was too large or went to invalid region
                # Try half step
                x_new = x + 0.5 * dx
                Fx_new = F(x_new)
                if not np.all(np.isfinite(Fx_new)):
                    return x, norm_Fx, False
            
            norm_Fx_new = np.linalg.norm(Fx_new)
            if norm_Fx_new <= tol:
                return x_new, norm_Fx_new, True
            
            # Update Jacobian (Broyden good step)
            s = x_new - x
            y = Fx_new - Fx
            J = J + np.outer(y - J @ s, s) / (s @ s)
            
            x = x_new
            Fx = Fx_new
            norm_Fx = norm_Fx_new
            
    except Exception:
        return x, norm_Fx, False
        
    return x, norm_Fx, norm_Fx <= tol

def _homotopy_start(F, x0, max_steps=20, t_step=0.2):
    """
    Try to find a better starting point by blending F with a trivial function.
    We blend F(x) with (x - x0) so that at t=0, the root is exactly x0.
    Then we increase t to 1.
    Returns (x_start, norm_F_start).
    """
    # Define blended function G_t(x) = (1-t)*(x - x0) + t*F(x)
    # At t=0, G_0(x) = x - x0, root is x0.
    # At t=1, G_1(x) = F(x), root is the target.
    
    # We need to solve G_t(x) = 0 for increasing t.
    # We use Broyden for each t step, starting from the previous solution.
    
    x_current = np.array(x0, dtype=float)
    
    for i in range(max_steps):
        t = (i + 1) * t_step
        if t > 1.0:
            t = 1.0
            
        # Define the blended function for this t
        def G(x):
            try:
                Fx = F(x)
                if not np.all(np.isfinite(Fx)):
                    return np.full(2, np.inf)
                return (1 - t) * (x - x0) + t * Fx
            except Exception:
                return np.full(2, np.inf)
        
        # Use Broyden to find root of G, starting from x_current
        x_new, norm_G, success = _broyden(G, x_current, max_iter=30, tol=1e-10)
        
        if success:
            x_current = x_new
        else:
            # If it fails, maybe step size is too big? 
            # For simplicity, we just return the current best guess
            # A more robust version would backtrack t
            pass
            
    # Check the norm of the original F at the final point
    norm_F_final = _safe_norm(F, x_current)
    return x_current, norm_F_final

def find_root_system(F, x0):
    """
    Finds a root x such that ||F(x)|| <= 1e-8.
    """
    start_time = time.time()
    x0 = np.array(x0, dtype=float)
    
    # 1. Check if x0 is already a root
    norm0 = _safe_norm(F, x0)
    if norm0 <= 1e-8:
        return x0
    
    best_x = x0
    best_norm = norm0
    
    # 2. Try Broyden directly from x0
    # This is cheap: ~5-10 calls if it converges.
    x_b, norm_b, success_b = _broyden(F, x0, max_iter=50, tol=1e-12, time_limit=start_time + 10)
    if success_b and norm_b <= 1e-8:
        return x_b
    if norm_b < best_norm:
        best_norm = norm_b
        best_x = x_b
    
    # 3. If direct Broyden failed or didn't converge to 1e-8, try Homotopy
    # Homotopy might take more calls, but it's robust to bad starts.
    if time.time() - start_time < _CALL_DEADLINE * 0.5:
        x_h, norm_h = _homotopy_start(F, x0, max_steps=10, t_step=0.1)
        if norm_h < best_norm:
            best_norm = norm_h
            best_x = x_h
        if norm_h <= 1e-8:
            return best_x
            
        # Polish the homotopy result with Broyden on the original F
        if norm_h < 1.0: # Only polish if we got somewhat close
            x_pol, norm_pol, success_pol = _broyden(F, best_x, max_iter=50, tol=1e-12, time_limit=start_time + 20)
            if success_pol and norm_pol <= 1e-8:
                return x_pol
            if norm_pol < best_norm:
                best_norm = norm_pol
                best_x = x_pol

    # 4. Multi-start Broyden from a few deterministic points
    # If homotopy failed, maybe the basin is far. Try a few random-ish points.
    # But we must be careful with call count.
    if time.time() - start_time < _CALL_DEADLINE * 0.7:
        # Generate a few points in a small circle around x0
        radii = [0.5, 1.0, 2.0]
        angles = [0, np.pi/2, np.pi, 3*np.pi/2]
        
        for r in radii:
            if time.time() - start_time > _CALL_DEADLINE * 0.8:
                break
            for a in angles:
                x_try = x0 + r * np.array([np.cos(a), np.sin(a)])
                # Quick check
                norm_try = _safe_norm(F, x_try)
                if norm_try < best_norm:
                    best_norm = norm_try
                    best_x = x_try
                if norm_try <= 1e-8:
                    return x_try
                    
                # If it's significantly better than x0, try Broyden
                if norm_try < 0.1 * norm0 or norm_try < 1e-4:
                    x_b2, norm_b2, success_b2 = _broyden(F, x_try, max_iter=30, tol=1e-12, time_limit=start_time + 5)
                    if success_b2 and norm_b2 <= 1e-8:
                        return x_b2
                    if norm_b2 < best_norm:
                        best_norm = norm_b2
                        best_x = x_b2

    # 5. Final polish with Broyden on the best candidate
    if time.time() - start_time < _CALL_DEADLINE * 0.9:
        x_final, norm_final, success_final = _broyden(F, best_x, max_iter=50, tol=1e-12, time_limit=start_time + 10)
        if success_final and norm_final <= 1e-8:
            return x_final
        if norm_final < best_norm:
            best_norm = norm_final
            best_x = x_final

    return best_x
# EVOLVE-BLOCK-END
