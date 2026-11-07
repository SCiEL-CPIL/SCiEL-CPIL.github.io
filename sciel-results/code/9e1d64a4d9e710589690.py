# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Strategy:
# 1. Log-spaced radial sweep to find a good starting point efficiently.
# 2. Broyden's method for fast local convergence (much fewer function calls than FD Newton).
# 3. Nelder-Mead as a fallback global optimizer if Broyden fails.
# 4. Strict deadline guard to ensure we return *something* before timeout.

def _safe_norm(F, x):
    try:
        Fx = F(x)
        if not np.all(np.isfinite(Fx)):
            return np.inf
        return np.linalg.norm(Fx)
    except Exception:
        return np.inf

def _broyden(F, x0, max_iter=50, tol=1e-12):
    """
    Broyden's method (good direction) for solving F(x) = 0.
    Uses 1 function evaluation per iteration after the first Jacobian estimate.
    """
    x = np.array(x0, dtype=float)
    n = len(x)
    
    # Initial Jacobian estimate using forward differences
    # This costs n evaluations.
    J = np.zeros((n, n))
    h = 1e-8
    try:
        Fx = F(x)
        if not np.all(np.isfinite(Fx)):
            return x, np.inf, False
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx <= tol:
            return x, norm_Fx, True
            
        for j in range(n):
            e = np.zeros(n)
            e[j] = h
            Fx_plus = F(x + e)
            if not np.all(np.isfinite(Fx_plus)):
                return x, np.inf, False
            J[:, j] = (Fx_plus - Fx) / h
    except Exception:
        return x, np.inf, False

    for _ in range(max_iter):
        try:
            # Check if we are close enough
            norm_Fx = np.linalg.norm(Fx)
            if norm_Fx <= tol:
                return x, norm_Fx, True
            
            # Solve J * dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                # If singular, use pseudo-inverse or break
                try:
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except:
                    return x, norm_Fx, False
                    
            x_new = x + dx
            
            # Check for non-finite
            if not np.all(np.isfinite(x_new)):
                return x, norm_Fx, False
            
            # Evaluate F at new point
            Fx_new = F(x_new)
            if not np.all(np.isfinite(Fx_new)):
                return x, norm_Fx, False
            
            # Update Jacobian (Broyden's good rank-1 update)
            # J_new = J + ( (Fx_new - Fx) + J*dx ) * dx^T / (dx^T * dx)
            # Actually, standard Broyden good is:
            # J_new = J + ( (y - J*dx) * dx^T ) / (dx^T * dx)
            # where y = F(x_new) - F(x)
            
            y = Fx_new - Fx
            Jdx = J @ dx
            delta = y - Jdx
            
            # Check for division by zero
            dx_norm_sq = np.dot(dx, dx)
            if dx_norm_sq < 1e-30:
                break
                
            J = J + np.outer(delta, dx) / dx_norm_sq
            
            x = x_new
            Fx = Fx_new
            
        except Exception:
            return x, norm_Fx if 'norm_Fx' in locals() else np.inf, False

    return x, np.linalg.norm(Fx) if np.all(np.isfinite(Fx)) else np.inf, False

def _nelder_mead_global(F, x0, max_iter=100):
    """
    Nelder-Mead on ||F(x)||^2.
    """
    def objective(x):
        try:
            fx = F(x)
            if not np.all(np.isfinite(fx)):
                return 1e100
            return np.dot(fx, fx)
        except Exception:
            return 1e100

    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'xatol': 1e-8, 'fatol': 1e-16, 'maxiter': max_iter, 'maxfev': 200})
        x_opt = res.x
        norm_opt = _safe_norm(F, x_opt)
        return x_opt, norm_opt
    except Exception:
        return x0, _safe_norm(F, x0)

def find_root_system(F, x0):
    """
    Finds a root x such that ||F(x)|| <= 1e-8.
    """
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin for 120s limit
    x0 = np.array(x0, dtype=float)
    
    # 1. Check x0
    norm0 = _safe_norm(F, x0)
    if norm0 <= 1e-8:
        return x0
    
    best_x = x0
    best_norm = norm0
    
    # 2. Determine scale
    # If F(x0) is very small, the root is likely close.
    # If F(x0) is large, the root might be far.
    # We use a log-spaced radial sweep.
    
    # Estimate a characteristic length scale.
    # If ||F(x0)|| is small, start with small radii.
    # If ||F(x0)|| is large, we might need larger radii.
    # However, without knowing the Jacobian scale, we just try a broad range.
    
    # Radii to try: log-spaced from 0.1 to 100
    radii = np.logspace(-2, 2, 25)
    
    # For each radius, try a few angles.
    # To save calls, we first evaluate F at a few points to see if any are good.
    # Then we run Broyden from the best ones.
    
    candidates = []
    
    # Add x0
    candidates.append((x0, norm0))
    
    # Generate candidates on circles
    # We limit the number of initial evaluations to keep it fast.
    # Let's say we evaluate at most 50 points in the screening phase.
    max_screening_calls = 50
    calls_made = 1 # for x0
    
    for r in radii:
        if time.time() - start_time > DEADLINE * 0.3:
            break
        # 4 angles per radius to keep it cheap
        angles = [0, np.pi/2, np.pi, 3*np.pi/2]
        for theta in angles:
            if calls_made >= max_screening_calls:
                break
            x_try = x0 + r * np.array([np.cos(theta), np.sin(theta)])
            norm_try = _safe_norm(F, x_try)
            calls_made += 1
            candidates.append((x_try, norm_try))
            if norm_try < best_norm:
                best_norm = norm_try
                best_x = x_try
            if norm_try <= 1e-8:
                return x_try
        
        if calls_made >= max_screening_calls:
            break
            
    # Sort candidates by norm
    candidates.sort(key=lambda t: t[1])
    
    # 3. Run Broyden from the top candidates
    # Broyden is cheap (1 call/iter), so we can afford to try several starts.
    top_k = 10
    for x_try, norm_try in candidates[:top_k]:
        if time.time() - start_time > DEADLINE * 0.8:
            break
        
        x_new, norm_new, success = _broyden(F, x_try, max_iter=30, tol=1e-12)
        
        if success and norm_new <= 1e-8:
            return x_new
            
        if norm_new < best_norm:
            best_norm = norm_new
            best_x = x_new
            
    # 4. If Broyden didn't find a root, try Nelder-Mead from the best point
    if time.time() - start_time < DEADLINE * 0.9:
        x_nm, norm_nm = _nelder_mead_global(F, best_x, max_iter=100)
        if norm_nm < best_norm:
            best_norm = norm_nm
            best_x = x_nm
        if norm_nm <= 1e-8:
            return best_x
        
        # Try Broyden polish from Nelder-Mead result
        if time.time() - start_time < DEADLINE * 0.95:
            x_new, norm_new, success = _broyden(F, best_x, max_iter=20, tol=1e-12)
            if norm_new < best_norm:
                best_norm = norm_new
                best_x = x_new
            if norm_new <= 1e-8:
                return best_x
                
    return best_x
# EVOLVE-BLOCK-END
