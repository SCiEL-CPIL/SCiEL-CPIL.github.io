"""Root finder for 3 fixed two-dimensional systems using Powell's method as primary strategy."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _is_finite(x):
    return np.all(np.isfinite(x)) and x.shape == (2,)

def _norm(x):
    return float(np.linalg.norm(x))

def _safe_call(F, x):
    """Call F safely, returning None on failure."""
    try:
        y = F(np.asarray(x, dtype=float))
        y = np.asarray(y, dtype=float)
        if y.shape != (2,):
            return None
        if not np.all(np.isfinite(y)):
            return None
        return y
    except Exception:
        return None

def _powell_method(F, x0, max_iter=100):
    """
    Powell's method adapted for root-finding by minimizing ||F(x)||^2.
    Uses conjugate direction search with periodic axis reset.
    """
    x = np.array(x0, dtype=float, copy=True)
    Fx = _safe_call(F, x)
    if Fx is None:
        return x
    obj = _norm(Fx) ** 2
    if obj <= 1e-16:  # equivalent to ||F|| <= 1e-8
        return x
    
    # Initial directions: standard basis
    directions = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    
    for _iter in range(max_iter):
        if _elapsed() > 115:
            break
        
        old_x = x.copy()
        old_obj = obj
        max_reduction = 0.0
        max_dir_idx = 0
        
        for i, d in enumerate(directions):
            # Line search along direction d
            best_t = 0.0
            best_val = obj
            
            # Try a few candidate steps
            t_candidates = [0.01, 0.05, 0.1, 0.5, 1.0, -0.01, -0.05, -0.1, -0.5, -1.0]
            for t in t_candidates:
                trial = x + t * d
                Ft = _safe_call(F, trial)
                if Ft is None:
                    continue
                val = _norm(Ft) ** 2
                if val < best_val:
                    best_val = val
                    best_t = t
            
            if best_t != 0.0:
                x = x + best_t * directions[i]
                Fx = _safe_call(F, x)
                if Fx is None:
                    x = old_x
                    Fx = _safe_call(F, x)
                    obj = _norm(Fx) ** 2
                    break
                obj = _norm(Fx) ** 2
                if obj <= 1e-16:
                    return x
                reduction = old_obj - obj
                if reduction > max_reduction:
                    max_reduction = reduction
                    max_dir_idx = i
        
        # Check convergence
        if obj <= 1e-16:
            return x
        
        # Replace the direction with the largest reduction
        if max_reduction > 1e-12:
            new_dir = x - old_x
            dir_norm = np.linalg.norm(new_dir)
            if dir_norm > 1e-12:
                new_dir = new_dir / dir_norm
                directions[max_dir_idx] = new_dir
        
        # Periodic axis reset to maintain conjugacy
        if (_iter + 1) % 2 == 0:
            directions = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        
        # Convergence check
        if np.linalg.norm(x - old_x) < 1e-12 and max_reduction < 1e-14:
            break
    
    return x

def _broyden_method(F, x0, max_iter=40):
    """Broyden's method as fallback."""
    x = np.array(x0, dtype=float, copy=True)
    Fx = _safe_call(F, x)
    if Fx is None:
        return x
    if _norm(Fx) <= 1e-8:
        return x
    
    # Initialize Jacobian with finite differences
    J = np.zeros((2, 2))
    eps = 1e-6
    for j in range(2):
        e = np.zeros(2)
        e[j] = eps
        Fp = _safe_call(F, x + e)
        if Fp is None:
            return x
        J[:, j] = (Fp - Fx) / eps
    
    for _ in range(max_iter):
        if _elapsed() > 115:
            break
        
        try:
            step = np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            break
        
        if not _is_finite(step):
            break
        
        # Line search
        t = 1.0
        accepted = False
        for _ls in range(15):
            new_x = x - t * step
            new_Fx = _safe_call(F, new_x)
            if new_Fx is None:
                t *= 0.5
                continue
            if _norm(new_Fx) < _norm(Fx) * (1 - 1e-4 * t):
                accepted = True
                break
            t *= 0.5
        
        if not accepted:
            new_x = x - 0.5 * step
            new_Fx = _safe_call(F, new_x)
            if new_Fx is None:
                break
            if _norm(new_Fx) > _norm(Fx) * 1.1:
                break
        
        # Broyden update
        dx = new_x - x
        dF = new_Fx - Fx
        Jdx = J @ dx
        numerator = dF - Jdx
        denom = np.dot(dx, dx)
        if denom > 1e-15:
            J = J + np.outer(numerator, dx) / denom
        
        x = new_x
        Fx = new_Fx
        
        if _norm(Fx) <= 1e-8:
            return x
    
    return x

def _grid_search(F, x0, n_grid=4, radius=1.5):
    """Coarse grid search to find a good starting point."""
    x0 = np.array(x0, dtype=float)
    best_x = x0.copy()
    F_best = _safe_call(F, x0)
    best_norm = _norm(F_best) if F_best is not None else np.inf
    
    for i in range(n_grid + 1):
        for j in range(n_grid + 1):
            x_try = x0 + np.array([i / n_grid - 0.5, j / n_grid - 0.5]) * radius
            F_try = _safe_call(F, x_try)
            if F_try is not None:
                n_val = _norm(F_try)
                if n_val < best_norm:
                    best_norm = n_val
                    best_x = x_try
                if n_val <= 1e-8:
                    return x_try
    
    return best_x

def find_root_system(F, x0):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    x0 = np.array(x0, dtype=float)
    
    # Get a good starting point via grid search
    x_start = _grid_search(F, x0, n_grid=3, radius=1.0)
    
    # Primary: Powell's method from grid-searched start
    x1 = _powell_method(F, x_start, max_iter=80)
    F1 = _safe_call(F, x1)
    norm1 = _norm(F1) if F1 is not None else np.inf
    
    if norm1 <= 1e-8 and _is_finite(x1):
        return x1
    
    # Secondary: Powell's method from original x0
    x2 = _powell_method(F, x0, max_iter=80)
    F2 = _safe_call(F, x2)
    norm2 = _norm(F2) if F2 is not None else np.inf
    
    if norm2 <= 1e-8 and _is_finite(x2):
        return x2
    
    # Tertiary: Broyden's method from grid-searched start
    x3 = _broyden_method(F, x_start, max_iter=30)
    F3 = _safe_call(F, x3)
    norm3 = _norm(F3) if F3 is not None else np.inf
    
    if norm3 <= 1e-8 and _is_finite(x3):
        return x3
    
    # Quaternary: Broyden's method from original x0
    x4 = _broyden_method(F, x0, max_iter=30)
    F4 = _safe_call(F, x4)
    norm4 = _norm(F4) if F4 is not None else np.inf
    
    if norm4 <= 1e-8 and _is_finite(x4):
        return x4
    
    # Return the best candidate found
    candidates = []
    if _is_finite(x1):
        candidates.append((x1, norm1))
    if _is_finite(x2):
        candidates.append((x2, norm2))
    if _is_finite(x3):
        candidates.append((x3, norm3))
    if _is_finite(x4):
        candidates.append((x4, norm4))
    
    if candidates:
        best_x, _ = min(candidates, key=lambda c: c[1])
        return best_x
    
    return x0
# EVOLVE-BLOCK-END
