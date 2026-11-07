"""
Root finder for 3 fixed 2-D systems: Hybrid Global-Local Search Strategy.

Strategy:
1. Global Phase: Perform a coarse scan (grid + random) to find a point with
   low ||F(x)||. This identifies the basin of attraction.
2. Local Phase: Use Broyden's method (quasi-Newton) with aggressive
   backtracking from the best global candidate. Broyden is chosen over
   Newton to avoid the cost of finite-difference Jacobians (which cost 2
   extra calls per iteration) and over Nelder-Mead because it converges
   quadratically near the root if started close enough.
3. Fallback: If the primary local search fails, try a few other top global
   candidates with a simpler damped Newton step.

This approach trades some initial global exploration calls for a much more
efficient local convergence, aiming to stay well under the 500 call limit
while robustly handling systems where x0 is far from the root.
"""
import time
import numpy as np

# Global deadline guard
_DEADLINE = time.time() + 135.0

def _time_left():
    return _DEADLINE - time.time()

def _clean(v):
    """Coerce to a finite float length-2 array, else None."""
    try:
        a = np.asarray(v, dtype=float).reshape(2)
    except Exception:
        return None
    if not np.all(np.isfinite(a)):
        return None
    return a

class _Budget(object):
    """Counts F calls and tracks the best candidate seen."""
    __slots__ = ("F", "limit", "n", "best_x", "best_f")

    def __init__(self, F, limit):
        self.F = F
        self.limit = limit
        self.n = 0
        self.best_x = None
        self.best_f = np.inf

    def __call__(self, x):
        if self.n >= self.limit:
            return None
        self.n += 1
        try:
            v = self.F(x)
        except Exception:
            return None
        v = _clean(v)
        if v is None:
            return None
        f = float(np.linalg.norm(v))
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return v

    def best(self):
        return self.best_x

def _broyden_local(g, x0, max_it=50):
    """
    Broyden's method with backtracking.
    Efficient local solver: 1 call per iteration + backtracking calls.
    """
    x = np.array(x0, dtype=float)
    J = np.eye(2)
    
    # Initial evaluation
    Fx = g(x)
    if Fx is None:
        return x
    f = float(np.linalg.norm(Fx))
    if f <= 1e-8:
        return x

    for _ in range(max_it):
        if g.n >= g.limit or _time_left() < 0.5:
            break
        
        # Solve J dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If J is singular, reset to identity and try again
            J = np.eye(2)
            try:
                dx = -Fx
            except Exception:
                break
        
        if not np.all(np.isfinite(dx)):
            break
            
        # Backtracking line search
        alpha = 1.0
        success = False
        for _ in range(10):
            x_new = x + alpha * dx
            Fx_new = g(x_new)
            if Fx_new is None:
                break
            f_new = float(np.linalg.norm(Fx_new))
            
            if f_new <= 1e-8:
                return x_new
            
            # Armijo condition: sufficient decrease
            # Simplified: just check if we decreased
            if f_new < f:
                # Update Broyden
                s = alpha * dx
                y = Fx_new - Fx
                # Broyden update: J_new = J + (y - J*s) * s^T / (s^T * s)
                denom = np.dot(s, s)
                if denom > 1e-15:
                    J = J + np.outer(y - J @ s, s) / denom
                
                x = x_new
                Fx = Fx_new
                f = f_new
                success = True
                break
            alpha *= 0.5
        
        if not success:
            # If backtracking fails, try a small random step? Or break?
            # Breaking is safer to avoid wasting calls
            break
            
    return x

def _damped_newton_local(g, x0, max_it=20):
    """
    Simple damped Newton with finite difference Jacobian.
    Used as a fallback if Broyden fails.
    """
    x = np.array(x0, dtype=float)
    h = 1e-5
    
    for _ in range(max_it):
        if g.n >= g.limit or _time_left() < 0.5:
            break
        
        Fx = g(x)
        if Fx is None:
            break
        f = float(np.linalg.norm(Fx))
        if f <= 1e-8:
            return x
        
        # Compute FD Jacobian
        J = np.zeros((2, 2))
        valid = True
        for j in range(2):
            xp = x.copy()
            xp[j] += h
            Fp = g(xp)
            if Fp is None:
                valid = False
                break
            J[:, j] = (Fp - Fx) / h
        
        if not valid:
            break
            
        try:
            d = np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            break
            
        if not np.all(np.isfinite(d)):
            break
            
        # Damping
        alpha = 1.0
        success = False
        for _ in range(8):
            x_new = x - alpha * d
            Fx_new = g(x_new)
            if Fx_new is None:
                break
            f_new = float(np.linalg.norm(Fx_new))
            if f_new < f:
                x = x_new
                f = f_new
                success = True
                break
            alpha *= 0.5
            
        if not success:
            break
            
    return x

def _global_scan(g, x0, budget=50):
    """
    Perform a coarse global search to find a good starting point.
    Uses a mix of grid points around x0 and random points.
    """
    x0 = np.array(x0, dtype=float)
    scale = max(1.0, np.linalg.norm(x0))
    
    candidates = []
    
    # 1. Grid points around x0
    # Try offsets in [-2, -1, 0, 1, 2] * scale
    # But limit to a few to save calls
    offsets = [
        [0, 0],
        [1, 0], [-1, 0], [0, 1], [0, -1],
        [1, 1], [-1, -1], [1, -1], [-1, 1],
        [2, 0], [-2, 0], [0, 2], [0, -2],
        [2, 2], [-2, -2]
    ]
    
    for off in offsets:
        if g.n >= budget:
            break
        x = x0 + np.array(off) * scale
        fx = g(x)
        if fx is not None:
            candidates.append((float(np.linalg.norm(fx)), x))
            
    # 2. Random points in a box around x0
    rng = np.random.default_rng(42)
    while g.n < budget:
        x = x0 + rng.uniform(-2 * scale, 2 * scale, size=2)
        fx = g(x)
        if fx is not None:
            candidates.append((float(np.linalg.norm(fx)), x))
            
    # Sort by norm
    candidates.sort(key=lambda t: t[0])
    return candidates

def find_root_system(F, x0):
    x0 = _clean(x0)
    if x0 is None:
        x0 = np.zeros(2)
        
    limit = 480  # Safe margin under 500
    g = _Budget(F, limit)
    
    # Phase 1: Global Scan
    # Reserve some calls for local search
    global_budget = 60
    global_candidates = _global_scan(g, x0, budget=global_budget)
    
    # If we found a root during global scan, return it
    if g.best_f <= 1e-8:
        return g.best_x
        
    # Phase 2: Local Search from best global candidates
    # Try the top 3 global candidates with Broyden
    best_x = g.best_x
    best_f = g.best_f
    
    top_candidates = [c[1] for c in global_candidates[:3]]
    # Ensure x0 is also considered if it wasn't in the top 3
    if not any(np.allclose(c, x0) for c in top_candidates):
        top_candidates.append(x0)
        
    for x_start in top_candidates:
        if g.n >= limit - 20 or _time_left() < 5.0:
            break
            
        # Try Broyden
        x_res = _broyden_local(g, x_start, max_it=40)
        x_res = _clean(x_res)
        if x_res is not None:
            fx = g(x_res)
            if fx is not None:
                f_val = float(np.linalg.norm(fx))
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_res
                if best_f <= 1e-8:
                    return best_x
                    
    # Phase 3: Fallback - Try damped Newton from the best candidate found so far
    if best_x is not None and best_f > 1e-8:
        if g.n < limit - 10 and _time_left() > 2.0:
            x_res = _damped_newton_local(g, best_x, max_it=15)
            x_res = _clean(x_res)
            if x_res is not None:
                fx = g(x_res)
                if fx is not None:
                    f_val = float(np.linalg.norm(fx))
                    if f_val < best_f:
                        best_f = f_val
                        best_x = x_res
                        if best_f <= 1e-8:
                            return best_x

    # Final check
    if best_x is None:
        best_x = x0.copy()
        
    return best_x
