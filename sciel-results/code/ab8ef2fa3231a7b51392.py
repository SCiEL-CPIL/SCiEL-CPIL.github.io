"""
Root finder for 3 fixed 2-D systems: Global Grid + Broyden's Method.

Strategy:
1. Global Phase: Perform a coarse scan (grid + random) to find a point with
   low ||F(x)||. This identifies the basin of attraction.
2. Local Phase: Use Broyden's method (quasi-Newton) with aggressive
   backtracking from the best global candidate. Broyden is chosen over
   Newton to avoid the cost of finite-difference Jacobians (which cost 2
   extra calls per iteration) and over Nelder-Mead because it converges
   quadratically near the root if started close enough.
3. Fallback: If Broyden fails or exceeds budget, return the best point found
   during the global scan or the initial guess.
"""

import time
import numpy as np

# ---------------------------------------------------------------------------
# Global deadline guard (evaluator kills at ~150 s; use a safety margin).
# ---------------------------------------------------------------------------
_DEADLINE = time.time() + 140.0
_EPS = 1e-12

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
    """Counts F calls and tracks the best (smallest-norm) candidate seen."""

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

def _global_scan(g, x0, budget):
    """
    Perform a coarse global scan to find a good starting point.
    Uses a grid around x0 and some random points.
    Returns the best x found.
    """
    best_x = x0.copy()
    best_f = np.inf
    
    # Initial evaluation
    Fx0 = g(x0)
    if Fx0 is not None:
        f0 = float(np.linalg.norm(Fx0))
        if f0 < best_f:
            best_f = f0
            best_x = x0.copy()
    
    # If x0 is already very good, skip scan
    if best_f < 1e-6:
        return best_x

    # Grid scan around x0
    # Scale the grid based on x0 magnitude
    scale = max(1.0, np.linalg.norm(x0))
    
    # Define grid points
    # We want to spend roughly `budget` calls in the global phase.
    # Let's use a 5x5 grid = 25 points, plus some random.
    # Actually, let's be more adaptive.
    
    # 1. Grid around x0
    radii = [0.5 * scale, scale, 2.0 * scale, 5.0 * scale]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
    
    candidates = []
    for r in radii:
        for a in angles:
            dx = r * np.cos(a)
            dy = r * np.sin(a)
            candidates.append(x0 + np.array([dx, dy]))
    
    # 2. Random points in a larger box
    rng = np.random.RandomState(42)
    for _ in range(10):
        dx = rng.uniform(-5 * scale, 5 * scale)
        dy = rng.uniform(-5 * scale, 5 * scale)
        candidates.append(x0 + np.array([dx, dy]))
    
    # Evaluate candidates
    for x in candidates:
        if g.n >= g.limit or _time_left() < 1.0:
            break
        Fx = g(x)
        if Fx is None:
            continue
        f = float(np.linalg.norm(Fx))
        if f < best_f:
            best_f = f
            best_x = x.copy()
            
        # If we find a very good point, stop early
        if best_f < 1e-6:
            break

    return best_x

def _broyden_local(g, x0, limit, max_it=100):
    """
    Broyden's method with backtracking.
    """
    x = np.array(x0, dtype=float)
    # Initialize Jacobian approximation
    # We don't have a Jacobian, so start with Identity or a scaled Identity
    J = np.eye(2)
    
    Fx = g(x)
    if Fx is None:
        return None
    
    f = float(np.linalg.norm(Fx))
    if f <= 1e-10:
        return x

    for _ in range(max_it):
        if g.n >= limit or _time_left() < 1.0:
            break
            
        try:
            # Solve J dx = -Fx
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If singular, reset J to Identity and try again? Or break?
            # Let's try a gradient step if solve fails
            # Approximate gradient of 0.5 * ||F||^2 is J^T F
            # But we don't have J^T. We have J.
            # Let's just break if J is singular and we can't recover easily.
            break
        
        if not np.all(np.isfinite(dx)):
            break
            
        # Backtracking line search
        step = dx
        ok = False
        for _ in range(20):
            xn = x + step
            Fx_new = g(xn)
            if Fx_new is None:
                break
            f_new = float(np.linalg.norm(Fx_new))
            
            if f_new <= 1e-10:
                return xn
                
            # Armijo condition: f_new <= f + c * (grad_f . step)
            # grad_f = J^T Fx (approx)
            # Since we are minimizing ||F||^2, the directional derivative is 2 * Fx^T J^T step
            # We want f_new < f
            # Simple condition: f_new < f
            if f_new < f:
                # Update J using Broyden's formula
                # J_new = J + (Fx_new - Fx - J * dx) * dx^T / (dx^T * dx)
                r = Fx_new - Fx - J @ dx
                denom = dx @ dx
                if denom < 1e-12:
                    J_new = J
                else:
                    J_new = J + np.outer(r, dx) / denom
                
                x = xn
                Fx = Fx_new
                f = f_new
                J = J_new
                ok = True
                break
            else:
                step = 0.5 * step
                
        if not ok:
            # If backtracking fails, try a smaller step or break
            # Let's try a gradient step with a small step size
            # Approx gradient: J^T Fx
            try:
                grad = J.T @ Fx
                if np.linalg.norm(grad) > 1e-12:
                    step = -0.01 * grad
                    xn = x + step
                    Fx_new = g(xn)
                    if Fx_new is not None:
                        f_new = float(np.linalg.norm(Fx_new))
                        if f_new < f:
                            x = xn
                            Fx = Fx_new
                            f = f_new
                            # Update J? No, this is not a Newton step.
                            # Keep J same.
                            continue
            except Exception:
                pass
            break
            
    return x

def find_root_system(F, x0):
    x0 = _clean(x0)
    if x0 is None:
        x0 = np.zeros(2)

    # Per-system call budget: evaluator's final check is free, so stay safely
    # below 500. Use a bit of headroom.
    limit = 490

    g = _Budget(F, limit)
    
    # Phase 1: Global Scan
    # Allocate ~50 calls for global scan
    global_budget = 50
    x_start = _global_scan(g, x0, global_budget)
    
    # Phase 2: Local Broyden
    x_result = _broyden_local(g, x_start, limit)
    
    if x_result is None:
        x_result = g.best()
        
    if x_result is None:
        x_result = x0.copy()
        
    return x_result
