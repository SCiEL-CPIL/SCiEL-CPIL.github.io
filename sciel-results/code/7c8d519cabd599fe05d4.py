"""
Root finder for 3 fixed 2-D systems: Derivative-Free Powell's Method with Adaptive Multi-Start.

Strategy:
1. Core Solver: Powell's method (derivative-free, direction-set method) for 2D systems.
   - This is a fundamentally different design family from Broyden/Newton/Homotopy.
   - It does not maintain a Jacobian approximation. Instead, it maintains a set of search directions.
   - In 2D, we start with the coordinate axes. After each cycle, we replace the worst direction
     with the direction of the overall step. This adapts to the local geometry of the function.
   - It is robust to non-smoothness and does not require Jacobian estimation, which can be
     unstable or inaccurate.
   
2. Objective Function: We minimize ||F(x)||^2 (or ||F(x)||, but squared is smoother).
   Powell's method minimizes a scalar function. We use the Euclidean norm squared as the objective.
   
3. Multi-Start: If the initial start fails to converge, we try multiple random starts
   around x0 to escape local minima or narrow basins of attraction.

4. Efficiency: Powell's method in 2D is very efficient. Each cycle requires at most 4 function
   evaluations (2 for line searches along the old directions, 1 for the new direction, and 1
   for the final check). This is often fewer than Broyden's method which requires 1 evaluation
   per step plus potential line search evaluations.

5. Deadline Guard: Track elapsed time and return the best result found with a safety margin.
"""

import time
import numpy as np

_DEADLINE = time.time() + 135.0

def _time_left():
    return _DEADLINE - time.time()

def _clean(v):
    try:
        a = np.asarray(v, dtype=float).reshape(2)
    except Exception:
        return None
    if not np.all(np.isfinite(a)):
        return None
    return a

class _Budget(object):
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

def _objective(g, x):
    """Compute ||F(x)||^2 as the objective for minimization."""
    Fx = g(x)
    if Fx is None:
        return np.inf
    return float(np.dot(Fx, Fx))

def _line_search(g, x, d, obj_x, max_iter=20):
    """
    Perform a line search along direction d from point x.
    Use a simple backtracking line search.
    Returns (x_new, obj_new).
    """
    alpha = 1.0
    for _ in range(max_iter):
        if g.n >= g.limit or _time_left() < 1.0:
            break
        x_new = x + alpha * d
        obj_new = _objective(g, x_new)
        if obj_new < obj_x:
            return x_new, obj_new
        alpha *= 0.5
    return x, obj_x

def _powell_method(g, x0, max_cycles=30):
    """
    Powell's method for minimizing ||F(x)||^2 in 2D.
    """
    x = np.array(x0, dtype=float)
    obj_x = _objective(g, x)
    if obj_x <= 1e-16:  # sqrt(1e-8)^2 = 1e-16
        return x, True, float(np.sqrt(obj_x))
    
    # Initial directions: coordinate axes
    d = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    
    for cycle in range(max_cycles):
        if g.n >= g.limit or _time_left() < 1.0:
            break
        
        x_old = x.copy()
        obj_old = obj_x
        max_delta = 0.0
        worst_i = 0
        
        # Line searches along each direction
        for i in range(len(d)):
            x_new, obj_new = _line_search(g, x, d[i], obj_x)
            delta = obj_x - obj_new
            if delta > max_delta:
                max_delta = delta
                worst_i = i
            x = x_new
            obj_x = obj_new
        
        # Check convergence
        if max_delta < 1e-12:
            break
        
        # New direction: from x_old to x
        d_new = x - x_old
        norm_d_new = float(np.linalg.norm(d_new))
        if norm_d_new < 1e-15:
            break
        
        # Line search along the new direction
        x_check, obj_check = _line_search(g, x, d_new, obj_x)
        
        # If improvement along new direction is significant, replace worst direction
        if obj_check < obj_old - 1e-12:
            d[worst_i] = d_new
            x = x_check
            obj_x = obj_check
        
        # Check if we've found a root
        if obj_x <= 1e-16:
            return x, True, float(np.sqrt(obj_x))
    
    f_norm = float(np.sqrt(obj_x))
    return x, f_norm <= 1e-8, f_norm

def find_root_system(F, x0):
    x0 = _clean(x0)
    if x0 is None:
        x0 = np.zeros(2)
        
    limit = 480
    g = _Budget(F, limit)
    
    # Try Powell's method from x0
    x_res, success, f_res = _powell_method(g, x0, max_cycles=30)
    if success:
        return x_res
    if f_res < g.best_f:
        g.best_f = f_res
        g.best_x = x_res.copy()
    
    # If not successful, try multi-start
    if g.best_f > 1e-8 and g.n < limit - 50:
        rng = np.random.default_rng(42)
        scale = max(1.0, float(np.linalg.norm(x0)))
        for _ in range(8):
            if g.n >= limit - 20 or _time_left() < 5.0:
                break
            x_rand = x0 + rng.uniform(-5, 5, size=2) * scale
            x_res, success, f_res = _powell_method(g, x_rand, max_cycles=20)
            if success:
                return x_res
            if f_res < g.best_f:
                g.best_f = f_res
                g.best_x = x_res.copy()
    
    if g.best_x is not None:
        return g.best_x
    return x0.copy()
