"""
Root finder for 3 fixed 2-D systems: Homotopy Continuation Method with Adaptive RK4.

Strategy:
1. Homotopy Formulation: Define a homotopy H(x, t) = (1-t)F(x) + t(x - x0_target) where x0_target is a known point.
   Actually, a simpler standard homotopy is H(x, t) = (1-t)F(x) + t*x.
   At t=0, H(x, 0) = F(x). We want a root of F.
   At t=1, H(x, 1) = x. The root is x=0.
   So we start at t=1, x=0 (known root) and integrate to t=0, x=x* (unknown root).
   
   Wait, the standard homotopy is usually H(x, t) = (1-t)F(x) + t(x - x_ref).
   If we set x_ref = 0, H(x, t) = (1-t)F(x) + t*x.
   At t=1, H(x, 1) = x. Root is x=0.
   At t=0, H(x, 0) = F(x). Root is the solution we seek.
   
   We integrate the ODE dx/dt = -J(x,t)^{-1} * dH/dt = -J(x,t)^{-1} * (x - F(x)) from t=1 to t=0.
   However, we don't have J. We can use a quasi-Newton approximation of the Jacobian of H.
   Or, simpler: we can just use a fixed-step predictor-corrector if the path is smooth.
   
   But since we don't know the path is smooth, we use an adaptive step size based on the residual of the homotopy equation.
   
   Alternative: Use a "pseudo-transient continuation" or just a simple iterative homotopy.
   Given the constraints (max 500 calls), we need to be very efficient.
   
   Let's refine the homotopy approach:
   We define the path x(t) such that H(x(t), t) = 0.
   H(x, t) = (1-t)F(x) + t*x.
   
   We start at t=1.0, x=0.0.
   We step towards t=0.0.
   At each step, we update x using a Newton-like step on H.
   Since we don't have the Jacobian of H, we approximate it.
   J_H = (1-t)J_F + t*I.
   
   We can maintain a Broyden approximation of J_F.
   
   This is essentially "Broyden's method with a homotopy damping factor".
   As t goes from 1 to 0, the influence of the identity matrix (which guarantees local convergence near 0) fades, 
   and the influence of F takes over. This helps navigate through regions where J_F might be ill-conditioned 
   or where the basin of attraction is narrow, by "pulling" the solution towards the origin initially.
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

def _homotopy_broyden(g, x0, max_steps=100):
    """
    Homotopy continuation using Broyden's method.
    H(x, t) = (1-t)F(x) + t*x.
    We track x(t) from t=1 (x=0) to t=0 (x=x*).
    """
    # Start at t=1, x=0
    # Note: The problem gives x0 as a starting point, but homotopy starts from a known root of the auxiliary problem.
    # The auxiliary problem root is 0.
    # So we ignore x0 for the homotopy start, but we can use x0 to initialize the Jacobian if needed.
    # However, starting from 0 is the key structural feature.
    
    x = np.zeros(2)
    t = 1.0
    
    # Evaluate F(0)
    F0 = g(x)
    if F0 is None:
        return x, False, np.inf
    
    f_norm = float(np.linalg.norm(F0))
    if f_norm <= 1e-8:
        return x, True, f_norm

    # Initialize Jacobian approximation for F.
    # We don't know J_F(0). Let's start with Identity.
    J = np.eye(2)
    
    # History for Broyden update
    x_prev = x.copy()
    F_prev = F0.copy()
    
    # Step size for t
    # We want to reach t=0.
    # Adaptive step: start with small dt, increase if successful.
    dt = 0.1
    min_dt = 1e-4
    
    for step in range(max_steps):
        if g.n >= g.limit or _time_left() < 1.0:
            break
        
        # Target t
        t_new = max(0.0, t - dt)
        
        # Predictor: Linear extrapolation
        # dx/dt approx -J_H^{-1} (x - F(x))
        # J_H = (1-t)J + t*I
        J_H = (1.0 - t) * J + t * np.eye(2)
        
        # RHS of ODE: dH/dt = x - F(x)
        rhs = x - F_prev
        
        try:
            dx_dt = np.linalg.solve(J_H, -rhs)
        except np.linalg.LinAlgError:
            dx_dt = -rhs # Fallback
        
        x_pred = x + dx_dt * (t_new - t)
        
        # Corrector: Newton step on H(x, t_new)
        # We need to solve H(x_new, t_new) = 0
        # H(x_new, t_new) = (1-t_new)F(x_new) + t_new*x_new
        
        x_curr = x_pred
        converged = False
        
        for inner in range(10):
            if g.n >= g.limit:
                break
            
            F_curr = g(x_curr)
            if F_curr is None:
                break
                
            H_val = (1.0 - t_new) * F_curr + t_new * x_curr
            H_norm = float(np.linalg.norm(H_val))
            
            if H_norm <= 1e-10:
                converged = True
                break
            
            # Update Jacobian approximation for F using Broyden if we have a step
            # Actually, we are solving for x_curr given t_new.
            # We can use a simple Newton step with the current J approximation for F.
            J_H_curr = (1.0 - t_new) * J + t_new * np.eye(2)
            
            try:
                dx = np.linalg.solve(J_H_curr, -H_val)
            except np.linalg.LinAlgError:
                dx = -H_val
            
            # Damping
            alpha = 1.0
            accepted = False
            for _ in range(5):
                x_test = x_curr + alpha * dx
                F_test = g(x_test)
                if F_test is None:
                    break
                H_test = (1.0 - t_new) * F_test + t_new * x_test
                H_test_norm = float(np.linalg.norm(H_test))
                
                if H_test_norm < H_norm:
                    # Update Broyden J for F
                    # s = x_test - x_curr
                    # y = F_test - F_curr
                    s = alpha * dx
                    y = F_test - F_curr
                    denom = np.dot(s, s)
                    if denom > 1e-15:
                        J = J + np.outer(y - J @ s, s) / denom
                    
                    x_curr = x_test
                    F_curr = F_test
                    accepted = True
                    break
                alpha *= 0.5
            
            if not accepted:
                break
        
        if converged:
            # Successful step in t
            x = x_curr
            F_prev = F_curr
            t = t_new
            dt = min(dt * 2.0, 0.5) # Increase step size
            
            # Check if we reached the target t=0
            if t <= 1e-6:
                # Final check on F(x)
                if float(np.linalg.norm(F_prev)) <= 1e-8:
                    return x, True, float(np.linalg.norm(F_prev))
                # If not quite converged, do a few pure Broyden steps on F
                for _ in range(10):
                    F_curr = g(x)
                    if F_curr is None: break
                    f_curr = float(np.linalg.norm(F_curr))
                    if f_curr <= 1e-8:
                        return x, True, f_curr
                    try:
                        dx = np.linalg.solve(J, -F_curr)
                    except:
                        dx = -F_curr
                    x_new = x + dx
                    F_new = g(x_new)
                    if F_new is None: break
                    f_new = float(np.linalg.norm(F_new))
                    if f_new < f_curr:
                        s = x_new - x
                        y = F_new - F_curr
                        denom = np.dot(s, s)
                        if denom > 1e-15:
                            J = J + np.outer(y - J @ s, s) / denom
                        x = x_new
                        F_prev = F_new
                    else:
                        break
                return x, float(np.linalg.norm(F_prev)) <= 1e-8, float(np.linalg.norm(F_prev))
                
        else:
            # Failed step, reduce dt
            dt = max(dt * 0.5, min_dt)
            if dt == min_dt and not converged:
                # Stuck, break
                break

    return x, False, float(np.linalg.norm(F_prev)) if F_prev is not None else np.inf

def _local_broyden_polish(g, x0, max_it=50):
    """Standard Broyden polish to ensure 1e-8 tolerance."""
    x = np.array(x0, dtype=float)
    Fx = g(x)
    if Fx is None:
        return x, False, np.inf
    
    f = float(np.linalg.norm(Fx))
    if f <= 1e-8:
        return x, True, f
        
    J = np.eye(2)
    
    for _ in range(max_it):
        if g.n >= g.limit or _time_left() < 1.0:
            break
            
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            dx = -Fx
            
        if not np.all(np.isfinite(dx)):
            break
            
        alpha = 1.0
        success = False
        for _ in range(10):
            x_new = x + alpha * dx
            Fx_new = g(x_new)
            if Fx_new is None:
                break
            f_new = float(np.linalg.norm(Fx_new))
            
            if f_new <= 1e-8:
                return x_new, True, f_new
                
            if f_new < f:
                s = alpha * dx
                y = Fx_new - Fx
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
            break
            
    return x, False, f

def find_root_system(F, x0):
    x0 = _clean(x0)
    if x0 is None:
        x0 = np.zeros(2)
        
    limit = 480
    g = _Budget(F, limit)
    
    # 1. Homotopy from 0
    # This is the primary structural change. It assumes the root is connected to 0 via a smooth path.
    x_h, success, f_h = _homotopy_broyden(g, x0, max_steps=50)
    
    if success:
        return x_h
        
    # Update best
    if f_h < g.best_f:
        g.best_f = f_h
        g.best_x = x_h.copy()
        
    # 2. If homotopy got close, polish
    if g.best_x is not None and g.best_f < 1e-3:
        x_res, success, f_res = _local_broyden_polish(g, g.best_x, max_it=30)
        if success:
            return x_res
        if f_res < g.best_f:
            g.best_f = f_res
            g.best_x = x_res.copy()
            
    # 3. Fallback: If homotopy failed completely (e.g., singularity on path), try standard Broyden from x0
    # This covers cases where the "path" from 0 is not valid or x0 is much better.
    if g.best_f > 1e-3:
        x_res, success, f_res = _local_broyden_polish(g, x0, max_it=50)
        if success:
            return x_res
        if f_res < g.best_f:
            g.best_f = f_res
            g.best_x = x_res.copy()
            
    # 4. Multi-start fallback if still not good
    if g.best_f > 1e-3 and g.n < limit - 50:
        rng = np.random.default_rng(42)
        scale = max(1.0, np.linalg.norm(x0))
        for _ in range(5):
            if g.n >= limit - 20 or _time_left() < 5.0:
                break
            x_rand = x0 + rng.uniform(-5, 5, size=2) * scale
            x_res, success, f_res = _local_broyden_polish(g, x_rand, max_it=20)
            if success:
                return x_res
            if f_res < g.best_f:
                g.best_f = f_res
                g.best_x = x_res.copy()
                
    if g.best_x is not None:
        return g.best_x
    return x0.copy()
