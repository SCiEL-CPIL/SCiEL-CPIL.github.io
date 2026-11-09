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

def find_root_system(F, x0):
    global _START_TIME
    _START_TIME = time.time()
    deadline = _START_TIME + 115.0  # Safety margin from 120s limit

    x = np.array(x0, dtype=float)
    
    # Initial evaluation
    Fx = _safe_call(F, x)
    if Fx is None:
        # If initial point is invalid, try a small perturbation
        x = x + np.array([0.1, 0.1])
        Fx = _safe_call(F, x)
        if Fx is None:
            return x

    if _norm(Fx) <= 1e-8:
        return x

    # Strategy: Hybrid Nelder-Mead + Polishing
    # Phase 1: Nelder-Mead on phi(x) = ||F(x)||^2
    # This is robust and does not require Jacobians.
    
    # Initialize simplex
    # Start with x0, and two points perturbed by a small step
    step = 0.1
    x1 = x + np.array([step, 0.0])
    x2 = x + np.array([0.0, step])
    
    # Ensure initial points are valid
    Fx1 = _safe_call(F, x1)
    Fx2 = _safe_call(F, x2)
    
    if Fx1 is None:
        x1 = x + np.array([0.01, 0.01])
        Fx1 = _safe_call(F, x1)
    if Fx2 is None:
        x2 = x + np.array([0.01, -0.01])
        Fx2 = _safe_call(F, x2)
        
    if Fx1 is None or Fx2 is None:
        # Fallback to simple Broyden if NM can't start
        return _broyden_fallback(F, x, Fx, deadline)

    simplex = [x, x1, x2]
    f_values = [_norm(Fx)**2, _norm(Fx1)**2, _norm(Fx2)**2]
    
    # Nelder-Mead parameters
    alpha = 1.0  # reflection
    gamma = 2.0  # expansion
    rho = 0.5   # contraction
    sigma = 0.5 # shrink
    
    max_iter_nm = 200
    best_x = x.copy()
    best_f = f_values[0]
    
    for i in range(max_iter_nm):
        if _elapsed() > deadline - 5.0: # Leave 5s for polishing
            break
            
        # Sort simplex by function value
        idx = np.argsort(f_values)
        simplex = [simplex[i] for i in idx]
        f_values = [f_values[i] for i in idx]
        
        best_x = simplex[0].copy()
        best_f = f_values[0]
        
        if np.sqrt(best_f) <= 1e-8:
            break
            
        # Check convergence of simplex
        if max(f_values) - min(f_values) < 1e-12:
            break
            
        # Centroid of all but the worst point
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        Fxr = _safe_call(F, xr)
        if Fxr is None:
            f_xr = np.inf
        else:
            f_xr = _norm(Fxr)**2
            
        if f_values[0] <= f_xr < f_values[-1]:
            # Accept reflection
            simplex[-1] = xr
            f_values[-1] = f_xr
        elif f_xr < f_values[0]:
            # Expansion
            xe = centroid + gamma * (xr - centroid)
            Fxe = _safe_call(F, xe)
            if Fxe is None:
                f_xe = np.inf
            else:
                f_xe = _norm(Fxe)**2
                
            if f_xe < f_xr:
                simplex[-1] = xe
                f_values[-1] = f_xe
            else:
                simplex[-1] = xr
                f_values[-1] = f_xr
        else:
            # Contraction
            xc = centroid + rho * (simplex[-1] - centroid)
            Fxc = _safe_call(F, xc)
            if Fxc is None:
                f_xc = np.inf
            else:
                f_xc = _norm(Fxc)**2
                
            if f_xc < f_values[-1]:
                simplex[-1] = xc
                f_values[-1] = f_xc
            else:
                # Shrink
                for j in range(1, len(simplex)):
                    simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                    Fsj = _safe_call(F, simplex[j])
                    if Fsj is None:
                        f_values[j] = np.inf
                    else:
                        f_values[j] = _norm(Fsj)**2

    # Phase 2: Polishing
    # If NM got us close, use a local derivative-based method to hit 1e-8 precisely.
    # We use Broyden here, initialized with the best point from NM.
    # We need to check if we are close enough to warrant the cost of Jacobian estimation.
    # NM is derivative-free, so we haven't built a Jacobian. We must estimate it.
    # Cost: 2 calls for FD Jacobian.
    
    current_x = best_x
    current_F = _safe_call(F, current_x)
    if current_F is None:
        return best_x
        
    current_norm = _norm(current_F)
    
    # If NM didn't get us very close, maybe just return best_x.
    # But the target is 1e-8. NM might stop at 1e-6 or 1e-7.
    # We attempt a few Broyden steps if we are reasonably close (< 1e-4).
    if current_norm < 1e-4 and _elapsed() < deadline - 1.0:
        current_x = _broyden_polish(F, current_x, current_F, deadline)
        
    return current_x

def _broyden_fallback(F, x, Fx, deadline):
    """Fallback Broyden if Nelder-Mead fails to start."""
    h = 1e-6
    e1 = np.array([h, 0.0])
    e2 = np.array([0.0, h])
    
    Fx1 = _safe_call(F, x + e1)
    Fx2 = _safe_call(F, x + e2)
    
    if Fx1 is None or Fx2 is None:
        return x

    J = np.zeros((2, 2))
    J[:, 0] = (Fx1 - Fx) / h
    J[:, 1] = (Fx2 - Fx) / h

    return _broyden_polish(F, x, Fx, deadline, J_init=J)

def _broyden_polish(F, x, Fx, deadline, J_init=None):
    """
    High-precision Broyden's method.
    Assumes we are close enough that the Jacobian estimate is decent.
    """
    if J_init is None:
        # Estimate Jacobian
        h = 1e-6
        e1 = np.array([h, 0.0])
        e2 = np.array([0.0, h])
        
        Fx1 = _safe_call(F, x + e1)
        Fx2 = _safe_call(F, x + e2)
        
        if Fx1 is None or Fx2 is None:
            return x

        J = np.zeros((2, 2))
        J[:, 0] = (Fx1 - Fx) / h
        J[:, 1] = (Fx2 - Fx) / h
    else:
        J = J_init.copy()

    x_old = x
    Fx_old = Fx
    best_x = x.copy()
    best_norm = _norm(Fx)
    
    max_iter = 50
    
    for i in range(max_iter):
        if _elapsed() > deadline:
            break
            
        if best_norm <= 1e-8:
            break

        # Solve J * d = -Fx
        try:
            d = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Regularize
            try:
                d = np.linalg.solve(J + 1e-8 * np.eye(2), -Fx)
            except np.linalg.LinAlgError:
                break

        if not np.all(np.isfinite(d)):
            break
            
        # Line search
        alpha = 1.0
        norm_current = _norm(Fx)
        found_step = False
        
        for _ls in range(20):
            if _elapsed() > deadline:
                break
                
            x_new = x + alpha * d
            Fx_new = _safe_call(F, x_new)
            
            if Fx_new is None:
                alpha *= 0.5
                continue
                
            norm_new = _norm(Fx_new)
            
            if norm_new < norm_current:
                x = x_new
                Fx = Fx_new
                found_step = True
                break
            else:
                alpha *= 0.5
                
        if not found_step:
            break
            
        if norm_new < best_norm:
            best_norm = norm_new
            best_x = x.copy()
            
        if best_norm <= 1e-8:
            break

        # Broyden Update
        s = x - x_old
        y = Fx - Fx_old
        s_dot_s = np.dot(s, s)
        
        if s_dot_s < 1e-16:
            break
            
        J = J + np.outer(y - J @ s, s) / s_dot_s
        x_old = x
        Fx_old = Fx

    return best_x
