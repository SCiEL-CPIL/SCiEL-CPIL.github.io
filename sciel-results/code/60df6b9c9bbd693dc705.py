"""Root finder for 3 fixed 2D systems using Nelder-Mead global search with Broyden polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, root


def _broyden_refine(F, x, J, max_iter=20):
    """
    Refine x using Broyden's method with the provided Jacobian.
    Returns (x, J, converged) where converged is True if ||F(x)|| <= 1e-8.
    """
    x = np.array(x, dtype=float).copy()
    J = np.array(J, dtype=float).copy()
    
    try:
        Fx = F(x)
    except Exception:
        return x, J, False
    
    if not np.all(np.isfinite(Fx)):
        return x, J, False
    
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x, J, True
    
    for _ in range(max_iter):
        try:
            d = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            break
        
        # Simple line search
        alpha = 1.0
        step_accepted = False
        for _ in range(10):
            x_new = x + alpha * d
            try:
                Fx_new = F(x_new)
            except Exception:
                alpha *= 0.5
                continue
            
            if not np.all(np.isfinite(Fx_new)):
                alpha *= 0.5
                continue
            
            norm_Fx_new = np.linalg.norm(Fx_new)
            
            if norm_Fx_new <= 1e-8:
                return x_new, J, True
            
            if norm_Fx_new < (1.0 - 1e-4 * alpha) * norm_Fx:
                # Broyden rank-1 update
                d_dot_d = np.dot(d, d)
                if d_dot_d > 1e-14:
                    delta_F = Fx_new - Fx - J @ d
                    J = J + np.outer(delta_F, d) / d_dot_d
                
                x = x_new
                Fx = Fx_new
                norm_Fx = norm_Fx_new
                step_accepted = True
                break
            else:
                alpha *= 0.5
        
        if not step_accepted:
            break
    
    return x, J, norm_Fx <= 1e-8


def _compute_jacobian(F, x, h=1e-5):
    """Compute Jacobian using central differences."""
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        try:
            F_plus = F(x + e)
            F_minus = F(x - e)
            if np.all(np.isfinite(F_plus)) and np.all(np.isfinite(F_minus)):
                J[:, j] = (F_plus - F_minus) / (2 * h)
            else:
                # Fallback to forward difference
                Fx = F(x)
                if np.all(np.isfinite(F_plus)):
                    J[:, j] = (F_plus - Fx) / h
                elif np.all(np.isfinite(F_minus)):
                    J[:, j] = (Fx - F_minus) / h
                else:
                    J[:, j] = np.array([1.0, 0.0]) if j == 0 else np.array([0.0, 1.0])
        except Exception:
            J[:, j] = np.array([1.0, 0.0]) if j == 0 else np.array([0.0, 1.0])
    return J


def _nelder_mead_global(F, x0, deadline, max_iter=200):
    """
    Use Nelder-Mead to minimize ||F(x)||^2 as a global search.
    Returns the best x found.
    """
    def time_up():
        return time.time() > deadline
    
    def objective(x):
        try:
            Fx = F(x)
            if not np.all(np.isfinite(Fx)):
                return 1e10
            return np.dot(Fx, Fx)
        except Exception:
            return 1e10
    
    best_x = x0.copy()
    best_obj = objective(best_x)
    
    # Initialize simplex around x0
    delta = 0.1
    simplex = [x0.copy()]
    for i in range(2):
        xi = x0.copy()
        xi[i] += delta
        simplex.append(xi)
    
    # Simple Nelder-Mead implementation
    for iteration in range(max_iter):
        if time_up():
            break
        
        # Evaluate objective at all simplex points
        objs = [objective(v) for v in simplex]
        
        # Sort by objective value
        order = np.argsort(objs)
        simplex = [simplex[i] for i in order]
        objs = [objs[i] for i in order]
        
        # Update best
        if objs[0] < best_obj:
            best_obj = objs[0]
            best_x = simplex[0].copy()
        
        # Check convergence
        if objs[0] <= 1e-16:  # ||F||^2 <= 1e-16 means ||F|| <= 1e-8
            break
        
        # Centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        
        # Reflection
        alpha = 1.0
        x_reflect = centroid + alpha * (centroid - worst)
        obj_reflect = objective(x_reflect)
        
        if obj_reflect < objs[0]:
            # Expansion
            gamma = 2.0
            x_expand = centroid + gamma * (centroid - worst)
            obj_expand = objective(x_expand)
            if obj_expand < obj_reflect:
                simplex[-1] = x_expand
            else:
                simplex[-1] = x_reflect
        elif obj_reflect < objs[-2]:
            simplex[-1] = x_reflect
        else:
            # Contraction
            rho = 0.5
            x_contract = centroid + rho * (worst - centroid)
            obj_contract = objective(x_contract)
            if obj_contract < objs[-1]:
                simplex[-1] = x_contract
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
    
    return best_x


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Primary: Nelder-Mead global search. Fallback: Broyden polish.
    """
    x0_arr = np.array(x0, dtype=float).copy()
    if x0_arr.shape != (2,):
        x0_arr = x0_arr.reshape(2)
    
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    def time_up():
        return time.time() > deadline
    
    best_x = x0_arr.copy()
    best_norm = np.inf
    
    # Try to get initial evaluation
    try:
        Fx = F(x0_arr)
        if np.all(np.isfinite(Fx)):
            norm_Fx = np.linalg.norm(Fx)
            if norm_Fx < best_norm:
                best_norm = norm_Fx
                best_x = x0_arr.copy()
            if norm_Fx <= 1e-8:
                return x0_arr
    except Exception:
        pass
    
    # Strategy 1: Nelder-Mead global search
    if not time_up():
        try:
            x_nm = _nelder_mead_global(F, x0_arr, deadline, max_iter=150)
            try:
                Fx = F(x_nm)
                if np.all(np.isfinite(Fx)):
                    norm_Fx = np.linalg.norm(Fx)
                    if norm_Fx < best_norm:
                        best_norm = norm_Fx
                        best_x = x_nm.copy()
                    if norm_Fx <= 1e-8:
                        return x_nm
            except Exception:
                pass
        except Exception:
            pass
    
    # Strategy 2: Broyden polish from best point
    if not time_up():
        try:
            J = _compute_jacobian(F, best_x)
            x, J, converged = _broyden_refine(F, best_x, J, max_iter=30)
            if converged:
                return x
            try:
                Fx = F(x)
                if np.all(np.isfinite(Fx)):
                    norm_Fx = np.linalg.norm(Fx)
                    if norm_Fx < best_norm:
                        best_norm = norm_Fx
                        best_x = x.copy()
            except Exception:
                pass
        except Exception:
            pass
    
    # Strategy 3: SciPy hybrid from best point
    if not time_up():
        try:
            res = root(F, best_x, method='hybr', options={'maxfev': 200})
            if res.success:
                return res.x
            if np.linalg.norm(res.fun) <= 1e-8:
                return res.x
            if np.linalg.norm(res.fun) < best_norm:
                best_norm = np.linalg.norm(res.fun)
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Strategy 4: Try a few random restarts with Nelder-Mead + Broyden
    if not time_up():
        rng = np.random.default_rng(42)
        for attempt in range(3):
            if time_up():
                break
            scale = 0.5 * (attempt + 1)
            perturbation = rng.normal(0, scale, size=2)
            x_try = x0_arr + perturbation
            try:
                x_nm = _nelder_mead_global(F, x_try, deadline, max_iter=50)
                try:
                    Fx = F(x_nm)
                    if np.all(np.isfinite(Fx)):
                        norm_Fx = np.linalg.norm(Fx)
                        if norm_Fx < best_norm:
                            best_norm = norm_Fx
                            best_x = x_nm.copy()
                        if norm_Fx <= 1e-8:
                            return x_nm
                except Exception:
                    pass
                
                # Broyden polish
                if not time_up():
                    J_try = _compute_jacobian(F, best_x)
                    x, _, converged = _broyden_refine(F, best_x, J_try, max_iter=15)
                    if converged:
                        return x
                    try:
                        Fx = F(x)
                        if np.all(np.isfinite(Fx)):
                            norm_Fx = np.linalg.norm(Fx)
                            if norm_Fx < best_norm:
                                best_norm = norm_Fx
                                best_x = x.copy()
                    except Exception:
                        pass
            except Exception:
                continue
    
    return best_x

# EVOLVE-BLOCK-END
