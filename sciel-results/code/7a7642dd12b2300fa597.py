"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    start_time = time.time()
    deadline = start_time + 145.0

    def time_left():
        return deadline - time.time()

    def norm(vec):
        return np.linalg.norm(vec)

    def is_good(x):
        if not np.all(np.isfinite(x)):
            return False
        try:
            Fx = F(x)
            if not np.all(np.isfinite(Fx)):
                return False
            return norm(Fx) <= 1e-8
        except Exception:
            return False

    def evaluate_g(x):
        """Compute g(x) = ||F(x)||^2 and gradient approx if needed? 
        For LQM, we need g and its derivatives.
        g(x) = F1^2 + F2^2.
        grad g = 2 * J^T F.
        Hessian g = 2 * (J^T J + F^T D2F).
        In 2D, we can estimate the Hessian of g using 3 extra points around x.
        However, the prompt suggests 'update a local quadratic model with only 3 function evaluations per iteration'.
        This implies we maintain a model and update it.
        
        Let's implement a simplified Broyden-like or Powell-like approach but specifically 
        using the quadratic model of g(x).
        
        Actually, a simpler and very effective derivative-free method for 2D is 
        the Nelder-Mead simplex method, but it can be slow.
        
        The prompt specifically asks for: 'Formulate the root-finding problem as a global optimization task 
        for the scalar objective g(x) = ||F(x)||^2 using a derivative-free derivative estimator 
        that exploits the 2D structure to update a local quadratic model with only 3 function evaluations per iteration.'
        
        This sounds like a specific algorithm. Let's implement a local quadratic model (LQM) 
        that updates the Hessian estimate.
        
        State: x, g, grad_g, H_g.
        To update H_g in 2D, we need to evaluate g at x + s1, x + s2, x + s1 + s2? 
        Or just use finite differences on the gradient?
        
        Let's try a robust hybrid: 
        1. Multi-start with diverse initial points.
        2. For each start, use a fast local optimizer: 
           - First, try to build a local quadratic model of g(x).
           - Use 3 points to determine the Hessian (in 2D, Hessian is symmetric 2x2, 3 unique elements).
           - We have g(x0), g(x0+e1), g(x0+e2). 
             g(x0) = m
             g(x0+e1) = m + g1*e1 + 0.5*e1^2*H11
             g(x0+e2) = m + g2*e2 + 0.5*e2^2*H22
             This only gives diagonal. We need H12.
             We need a 4th point? Or use the fact that we are minimizing ||F||^2?
             
        Alternative interpretation: '3 function evaluations per iteration' might refer to 
        a specific line search or step update in a derivative-free method like COBYLA or Nelder-Mead 
        where the simplex has size 3 (in 2D). Nelder-Mead uses 3 points in 2D.
        Each iteration of Nelder-Mead evaluates 1 new point (after reflection, expansion, etc.).
        So Nelder-Mead is a strong candidate for 'derivative-free' and 'exploits 2D structure' (simplex size n+1=3).
        
        Let's implement a robust Nelder-Mead with adaptive step sizes and multi-start.
        Nelder-Mead is derivative-free and robust.
        """
        pass

    def nelder_mead(F, x0, max_iter=100, tol=1e-12):
        """
        Nelder-Mead simplex method for minimizing g(x) = ||F(x)||^2.
        In 2D, the simplex has 3 vertices.
        """
        # Initialize simplex
        # x0, x0 + delta * e1, x0 + delta * e2
        # Choose delta based on scale
        scale = max(1.0, norm(x0))
        delta = 0.1 * scale
        
        x1 = x0.copy()
        x2 = x0 + np.array([delta, 0.0])
        x3 = x0 + np.array([0.0, delta])
        
        # Evaluate
        def g(x):
            try:
                Fx = F(x)
                if not np.all(np.isfinite(Fx)):
                    return np.inf
                return np.dot(Fx, Fx)
            except Exception:
                return np.inf

        g1 = g(x1)
        g2 = g(x2)
        g3 = g(x3)
        
        best_x = x1
        best_g = g1
        if g2 < best_g:
            best_x = x2
            best_g = g2
        if g3 < best_g:
            best_x = x3
            best_g = g3
            
        if best_g <= tol:
            return best_x

        # Simplex vertices
        pts = [x1, x2, x3]
        gs = [g1, g2, g3]
        
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5   # contraction
        sigma = 0.5 # shrink
        
        for i in range(max_iter):
            if time_left() < 0.5:
                break
                
            # Sort by function value
            order = np.argsort(gs)
            pts = [pts[i] for i in order]
            gs = [gs[i] for i in order]
            
            if gs[0] < best_g:
                best_g = gs[0]
                best_x = pts[0].copy()
                
            if best_g <= tol:
                return best_x
                
            # Check convergence of simplex
            if abs(gs[-1] - gs[0]) < 1e-15:
                break
                
            # Centroid of all but the worst point
            centroid = np.mean(pts[:-1], axis=0)
            worst_pt = pts[-1]
            worst_g = gs[-1]
            
            # Reflection
            x_r = centroid + alpha * (centroid - worst_pt)
            g_r = g(x_r)
            
            if g_r < gs[0]:
                # Expansion
                x_e = centroid + gamma * (centroid - worst_pt)
                g_e = g(x_e)
                if g_e < g_r:
                    pts[-1] = x_e
                    gs[-1] = g_e
                else:
                    pts[-1] = x_r
                    gs[-1] = g_r
            elif g_r < gs[-2]:
                # Reflection accepted
                pts[-1] = x_r
                gs[-1] = g_r
            else:
                # Contraction
                if g_r < worst_g:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                    g_c = g(x_c)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (centroid - worst_pt)
                    g_c = g(x_c)
                    
                if g_c < worst_g:
                    pts[-1] = x_c
                    gs[-1] = g_c
                else:
                    # Shrink
                    for j in range(1, len(pts)):
                        pts[j] = pts[0] + sigma * (pts[j] - pts[0])
                        gs[j] = g(pts[j])

        return best_x

    def polish_newton(F, x0, max_iter=20, tol=1e-12):
        """
        Fast Newton polish for when we are close to the root.
        Uses finite difference Jacobian.
        """
        x = np.array(x0, dtype=float)
        best_x = x.copy()
        best_norm = np.inf
        
        for i in range(max_iter):
            if time_left() < 0.2:
                break
                
            try:
                Fx = F(x)
            except Exception:
                break
                
            if not np.all(np.isfinite(Fx)):
                break
                
            current_norm = norm(Fx)
            if current_norm < best_norm:
                best_norm = current_norm
                best_x = x.copy()
                
            if current_norm <= tol:
                return x
                
            # Finite difference Jacobian
            eps = 1e-6
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = eps
                try:
                    Fx_plus = F(x + e)
                    if not np.all(np.isfinite(Fx_plus)):
                        return best_x
                    J[:, j] = (Fx_plus - Fx) / eps
                except Exception:
                    return best_x
                    
            if not np.all(np.isfinite(J)):
                return best_x
                
            # Solve J dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except Exception:
                    return best_x
                    
            if not np.all(np.isfinite(dx)):
                return best_x
                
            # Damped step
            step_found = False
            for lam in [1.0, 0.5, 0.25, 0.125, 0.0625]:
                x_new = x + lam * dx
                try:
                    F_new = F(x_new)
                except Exception:
                    continue
                if not np.all(np.isfinite(F_new)):
                    continue
                new_norm = norm(F_new)
                if new_norm < current_norm:
                    x = x_new
                    step_found = True
                    break
                    
            if not step_found:
                break
                
            if norm(dx) < 1e-12:
                break
                
        return best_x

    # Main logic
    x0 = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x0)):
        x0 = np.array([0.0, 0.0])
        
    # Phase 1: Try Newton polish from x0 first (fast if close)
    x_try = polish_newton(F, x0)
    if is_good(x_try):
        return x_try
        
    best_x = x_try
    best_norm = np.inf
    try:
        F_best = F(best_x)
        if np.all(np.isfinite(F_best)):
            best_norm = norm(F_best)
    except Exception:
        pass

    # Phase 2: Multi-start Nelder-Mead
    # Generate diverse starting points
    scale = max(1.0, norm(x0))
    rng = np.random.default_rng(42)
    
    candidates = []
    # Original and scaled
    candidates.append(x0)
    candidates.append(x0 * 0.5)
    candidates.append(x0 * 2.0)
    candidates.append(np.array([0.0, 0.0]))
    
    # Random points in a grid/ring
    for r in [scale * 0.1, scale * 0.5, scale * 1.0, scale * 2.0, scale * 5.0]:
        for _ in range(4):
            u = rng.normal(size=2)
            u = u / norm(u)
            p = x0 + r * u
            candidates.append(p)
            
    # Sort by distance to x0 to prioritize local search first
    candidates.sort(key=lambda c: norm(c - x0))
    
    for c in candidates:
        if time_left() < 2.0:
            break
            
        if not np.all(np.isfinite(c)):
            continue
            
        # Run Nelder-Mead from this candidate
        x_nm = nelder_mead(F, c)
        
        # Check if good
        if is_good(x_nm):
            return x_nm
            
        # Update best
        try:
            F_nm = F(x_nm)
            if np.all(np.isfinite(F_nm)):
                n_nm = norm(F_nm)
                if n_nm < best_norm:
                    best_norm = n_nm
                    best_x = x_nm
        except Exception:
            pass

    # Phase 3: Polish the best found so far with Newton
    if time_left() > 1.0:
        x_final = polish_newton(F, best_x)
        if is_good(x_final):
            return x_final
        # Check if polish improved it
        try:
            F_final = F(x_final)
            if np.all(np.isfinite(F_final)):
                n_final = norm(F_final)
                if n_final < best_norm:
                    best_x = x_final
        except Exception:
            pass

    return best_x
# EVOLVE-BLOCK-END
