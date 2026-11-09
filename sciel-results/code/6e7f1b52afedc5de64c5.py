"""Root finder for 3 fixed 2D systems using derivative-free quadratic model optimization."""
# EVOLVE-BLOCK-START
import numpy as np


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    Phase 1: Newton's method with finite-difference Jacobian for fast local convergence.
    Phase 2: Derivative-free quadratic model method exploiting 2D structure.
            Maintains 3 points, fits a reduced quadratic model, and uses aggressive
            line search along the model's steepest descent direction.
    """
    x = np.array(x0, dtype=float)
    
    def g(x_val):
        Fv = F(x_val)
        return np.dot(Fv, Fv)
    
    # Phase 1: Newton's method with finite-difference Jacobian
    max_newton = 30
    for i in range(max_newton):
        Fx = F(x)
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx <= 1e-12:
            return x
        
        # Compute Jacobian via central differences
        eps = 1e-7
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = eps
            J[:, j] = (F(x + e) - F(x - e)) / (2 * eps)
        
        try:
            dx = np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            break
        
        if not np.all(np.isfinite(dx)):
            break
        
        # Damped Newton
        alpha = 1.0
        improved = False
        for _ in range(20):
            x_new = x - alpha * dx
            F_new = F(x_new)
            if np.linalg.norm(F_new) < norm_Fx:
                x = x_new
                improved = True
                break
            alpha *= 0.5
        if not improved:
            break
    
    Fx = F(x)
    if np.linalg.norm(Fx) <= 1e-8:
        return x
    
    # Phase 2: Quadratic model with 3 evals per iteration
    # Maintain 3 points. Fit a quadratic of the form:
    #   q(t) = a*t^2 + b*t + c  along the direction from best to worst point
    # Use this to determine a step, then evaluate at the new point.
    # This gives exactly 1 new evaluation per iteration (plus we reuse old values).
    # To get 3 evals/iter, we can also probe two additional directions.
    
    # Initialize simplex: best point and two offset points
    h = max(1e-3, 0.1 * np.linalg.norm(x))
    if h < 1e-4:
        h = 1e-4
    
    p0 = x.copy()
    p1 = x + np.array([h, 0.0])
    p2 = x + np.array([0.0, h])
    
    g0 = g(p0)
    g1 = g(p1)
    g2 = g(p2)
    
    # Sort so p0 is best
    pts = [p0, p1, p2]
    gts = [g0, g1, g2]
    order = np.argsort(gts)
    pts = [pts[i] for i in order]
    gts = [gts[i] for i in order]
    
    best_x = pts[0].copy()
    best_g = gts[0]
    
    max_iter = 300
    for it in range(max_iter):
        if best_g <= 1e-16:
            return best_x
        
        # Fit quadratic along direction from worst to best point
        # Direction: d = p0 - p2 (from worst to best)
        d = pts[0] - pts[2]
        norm_d = np.linalg.norm(d)
        if norm_d < 1e-15:
            # Points too close, shrink
            pts[1] = pts[0] + 0.5 * (pts[1] - pts[0])
            pts[2] = pts[0] + 0.5 * (pts[2] - pts[0])
            gts[1] = g(pts[1])
            gts[2] = g(pts[2])
            order = np.argsort(gts)
            pts = [pts[i] for i in order]
            gts = [gts[i] for i in order]
            continue
        
        # Parametrize: x(t) = pts[2] + t * d, t in [0, 1] gives pts[2] to pts[0]
        # We have:
        #   t=0: g = gts[2]
        #   t=1: g = gts[0]
        #   Need a third point: use pts[1] projected onto this line
        # Project pts[1] onto the line: t1 = dot(pts[1]-pts[2], d) / dot(d,d)
        v = pts[1] - pts[2]
        t1 = np.dot(v, d) / (norm_d ** 2)
        
        # If t1 is close to 0 or 1, use a different approach
        if t1 < 0.1 or t1 > 0.9:
            # Use the midpoint as third point
            t1 = 0.5
            x_mid = pts[2] + t1 * d
            g_mid = g(x_mid)
        else:
            g_mid = gts[1]
        
        # Fit quadratic q(t) = a*t^2 + b*t + c through (0, gts[2]), (1, gts[0]), (t1, g_mid)
        # c = gts[2]
        c = gts[2]
        # a + b + c = gts[0]
        # a*t1^2 + b*t1 + c = g_mid
        # Solve:
        #   a + b = gts[0] - c
        #   a*t1^2 + b*t1 = g_mid - c
        A = np.array([[1.0, 1.0], [t1**2, t1]])
        b_vec = np.array([gts[0] - c, g_mid - c])
        try:
            ab = np.linalg.solve(A, b_vec)
            a_q, b_q = ab
        except np.linalg.LinAlgError:
            a_q, b_q = 0.0, (gts[0] - c)
        
        # Find minimizer of q(t): t* = -b_q / (2*a_q) if a_q > 0
        if a_q > 1e-15:
            t_star = -b_q / (2.0 * a_q)
        else:
            # Linear: minimum at endpoint
            t_star = 0.0 if b_q > 0 else 1.0
        
        # Clamp t_star to [0, 1] with some margin
        t_star = max(0.05, min(0.95, t_star))
        
        # New point
        x_new = pts[2] + t_star * d
        g_new = g(x_new)
        
        # Replace the worst point
        pts[2] = x_new
        gts[2] = g_new
        
        # Re-sort
        order = np.argsort(gts)
        pts = [pts[i] for i in order]
        gts = [gts[i] for i in order]
        
        if gts[0] < best_g:
            best_g = gts[0]
            best_x = pts[0].copy()
        
        # Check convergence
        if np.max(gts) - np.min(gts) < 1e-14:
            # Shrink the simplex
            for i in range(1, 3):
                pts[i] = pts[0] + 0.5 * (pts[i] - pts[0])
                gts[i] = g(pts[i])
                order = np.argsort(gts)
                pts = [pts[i] for i in order]
                gts = [gts[i] for i in order]
    
    return best_x
# EVOLVE-BLOCK-END
