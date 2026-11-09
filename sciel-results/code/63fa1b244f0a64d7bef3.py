"""Root finder for 3 fixed 2D systems using derivative-free quadratic model optimization."""
# EVOLVE-BLOCK-START
import numpy as np


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Start with Newton's method using finite-difference Jacobian (fast local convergence).
    2. If Newton fails or converges slowly, switch to a derivative-free method based on
       minimizing g(x) = ||F(x)||^2 using a quadratic model updated with 3 evaluations per step.
    3. Use aggressive line search with step halving to ensure robustness.
    
    The 2D structure allows us to build a quadratic model from 3 points (vertex + 2 edges),
    giving a full 3-parameter quadratic in 2D (a, b, c, d, e, f) — but we can use the
    structure of g(x) = ||F(x)||^2 to our advantage.
    
    Actually, let's use a simpler and more robust approach:
    - Phase 1: Newton with FD Jacobian for up to 30 iterations.
    - Phase 2: If not converged, use Nelder-Mead or a custom derivative-free method.
    - We'll implement a custom 2D derivative-free method based on quadratic interpolation.
    """
    x = np.array(x0, dtype=float)
    
    # Phase 1: Newton's method with finite-difference Jacobian
    max_newton = 40
    for i in range(max_newton):
        Fx = F(x)
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx <= 1e-12:
            return x
        
        # Compute Jacobian via central differences for better accuracy
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
        
        # Check if step is reasonable
        if not np.all(np.isfinite(dx)):
            break
        
        # Damped Newton: try full step, then halve
        alpha = 1.0
        for _ in range(20):
            x_new = x - alpha * dx
            F_new = F(x_new)
            if np.linalg.norm(F_new) < norm_Fx:
                x = x_new
                break
            alpha *= 0.5
        else:
            # Step too small or no improvement, break to phase 2
            break
    
    # Check if we converged
    Fx = F(x)
    if np.linalg.norm(Fx) <= 1e-8:
        return x
    
    # Phase 2: Derivative-free optimization of g(x) = ||F(x)||^2
    # Use a 2D quadratic model approach
    # We maintain 3 points that form a triangle, fit a quadratic, and move to the minimizer
    
    def g(x_val):
        Fv = F(x_val)
        return np.dot(Fv, Fv)
    
    # Initialize 3 points around current x
    h = max(1e-3, 0.1 * np.linalg.norm(x))
    p0 = x.copy()
    p1 = x + np.array([h, 0.0])
    p2 = x + np.array([0.0, h])
    
    g0 = g(p0)
    g1 = g(p1)
    g2 = g(p2)
    
    best_x = p0
    best_g = g0
    if g1 < best_g:
        best_x, best_g = p1, g1
    if g2 < best_g:
        best_x, best_g = p2, g2
    
    # Iterate quadratic model
    max_iter = 200
    for it in range(max_iter):
        # Fit quadratic q(x) = a*x0^2 + b*x0*x1 + c*x1^2 + d*x0 + e*x1 + f
        # Using 3 points
        # We have 3 equations for 6 unknowns, so we need more constraints or use a simpler model
        # Let's use a simpler approach: assume the quadratic is of the form
        # q(x) = g0 + grad^T (x - p0) + 0.5 * (x - p0)^T H (x - p0)
        # But with only 3 points, we can't determine both grad and H fully.
        
        # Alternative: use the 3 points to determine a quadratic that interpolates them,
        # assuming the quadratic is of the form:
        # q(x0, x1) = c0 + c1*x0 + c2*x1 + c3*x0^2 + c4*x1^2 + c5*x0*x1
        # With 3 points, we have 3 equations, 6 unknowns. Underdetermined.
        
        # Let's use a different approach: Nelder-Mead simplex method
        # This is robust and derivative-free.
        
        # Actually, let's implement a simple but effective 2D derivative-free method:
        # Powell's method or just a well-tuned Nelder-Mead.
        
        # Let's switch to Nelder-Mead
        break
    
    # Nelder-Mead implementation
    # Initialize simplex
    h = max(1e-3, 0.1 * np.linalg.norm(x))
    if h < 1e-4:
        h = 1e-4
    
    simplex = np.array([
        x,
        x + np.array([h, 0.0]),
        x + np.array([0.0, h])
    ])
    
    fvals = np.array([g(simplex[i]) for i in range(3)])
    
    # Sort simplex
    order = np.argsort(fvals)
    simplex = simplex[order]
    fvals = fvals[order]
    
    best_x = simplex[0]
    best_g = fvals[0]
    
    alpha = 1.0  # reflection
    gamma = 2.0  # expansion
    rho = 0.5   # contraction
    sigma = 0.5 # shrink
    
    max_nm = 500
    for it in range(max_nm):
        # Check convergence
        if best_g <= 1e-16:  # g = ||F||^2, so ||F|| <= 1e-8 means g <= 1e-16
            return best_x
        
        # Centroid of best 2 points
        centroid = np.mean(simplex[:2], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[2])
        fr = g(xr)
        
        if fr < fvals[0]:
            # Try expansion
            xe = centroid + gamma * (centroid - simplex[2])
            fe = g(xe)
            if fe < fr:
                simplex[2] = xe
                fvals[2] = fe
            else:
                simplex[2] = xr
                fvals[2] = fr
        elif fr < fvals[1]:
            simplex[2] = xr
            fvals[2] = fr
        else:
            # Contraction
            if fr < fvals[2]:
                # Outside contraction
                xc = centroid + rho * (xr - centroid)
            else:
                # Inside contraction
                xc = centroid - rho * (simplex[2] - centroid)
            fc = g(xc)
            
            if fc < fvals[2]:
                simplex[2] = xc
                fvals[2] = fc
            else:
                # Shrink
                for i in range(1, 3):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = g(simplex[i])
        
        # Re-sort
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]
        
        if fvals[0] < best_g:
            best_g = fvals[0]
            best_x = simplex[0].copy()
        
        # Check for convergence
        if np.max(fvals) - np.min(fvals) < 1e-14:
            break
    
    return best_x
# EVOLVE-BLOCK-END
