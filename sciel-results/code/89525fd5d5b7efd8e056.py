# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place n squares with pairwise disjoint
    interiors inside the unit square [0,1]^2 so as to MAXIMIZE the sum of their
    side lengths. Here n = 10.

    Uses a continuous repulsion potential model where each square's side length
    is derived from local minimum distances to other squares and boundaries.
    This creates a smooth differentiable energy landscape for optimization.

    Returns:
        centers: np.ndarray of shape (10, 2) with the (x, y) center of each square.
        sides:   np.ndarray of shape (10,) with the side length of each square.
    The i-th axis-aligned square occupies
        [cx - s/2, cx + s/2] x [cy - s/2, cy + s/2].
    """
    n = 10
    tol = 1e-7
    softplus_eps = 1e-8

    def softplus(x):
        """Numerically stable softplus for smooth max(0, x)."""
        return np.where(x > 20, x, np.log1p(np.exp(x)))

    def compute_repulsion_energy(centers, sides, beta=50.0):
        """
        Compute smooth repulsion energy between all pairs of squares.
        Uses exponential repulsion potential that is smooth and differentiable.
        
        The potential encourages squares to maintain separation proportional
        to their sizes, creating a smooth landscape for the optimizer.
        """
        energy = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                min_sep = (sides[i] + sides[j]) / 2.0
                
                # Smooth repulsion: penalize when separation < min_sep
                # Using exponential kernel for smoothness
                px = softplus(beta * (min_sep - abs(dx)))
                py = softplus(beta * (min_sep - abs(dy)))
                energy += px * py
        
        # Boundary repulsion
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            # Distance to each boundary
            d_left = cx - s / 2.0
            d_right = 1.0 - cx - s / 2.0
            d_bottom = cy - s / 2.0
            d_top = 1.0 - cy - s / 2.0
            
            # Penalize if any boundary is violated
            energy += softplus(beta * (-d_left))
            energy += softplus(beta * (-d_right))
            energy += softplus(beta * (-d_bottom))
            energy += softplus(beta * (-d_top))
        
        return energy

    def objective(x):
        """
        Objective: maximize sum of sides minus repulsion energy.
        We minimize the negative of this.
        """
        centers = x[:2*n].reshape(n, 2)
        sides = x[2*n:]
        
        # Ensure sides are positive with smooth constraint
        sides_safe = np.maximum(sides, 1e-10)
        
        # Objective: sum of sides
        obj = np.sum(sides_safe)
        
        # Repulsion energy (penalty for overlaps and boundary violations)
        energy = compute_repulsion_energy(centers, sides_safe, beta=50.0)
        
        # Weighted combination: we want to maximize obj while keeping energy low
        weight = 100.0
        return -(obj - weight * energy)

    def gradient(x):
        """
        Analytical gradient of the objective function.
        """
        centers = x[:2*n].reshape(n, 2)
        sides = x[2*n:]
        sides_safe = np.maximum(sides, 1e-10)
        
        grad = np.zeros_like(x)
        beta = 50.0
        weight = 100.0
        
        # Gradient of sum(sides)
        for i in range(n):
            if sides[i] > 1e-10:
                grad[2*n + i] = 1.0
        
        # Gradient of repulsion energy
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                min_sep = (sides_safe[i] + sides_safe[j]) / 2.0
                
                # Derivative of softplus(beta * (min_sep - |dx|))
                sig_x = 1.0 / (1.0 + np.exp(-beta * (min_sep - abs(dx))))
                sig_y = 1.0 / (1.0 + np.exp(-beta * (min_sep - abs(dy))))
                
                # d/dx of softplus(beta*(min_sep - |dx|)) = sig * (-beta * sign(dx))
                if dx >= 0:
                    dpx_dx = -beta * sig_x
                else:
                    dpx_dx = beta * sig_x
                
                if dy >= 0:
                    dpy_dy = -beta * sig_y
                else:
                    dpy_dy = beta * sig_y
                
                # Product rule for px * py
                # d(px*py)/dx_i = (dpx/dx_i) * py
                # d(px*py)/dx_j = (dpx/dx_j) * py = (-dpx/dx_i) * py
                # d(px*py)/dy_i = px * (dpy/dy_i)
                # d(px*py)/dy_j = px * (dpy/dy_j) = px * (-dpy/dy_i)
                
                py_val = softplus(beta * (min_sep - abs(dy)))
                px_val = softplus(beta * (min_sep - abs(dx)))
                
                grad[i*2 + 0] += weight * dpx_dx * py_val
                grad[j*2 + 0] += weight * (-dpx_dx) * py_val
                grad[i*2 + 1] += weight * px_val * dpy_dy
                grad[j*2 + 1] += weight * px_val * (-dpy_dy)
                
                # Gradient with respect to sides
                # d(min_sep)/d(s_i) = 0.5
                # d(px*py)/d(s_i) = 0.5 * (dpx/d(min_sep) * py + px * dpy/d(min_sep))
                # dpx/d(min_sep) = beta * sig_x
                # dpy/d(min_sep) = beta * sig_y
                
                dpx_dsep = beta * sig_x
                dpy_dsep = beta * sig_y
                
                if sides[i] > 1e-10:
                    grad[2*n + i] += weight * 0.5 * (dpx_dsep * py_val + px_val * dpy_dsep)
                if sides[j] > 1e-10:
                    grad[2*n + j] += weight * 0.5 * (dpx_dsep * py_val + px_val * dpy_dsep)
            
            # Boundary gradients
            cx, cy = centers[i]
            s = sides_safe[i]
            
            d_left = cx - s / 2.0
            d_right = 1.0 - cx - s / 2.0
            d_bottom = cy - s / 2.0
            d_top = 1.0 - cy - s / 2.0
            
            sig_l = 1.0 / (1.0 + np.exp(-beta * (-d_left)))
            sig_r = 1.0 / (1.0 + np.exp(-beta * (-d_right)))
            sig_b = 1.0 / (1.0 + np.exp(-beta * (-d_bottom)))
            sig_t = 1.0 / (1.0 + np.exp(-beta * (-d_top)))
            
            # d(softplus(beta*(-d_left)))/d(cx) = beta * sig_l * (-1)
            grad[i*2 + 0] += weight * (-beta * sig_l)  # from left
            grad[i*2 + 0] += weight * (beta * sig_r)   # from right
            grad[i*2 + 1] += weight * (-beta * sig_b)  # from bottom
            grad[i*2 + 1] += weight * (beta * sig_t)   # from top
            
            # d(softplus(beta*(-d_left)))/d(s) = beta * sig_l * (-0.5)
            if s > 1e-10:
                grad[2*n + i] += weight * (-0.5 * beta * sig_l)
                grad[2*n + i] += weight * (-0.5 * beta * sig_r)
                grad[2*n + i] += weight * (-0.5 * beta * sig_b)
                grad[2*n + i] += weight * (-0.5 * beta * sig_t)
        
        return grad

    # Initial guess: symmetric grid packing
    k = int(np.ceil(np.sqrt(n)))
    side_init = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx_init = (cols + 0.5) * side_init
    cy_init = (rows + 0.5) * side_init
    centers_init = np.stack([cx_init, cy_init], axis=1).flatten()
    sides_init = np.full(n, side_init)
    x0 = np.concatenate([centers_init, sides_init])
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0, 1))  # cx_i
        bounds.append((0, 1))  # cy_i
        bounds.append((1e-10, 1))  # s_i
    
    # Optimize with multiple restarts from different initial configurations
    best_x = None
    best_obj = np.inf
    
    # Try several different initial configurations
    initial_configs = []
    
    # Config 1: Grid
    initial_configs.append(x0.copy())
    
    # Config 2: Slightly perturbed grid
    np.random.seed(42)
    x0_perturbed = x0.copy()
    x0_perturbed[:2*n] += np.random.randn(2*n) * 0.01
    initial_configs.append(x0_perturbed)
    
    # Config 3: Different grid arrangement (2x5 vs 3x3+1)
    cx3 = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.1, 0.3, 0.5, 0.7, 0.9])
    cy3 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9])
    sides3 = np.full(n, 0.18)
    x0_3 = np.concatenate([np.stack([cx3, cy3], axis=1).flatten(), sides3])
    initial_configs.append(x0_3)
    
    # Config 4: Corner and center arrangement
    cx4 = np.array([0.15, 0.85, 0.15, 0.85, 0.5, 0.15, 0.85, 0.15, 0.85, 0.5])
    cy4 = np.array([0.15, 0.15, 0.85, 0.85, 0.5, 0.15, 0.15, 0.85, 0.85, 0.5])
    sides4 = np.array([0.25, 0.25, 0.25, 0.25, 0.35, 0.25, 0.25, 0.25, 0.25, 0.35])
    x0_4 = np.concatenate([np.stack([cx4, cy4], axis=1).flatten(), sides4])
    initial_configs.append(x0_4)
    
    for x_init in initial_configs:
        # Clip to bounds
        x_init = np.clip(x_init, [b[0] for b in bounds], [b[1] for b in bounds])
        
        try:
            result = minimize(objective, x_init, jac=gradient, method='L-BFGS-B', 
                             bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10})
            
            obj_val = objective(result.x)
            if obj_val < best_obj:
                best_obj = obj_val
                best_x = result.x
        except Exception:
            continue
    
    # If optimization failed, use the best initial config
    if best_x is None:
        best_x = x0
    
    x_opt = best_x
    centers = x_opt[:2*n].reshape(n, 2)
    sides = x_opt[2*n:]
    sides = np.maximum(sides, 1e-10)
    
    # Verify and fix any remaining issues
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        k = int(np.ceil(np.sqrt(n)))
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
    
    # Final validation: ensure no overlaps and within bounds
    # If there are violations, shrink sides slightly
    for iter_fix in range(10):
        valid = True
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1 + tol or cy - s/2 < -tol or cy + s/2 > 1 + tol:
                valid = False
                break
            for j in range(i + 1, n):
                dx = abs(centers[i][0] - centers[j][0])
                dy = abs(centers[i][1] - centers[j][1])
                t = (sides[i] + sides[j])/2 - tol
                if dx < t and dy < t:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            break
        
        # Shrink all sides slightly
        sides *= 0.999
    
    return centers.astype(np.float64), sides.astype(np.float64)


# EVOLVE-BLOCK-END
