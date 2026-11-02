import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0

# Pre-compute triplet indices for the 165 triangles
# We need indices (i, j, k) with i < j < k
TRIPLET_INDICES = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            TRIPLET_INDICES.append((i, j, k))
TRIPLET_INDICES = np.array(TRIPLET_INDICES) # shape (165, 3)

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2)
                area = 0.5 * cross
                if area < min_area:
                    min_area = area
    return min_area

def _check_inside(pts, eps=1e-9):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    for p in pts:
        x, y = p
        if y < -eps:
            return False
        if y > np.sqrt(3) * x + eps:
            return False
        if y > np.sqrt(3) * (1 - x) + eps:
            return False
        if x < -eps or x > 1 + eps:
            return False
    return True

def _random_point_in_triangle(rng):
    """Generate a random point inside the equilateral triangle."""
    u = rng.random()
    v = rng.random()
    while u + v > 1:
        u = rng.random()
        v = rng.random()
    x = u + 0.5 * v
    y = S3H * v
    return [x, y]

def _construct_slsqp_objective_and_jacobian():
    """
    Returns the objective function and its gradient for SLSQP.
    Variables: x_flat (22 coords) + t (1 scalar) = 23 variables.
    Objective: minimize -t (to maximize t).
    Constraints: 
      1. Area(i,j,k) - t >= 0 for all 165 triplets.
      2. Barycentric coordinates >= 0 (3 constraints per point, 33 total).
    """
    
    # Pre-compute coefficients for area calculation
    # Area = 0.5 * | (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i) |
    # Let's expand to linear terms in x, y for gradient calculation.
    # Area * 2 = x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)
    # This is a linear function of the coordinates for a fixed triplet? 
    # No, wait. The area is 0.5 * |cross product|.
    # cross = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
    # cross = x_j y_k - x_j y_i - x_i y_k + x_i y_i - y_j x_k + y_j x_i + y_i x_k - y_i x_i
    # cross = x_j y_k - x_j y_i - x_i y_k - y_j x_k + y_j x_i + y_i x_k
    # This is bilinear.
    
    # For SLSQP, we need constraints g(x) >= 0.
    # The area is non-negative by definition if we take absolute value, but the sign of the cross product depends on ordering.
    # Since we just want the magnitude, we can use:
    # 0.5 * (cross^2) - t >= 0 ? No, that's quadratic and non-convex.
    # Better: 0.5 * |cross| - t >= 0.
    # SLSQP handles non-smoothness poorly. 
    # However, for a fixed configuration, the sign of the cross product for a specific triplet is constant unless points cross.
    # We can enforce that the signed area (with a fixed orientation) is >= t.
    # But we don't know the orientation a priori.
    # Alternative: Use the squared area constraint? 
    # sqrt(0.25 * cross^2) >= t  => 0.25 * cross^2 >= t^2
    # This is smooth and differentiable (mostly).
    # Let's use the constraint: 0.25 * cross^2 - t^2 >= 0.
    # Wait, maximizing t subject to min_area >= t is equivalent to maximizing min_area.
    # If we use squared areas, we are maximizing sqrt(min_area_sq).
    # This is monotonic, so it's equivalent.
    
    # Let's stick to the linear-ish structure if possible, or just use the smooth square.
    # Constraint: 0.25 * cross^2 - t^2 >= 0
    
    def objective(x):
        # x: [x_0, y_0, ..., x_10, y_10, t]
        # Maximize t => Minimize -t
        return -x[-1]
    
    def objective_grad(x):
        grad = np.zeros(23)
        grad[-1] = -1.0
        return grad
        
    # Constraints for SLSQP: type 'ineq' means fun(x) >= 0
    def constraints(x):
        pts = x[:22].reshape(N, 2)
        t = x[-1]
        cons = []
        
        # 1. Area constraints
        # We need to be careful with the sign of the cross product.
        # If we use cross^2, it's always positive.
        # Constraint: 0.25 * cross^2 - t^2 >= 0
        for idx in TRIPLET_INDICES:
            i, j, k = idx
            pi = pts[i]
            pj = pts[j]
            pk = pts[k]
            
            dx1 = pj[0] - pi[0]
            dy1 = pj[1] - pi[1]
            dx2 = pk[0] - pi[0]
            dy2 = pk[1] - pi[1]
            cross = dx1 * dy2 - dy1 * dx2
            area_sq = 0.25 * cross * cross
            
            # Constraint: area_sq - t^2 >= 0
            cons.append(area_sq - t * t)
            
        # 2. Barycentric / Triangle constraints
        # Points must be inside (0,0), (1,0), (0.5, sqrt(3)/2)
        # Inequalities:
        # y >= 0
        # y <= sqrt(3) * x
        # y <= sqrt(3) * (1 - x)
        
        for i in range(N):
            xi, yi = pts[i]
            cons.append(yi) # y >= 0
            cons.append(np.sqrt(3) * xi - yi) # y <= sqrt(3)x
            cons.append(np.sqrt(3) * (1 - xi) - yi) # y <= sqrt(3)(1-x)
            
        return cons

    def constraints_jac(x):
        pts = x[:22].reshape(N, 2)
        t = x[-1]
        # Jacobian shape: (num_constraints, 23)
        # num_constraints = 165 + 3*11 = 198
        jac = np.zeros((198, 23))
        
        # 1. Area constraints Jacobian
        # f = 0.25 * cross^2 - t^2
        # df/dt = -2t
        # df/dx_i = 0.25 * 2 * cross * (d cross / d x_i) = 0.5 * cross * (d cross / d x_i)
        
        for c_idx, idx in enumerate(TRIPLET_INDICES):
            i, j, k = idx
            pi = pts[i]
            pj = pts[j]
            pk = pts[k]
            
            dx1 = pj[0] - pi[0]
            dy1 = pj[1] - pi[1]
            dx2 = pk[0] - pi[0]
            dy2 = pk[1] - pi[1]
            cross = dx1 * dy2 - dy1 * dx2
            
            # Derivatives of cross with respect to coordinates
            # cross = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
            # d cross / d x_i = -(y_k - y_i) + (y_j - y_i) = y_j - y_k
            # d cross / d y_i = -(x_k - x_i) + (x_j - x_i) = x_j - x_k
            # d cross / d x_j = (y_k - y_i)
            # d cross / d y_j = -(x_k - x_i)
            # d cross / d x_k = -(y_j - y_i)
            # d cross / d y_k = (x_j - x_i)
            
            d_cross_dx_i = pj[1] - pk[1]
            d_cross_dy_i = pj[0] - pk[0]
            d_cross_dx_j = pk[1] - pi[1]
            d_cross_dy_j = -(pk[0] - pi[0])
            d_cross_dx_k = -(pj[1] - pi[1])
            d_cross_dy_k = pj[0] - pi[0]
            
            factor = 0.5 * cross
            
            jac[c_idx, 2*i] = factor * d_cross_dx_i
            jac[c_idx, 2*i+1] = factor * d_cross_dy_i
            jac[c_idx, 2*j] = factor * d_cross_dx_j
            jac[c_idx, 2*j+1] = factor * d_cross_dy_j
            jac[c_idx, 2*k] = factor * d_cross_dx_k
            jac[c_idx, 2*k+1] = factor * d_cross_dy_k
            
            jac[c_idx, 22] = -2 * t # d/dt of -t^2
            
        # 2. Barycentric constraints Jacobian
        # y >= 0 -> dy/dy_i = 1
        # sqrt(3)x - y >= 0 -> d/dx_i = sqrt(3), d/dy_i = -1
        # sqrt(3)(1-x) - y >= 0 -> d/dx_i = -sqrt(3), d/dy_i = -1
        
        for i in range(N):
            c_base = 165 + 3*i
            
            # y >= 0
            jac[c_base, 2*i+1] = 1.0
            
            # sqrt(3)x - y >= 0
            jac[c_base+1, 2*i] = np.sqrt(3)
            jac[c_base+1, 2*i+1] = -1.0
            
            # sqrt(3)(1-x) - y >= 0
            jac[c_base+2, 2*i] = -np.sqrt(3)
            jac[c_base+2, 2*i+1] = -1.0
            
        return jac

    # Bounds for SLSQP
    # x in [0, 1], y in [0, sqrt(3)/2], t in [0, 1] (max area is 1/4 for unit triangle? No, area is 1, max triangle area is 1)
    # Actually, the triangle area is 1. The max possible area for a sub-triangle is 1.
    # So t is in [0, 1].
    bounds = []
    for i in range(N):
        bounds.append((0, 1)) # x
        bounds.append((0, S3H)) # y
    bounds.append((0, 1.0)) # t
    
    return objective, objective_grad, constraints, constraints_jac, bounds

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    obj_func, obj_grad, cons_func, cons_jac, bounds = _construct_slsqp_objective_and_jacobian()
    
    best_pts = None
    best_score = 0.0
    
    # Multiple random restarts to handle non-convexity
    # SLSQP is a local solver.
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Generate initial guess
        # Mix of random and structured
        if r == 0:
            # Structured: Centroid + corners
            pts = np.zeros((N, 2))
            pts[0] = [0.5, S3H/3] # Centroid
            pts[1] = [0.1, 0.1]
            pts[2] = [0.9, 0.1]
            pts[3] = [0.5, S3H*0.8]
            for i in range(4, N):
                pts[i] = _random_point_in_triangle(rng)
        else:
            pts = np.array([_random_point_in_triangle(rng) for _ in range(N)])
            
        # Initial t: small positive value
        t_init = 0.01
        x0 = np.concatenate([pts.flatten(), [t_init]])
        
        # Ensure x0 is strictly feasible for barycentric constraints? 
        # SLSQP can handle it, but it's better to project slightly inside.
        for i in range(N):
            x, y = pts[i]
            # Project to [0.01, 0.99] in x, [0.01, S3H-0.01] in y roughly?
            # Just clamp to be safe
            if x < 0.01: x = 0.01
            if x > 0.99: x = 0.99
            if y < 0.01: y = 0.01
            max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
            if y > max_y - 0.01: y = max_y - 0.01
            pts[i] = [x, y]
        x0[:22] = pts.flatten()
        
        try:
            result = minimize(
                obj_func,
                x0,
                jac=obj_grad,
                method='SLSQP',
                bounds=bounds,
                constraints=[{'type': 'ineq', 'fun': cons_func, 'jac': cons_jac}],
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            
            if result.success:
                pts_res = result.x[:22].reshape(N, 2)
                t_res = result.x[-1]
                
                # Verify validity
                if _check_inside(pts_res):
                    # Calculate actual min area to be sure
                    actual_score = _min_triangle_area(pts_res)
                    if actual_score > best_score:
                        best_score = actual_score
                        best_pts = pts_res.copy()
                        
        except Exception:
            continue
            
    # If SLSQP failed or didn't find anything good, fallback to a simple greedy/random search
    if best_pts is None or best_score < 0.01:
        # Fallback: Random search with local improvement
        for _ in range(100):
            if time.time() - start_time > DEADLINE - 5:
                break
            pts = np.array([_random_point_in_triangle(rng) for _ in range(N)])
            score = _min_triangle_area(pts)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
                
    if best_pts is None:
        # Absolute fallback
        best_pts = np.array([_random_point_in_triangle(rng) for _ in range(N)])
        
    return best_pts

# Main entry point
if __name__ == "__main__":
    construct()
