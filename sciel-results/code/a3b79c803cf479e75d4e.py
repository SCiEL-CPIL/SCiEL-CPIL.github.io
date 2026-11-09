import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize

N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize

def construct():
    # 1. Initialization
    # Use a triangular lattice subset for a good "mesh-like" start
    # This avoids the symmetric minima of a regular polygon which might not be optimal for min-area
    # A triangular lattice has equilateral triangles, which is the ideal local configuration
    
    # Generate a grid of points in a triangular lattice
    # We need 13 points. 
    # Let's pick a 4x4 grid and take the first 13, or a specific pattern.
    # Better: Start with a regular 13-gon but perturb it slightly to break symmetry,
    # OR use a lattice.
    
    # Let's try a lattice approach.
    # Side length 1.
    # Points (i, j) -> (i + 0.5*j, j*sqrt(3)/2)
    # We want them compact.
    
    # Let's just use a regular polygon with a random rotation and slight radial noise.
    # The directive says "avoid symmetric local minima of polygon-based topologies".
    # So, let's initialize with a slightly irregular mesh.
    
    # Random initialization with a seed for reproducibility, but structured.
    rng = np.random.RandomState(42)
    
    # Start with a regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # Add small random perturbation to angles and radii to break symmetry
    angles += rng.uniform(-0.1, 0.1, N)
    radii = 1.0 + rng.uniform(-0.1, 0.1, N)
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    points = np.column_stack((x, y))
    
    # 2. Define the objective function and gradient
    # We want to maximize min(triangle_area)
    # We use soft-min: -1/k * log(sum(exp(-k * A_i)))
    # Maximizing this is equivalent to minimizing -softmin
    
    k_temp = 10.0  # Temperature for soft-min
    
    def get_delaunay_triangles(pts):
        try:
            tri = Delaunay(pts)
            return tri.simplices
        except Exception:
            # If triangulation fails (e.g., collinear points), return None
            return None

    def soft_min_area(pts):
        triangles = get_delaunay_triangles(pts)
        if triangles is None:
            return 0.0, np.zeros_like(pts)
        
        areas = np.zeros(len(triangles))
        # Compute areas
        for i, tri_idx in enumerate(triangles):
            p1 = pts[tri_idx[0]]
            p2 = pts[tri_idx[1]]
            p3 = pts[tri_idx[2]]
            # Signed area
            cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
            areas[i] = 0.5 * abs(cross)
            
        # Soft min
        # To avoid overflow/underflow, we can subtract max or use log-sum-exp trick
        # log(sum(exp(-k*A))) = -k*max(A) + log(sum(exp(-k*(A - max(A)))))
        # Actually, we want to maximize min(A), so we minimize -softmin.
        # softmin(A) = -1/k * log(sum(exp(-k*A)))
        # -softmin(A) = 1/k * log(sum(exp(-k*A)))
        
        # We want the gradient of -softmin(A)
        # Let S = sum(exp(-k*A_i))
        # f = 1/k * log(S)
        # df/dp = 1/k * (1/S) * dS/dp
        # dS/dp = sum( -k * exp(-k*A_i) * dA_i/dp )
        # df/dp = 1/k * 1/S * -k * sum( exp(-k*A_i) * dA_i/dp )
        # df/dp = - 1/S * sum( w_i * dA_i/dp ) where w_i = exp(-k*A_i)
        
        exp_vals = np.exp(-k_temp * areas)
        S = np.sum(exp_vals)
        
        if S == 0:
            return 0.0, np.zeros_like(pts)
            
        weights = exp_vals / S
        
        grad = np.zeros_like(pts)
        
        for i, tri_idx in enumerate(triangles):
            p1 = pts[tri_idx[0]]
            p2 = pts[tri_idx[1]]
            p3 = pts[tri_idx[2]]
            
            # Gradients of signed area A = 0.5 * cross
            # dA/dp1 = 0.5 * (p3 - p2) rotated 90 deg?
            # cross = (x2-x1)(y3-y1) - (y2-y1)(x3-x1)
            # d/dx1: -(y3-y1) + (y2-y1)*0 - 0 + (y2-y1)*(x3-x1)*0 ... wait.
            # Let's use the standard vector formula.
            # A = 0.5 * ( (p2-p1) x (p3-p1) )
            # dA/dp1 = -0.5 * ( (p2-p1) x (p3-p1) )_vec? No.
            # In 2D, cross(u, v) = u_x v_y - u_y v_x.
            # Let u = p2 - p1, v = p3 - p1.
            # A = 0.5 (u_x v_y - u_y v_x)
            # dA/dp1:
            # du/dp1 = -I, dv/dp1 = -I
            # dA/dp1 = 0.5 ( (-1)*v_y - (-1)*v_x * 0 ... ) -> 0.5 ( -v_y + u_x * 0 )?
            # Let's do it carefully.
            # A = 0.5 ( (x2-x1)(y3-y1) - (y2-y1)(x3-x1) )
            # dA/dx1 = 0.5 ( -1*(y3-y1) - (y2-y1)*(-1) ) = 0.5 ( y1-y3 + y2-y1 ) = 0.5 ( y2 - y3 )
            # dA/dy1 = 0.5 ( (x2-x1)*(-1) - 1*(x3-x1) ) = 0.5 ( x1-x2 - x3 + x1 ) = 0.5 ( 2x1 - x2 - x3 )? 
            # Wait.
            # d/dy1 ( (x2-x1)(y3-y1) ) = (x2-x1)*(-1)
            # d/dy1 ( -(y2-y1)(x3-x1) ) = -1*(x3-x1)
            # Sum = -(x2-x1) - (x3-x1) = -x2 + x1 - x3 + x1 = 2x1 - x2 - x3.
            # This seems correct for the derivative of the signed area.
            
            # However, a more elegant way:
            # grad_p1 = 0.5 * perp(p3 - p2)
            # grad_p2 = 0.5 * perp(p1 - p3)
            # grad_p3 = 0.5 * perp(p2 - p1)
            # where perp(v) = (-v_y, v_x)
            
            v32 = p3 - p2
            v13 = p1 - p3
            v21 = p2 - p1
            
            g1 = 0.5 * np.array([-v32[1], v32[0]])
            g2 = 0.5 * np.array([-v13[1], v13[0]])
            g3 = 0.5 * np.array([-v21[1], v21[0]])
            
            # Note: These are gradients of the *signed* area.
            # We used abs(area) in the objective. 
            # If the triangle is oriented CW, signed area is negative.
            # The soft-min uses A_i > 0.
            # If we use signed area in the gradient, and the triangle flips, the gradient direction flips.
            # Since we are maximizing the minimum area, and we start with a valid mesh,
            # we assume the orientation stays consistent.
            # If we want to be robust, we should multiply the gradient by the sign of the area.
            # sign = 1 if signed_area > 0 else -1.
            # But we computed `areas` using abs().
            # Let's compute signed area to get the sign.
            
            cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
            sign = 1.0 if cross >= 0 else -1.0
            
            # The gradient of |A| is sign(A) * grad(A_signed)
            # So we multiply g1, g2, g3 by sign.
            
            grad[tri_idx[0]] += weights[i] * sign * g1
            grad[tri_idx[1]] += weights[i] * sign * g2
            grad[tri_idx[2]] += weights[i] * sign * g3
            
        # The function value to minimize is -softmin = 1/k * log(S)
        # We return the value and the gradient.
        # Wait, the formula derived was:
        # df/dp = - 1/S * sum( w_i * dA_i/dp )
        # where w_i = exp(-k*A_i)
        # My code above accumulated `weights[i] * grad_A_i`.
        # `weights` here is `exp_vals / S`.
        # So `sum(weights[i] * grad_A_i)` is `1/S * sum(exp(-k*A_i) * grad_A_i)`.
        # The derivative of `1/k * log(S)` is `1/k * (1/S) * dS/dp`.
        # `dS/dp = sum( -k * exp(-k*A_i) * grad_A_i )`.
        # So `df/dp = 1/k * 1/S * -k * sum( exp(-k*A_i) * grad_A_i ) = - sum( (exp(-k*A_i)/S) * grad_A_i )`.
        # So the gradient is `-1 * accumulated_grad`.
        
        grad = -grad
        
        val = (1.0 / k_temp) * np.log(S)
        
        return val, grad

    # 3. Optimization
    # We minimize the negative soft-min area.
    # Initial points
    x0 = points.flatten()
    
    # Use L-BFGS-B
    # We need to handle the case where the triangulation changes.
    # L-BFGS-B uses numerical gradient check if we don't provide analytic, but we provide analytic.
    # If the topology changes, the analytic gradient becomes invalid.
    # We can set `maxiter` and rely on the optimizer to converge.
    # To be safer, we can use a smaller step size or re-compute topology.
    # For now, we trust the analytic gradient within the local topology.
    
    res = minimize(
        lambda x: soft_min_area(x.reshape(N, 2))[0],
        x0,
        jac=lambda x: soft_min_area(x.reshape(N, 2))[1].flatten(),
        method='L-BFGS-B',
        options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-6}
    )
    
    # 4. Final processing
    # The optimizer returns the points that minimize the negative soft-min area,
    # which corresponds to maximizing the soft-min area.
    
    final_points = res.x.reshape(N, 2)
    
    # Optional: Normalize to unit area?
    # The task says "normalized to unit area".
    # The evaluator normalizes by convex hull area.
    # It is safer to return the points as is, or normalized to have area 1.
    # If we normalize, we might change the relative positions if the hull area is not 1.
    # The evaluator does: score = min_tri_area / hull_area.
    # This ratio is scale-invariant.
    # So we can return the points at any scale.
    # However, if the points are very large or small, numerical issues might arise in the evaluator.
    # Let's normalize the final points to have a convex hull area of 1.
    
    try:
        hull = Delaunay(final_points).convex_hull
        hull_pts = final_points[hull]
        # Shoelace formula for area
        x = hull_pts[:, 0]
        y = hull_pts[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        if area > 0:
            scale = 1.0 / np.sqrt(area)
            final_points = final_points * scale
    except Exception:
        pass
        
    return final_points

# EVOLVE-BLOCK-END
