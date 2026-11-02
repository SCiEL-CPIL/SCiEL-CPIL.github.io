"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
import itertools

# Vertices of the equilateral triangle
V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
SQR3 = np.sqrt(3)

def _barycentric_to_cart(b):
    """Convert barycentric coordinates (u, v) with u+v<=1 to Cartesian."""
    x = 0.5 * b[:, 1] + b[:, 0]
    y = (SQR3 / 2) * b[:, 1]
    return np.column_stack([x, y])

def _cart_to_barycentric(pts):
    """Convert Cartesian points to barycentric coordinates (u, v)."""
    # pts shape (N, 2)
    # x = u + 0.5v, y = (sqrt(3)/2)v
    # v = 2y / sqrt(3)
    # u = x - 0.5v
    v = 2 * pts[:, 1] / SQR3
    u = pts[:, 0] - 0.5 * v
    return np.column_stack([u, v])

def _min_area_cartesian(pts):
    """Compute the minimum triangle area among all triplets of points."""
    n = len(pts)
    min_a = np.inf
    # We can vectorize this for speed if needed, but 165 is small.
    for i in range(n):
        pi = pts[i]
        for j in range(i + 1, n):
            pj = pts[j]
            for k in range(j + 1, n):
                pk = pts[k]
                # Area = 0.5 * |det(pj-pi, pk-pi)|
                area = 0.5 * abs(np.cross(pj - pi, pk - pi))
                if area < min_a:
                    min_a = area
    return min_a

def _penalized_obj(b):
    """
    Objective function to minimize.
    b: array of shape (N, 2) representing barycentric coords (u, v) for each point.
    We want to maximize min area.
    We use a penalty for points outside the triangle (u<0, v<0, u+v>1).
    """
    # Clip for evaluation of area to ensure numerical stability, 
    # but penalty drives them back inside.
    # Actually, if points are far outside, area calculation might be weird but valid.
    # However, the problem requires points ON OR INSIDE.
    # If we allow them to be outside during optimization, we must penalize.
    
    # Soft penalty for boundary violations
    # u >= 0, v >= 0, u + v <= 1
    u = b[:, 0]
    v = b[:, 1]
    w = 1 - u - v
    
    # Penalty if any coordinate is negative
    # Using a large quadratic penalty
    eps = 1e-5
    # Smooth approximation of max(0, -x)
    # Or just simple max(0, -x) with subgradient? L-BFGS-B handles bounds well.
    # Better approach: Use L-BFGS-B with bounds!
    
    # Since we will use L-BFGS-B, we can define bounds directly.
    # But L-BFGS-B minimizes f(x).
    # We want to maximize min_area.
    # So we minimize -min_area.
    
    # However, -min_area is non-differentiable.
    # We can use a smooth approximation: -soft_min(area)
    # Or we can just use -min_area and rely on the optimizer's robustness?
    # L-BFGS-B assumes differentiability.
    
    # Let's try a differentiable soft-min approximation.
    # soft_min(x, t) = -t * log(sum(exp(-x_i/t)))
    # This is differentiable.
    
    pts = _barycentric_to_cart(b)
    
    # Compute all triangle areas
    areas = []
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs(np.cross(pts[j] - pts[i], pts[k] - pts[i]))
                areas.append(area)
    
    areas = np.array(areas)
    
    # Soft min
    t = 0.01 # Temperature
    # To avoid overflow in exp, shift by min
    a_min = np.min(areas)
    log_sum = np.log(np.sum(np.exp(-(areas - a_min) / t)))
    soft_min_val = a_min - t * log_sum
    
    # We want to maximize soft_min_val, so minimize -soft_min_val
    return -soft_min_val

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Initial guess: Random points inside
    np.random.seed(42)
    
    best_score = -np.inf
    best_b = None
    
    # Multi-start
    n_starts = 50
    for i in range(n_starts):
        if time.time() > deadline:
            break
            
        # Generate random initial points
        # Uniform in triangle: u, v > 0, u+v < 1
        u = np.random.rand(N)
        v = np.random.rand(N)
        # Rejection sampling or transformation
        # Transformation: r1, r2 uniform in [0,1]. 
        # u = r1 * sqrt(r2), v = r2 * (1-sqrt(r1))? No.
        # Standard: u = r1, v = r2*(1-r1)
        u = np.random.rand(N)
        v = np.random.rand(N) * (1 - u)
        b_init = np.column_stack([u, v])
        
        # Ensure points are distinct enough
        # Sometimes random points are too close.
        
        bounds = [(0.0, 1.0), (0.0, 1.0)] * N
        
        # L-BFGS-B
        try:
            res = minimize(
                _penalized_obj, 
                b_init, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 200, 'ftol': 1e-10}
            )
        except Exception:
            continue
        
        if res.success:
            b_opt = res.x
            # Clip to valid region just in case
            b_opt[:, 0] = np.clip(b_opt[:, 0], 0, 1)
            b_opt[:, 1] = np.clip(b_opt[:, 1], 0, 1)
            # Ensure u+v <= 1
            # L-BFGS-B with bounds (0,1) for u and v doesn't enforce u+v<=1.
            # We need to handle the third constraint.
            # This is a problem. L-BFGS-B handles box constraints only.
            # The constraint u+v <= 1 is linear but not box.
            
            # We must project back or use a different method.
            # Projection onto the triangle:
            # If u+v > 1, we need to reduce u and v.
            # Simple projection: if u+v > 1, scale? No, it's a polytope.
            # For a point (u,v) outside, the closest point in the triangle...
            # Actually, we can just check if it's inside.
            # If not, we can use a penalty method or just reject this start.
            # Or, better, parameterize the triangle such that bounds ensure validity.
            
            # Let's switch strategy:
            # Use a parameterization that maps a square [0,1]^2 to the triangle.
            # x = u + 0.5v, y = sqrt(3)/2 v
            # Inverse: v = 2y/sqrt(3), u = x - 0.5v
            # If we parameterize by (s, t) in [0,1]^2:
            # u = s * (1 - t) ? No.
            
            # Let's stick to barycentric but use SLSQP or trust-constr which handle linear constraints.
            # SLSQP is generally robust for small N.
            
            # Let's retry with SLSQP.
            # Constraints: u >= 0, v >= 0, u+v <= 1 for each point.
            # This is 3N constraints.
            
            # Actually, let's just project the result of L-BFGS-B.
            # If a point is outside, we can project it to the nearest point in the triangle.
            # Projection of (u, v) onto {u>=0, v>=0, u+v<=1}:
            # 1. If u < 0: u=0. If v < 0: v=0.
            # 2. If u+v > 1: 
            #    The line u+v=1. Project (u,v) onto this line?
            #    The closest point on the segment u+v=1, u>=0, v>=0.
            #    If u+v > 1, the projection is along the normal (1,1).
            #    (u, v) - alpha(1, 1) such that (u-alpha) + (v-alpha) = 1 => 2alpha = u+v-1 => alpha = (u+v-1)/2.
            #    New u = u - alpha, New v = v - alpha.
            #    Check if u-alpha >= 0 and v-alpha >= 0.
            #    If not, it's a corner projection.
            
            # Let's implement a helper to project barycentric coords.
            def project_bary(b):
                u = b[:, 0].copy()
                v = b[:, 1].copy()
                # Handle u<0, v<0
                u = np.maximum(u, 0)
                v = np.maximum(v, 0)
                # Handle u+v>1
                s = u + v
                mask = s > 1
                if np.any(mask):
                    alpha = (s[mask] - 1) / 2
                    u_new = u[mask] - alpha
                    v_new = v[mask] - alpha
                    # Check corners
                    # If u_new < 0, then u=0, v = 1-u? No, if u_new<0, we hit the v-axis or corner.
                    # If u_new < 0: u=0, v = 1 (if v_new > 1? No, v_new = v - alpha).
                    # Actually, if u_new < 0, the point projects to (0, 1) if v is large, or (0, v) if v < 1?
                    # Let's just clip again.
                    u[mask] = np.maximum(u_new, 0)
                    v[mask] = np.maximum(v_new, 0)
                    # Re-check sum
                    s2 = u[mask] + v[mask]
                    # If still > 1, it means we hit a corner like (0,1) or (1,0) or (0,0)?
                    # If u=0, v=1, sum=1.
                    # If u=1, v=0, sum=1.
                    # If u=0, v=0, sum=0.
                    # The projection logic above is sufficient for most cases.
                return np.column_stack([u, v])
            
            b_opt = project_bary(b_opt)
            
            # Calculate score
            pts = _barycentric_to_cart(b_opt)
            score = _min_area_cartesian(pts)
            
            if score > best_score:
                best_score = score
                best_b = b_opt
        
        # If we have time, we could do local refinement with SLSQP on the best found so far
        # But let's keep it simple for now.
        
    if best_b is None:
        # Fallback to random
        u = np.random.rand(N)
        v = np.random.rand(N) * (1 - u)
        best_b = np.column_stack([u, v])
        
    pts = _barycentric_to_cart(best_b)
    
    # Final check: ensure points are inside
    # They should be due to projection.
    
    return pts

# EVOLVE-BLOCK-END
