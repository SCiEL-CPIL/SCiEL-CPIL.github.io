import numpy as np
from scipy.optimize import minimize

N = 13

def construct():
    # 1. Initialization
    # Use a random initialization to avoid symmetric local minima.
    # We use a fixed seed for reproducibility.
    rng = np.random.RandomState(42)
    
    # Start with points in a unit disk, scaled to have reasonable separation
    # A regular polygon is a bad start due to symmetry.
    # A random set of points in a disk works well.
    angles = rng.uniform(0, 2 * np.pi, N)
    radii = rng.uniform(0.5, 1.0, N) # Avoid center to ensure convexity and spread
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    points = np.column_stack((x, y))
    
    # Ensure points are in general position (no 3 collinear)
    # The random init is very unlikely to have collinear points.
    
    # Pre-compute all triangle indices
    # There are N*(N-1)*(N-2)/6 triangles
    tri_indices = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                tri_indices.append((i, j, k))
    tri_indices = np.array(tri_indices)
    n_tris = len(tri_indices)
    
    # 2. Objective Function and Gradient
    # We minimize g(P) = (1/k) * log( sum(exp(-k * A_t)) )
    # This approximates -min(A_t) as k -> infinity.
    # Since we want to maximize min(A_t), minimizing g is correct.
    
    k_temp = 50.0 # Sharpness of the soft-min
    
    def objective_and_grad(x_flat):
        pts = x_flat.reshape(N, 2)
        
        # Compute areas for all triangles
        # Vectorized computation
        p1 = pts[tri_indices[:, 0]]
        p2 = pts[tri_indices[:, 1]]
        p3 = pts[tri_indices[:, 2]]
        
        # Signed area * 2 = cross product
        # cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
        cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
                (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
        
        # Unsigned area
        areas = 0.5 * np.abs(cross)
        
        # Avoid log(0) or exp overflow
        # If any area is 0, the objective is infinite (bad).
        # In practice, with random init and L-BFGS, we shouldn't hit exactly 0.
        # But we clip to be safe.
        areas = np.maximum(areas, 1e-9)
        
        # Soft min computation
        # val = (1/k) * log( sum( exp(-k * areas) ) )
        # To prevent overflow in exp, subtract max area from areas?
        # No, we have exp(-k * areas). If areas are small, exp is large.
        # If areas are large, exp is small.
        # The sum is dominated by the smallest areas.
        # Let m = min(areas).
        # sum(exp(-k * A_i)) = exp(-k * m) * sum(exp(-k * (A_i - m)))
        # log(sum) = -k * m + log(sum(exp(-k * (A_i - m))))
        # val = (1/k) * (-k * m + log(sum(exp(-k * (A_i - m)))))
        # val = -m + (1/k) * log(sum(exp(-k * (A_i - m))))
        
        m = np.min(areas)
        shifted_areas = areas - m
        exp_vals = np.exp(-k_temp * shifted_areas)
        S = np.sum(exp_vals)
        
        if S == 0:
            return 0.0, np.zeros_like(x_flat)
            
        val = -m + (1.0 / k_temp) * np.log(S)
        
        # Weights w_i = exp(-k * A_i) / S
        # w_i = exp(-k * (A_i - m)) / S  (since exp(-k*m) cancels in numerator/denominator? No.)
        # S = sum(exp(-k * A_j)) = exp(-k*m) * sum(exp(-k*(A_j-m))) = exp(-k*m) * S_shifted
        # w_i = exp(-k * A_i) / S = exp(-k * A_i) / (exp(-k*m) * S_shifted) = exp(-k * (A_i - m)) / S_shifted
        # So w_i = exp_vals / S_shifted. Note S in code above is S_shifted.
        # Wait, in my code:
        # exp_vals = exp(-k * shifted_areas)
        # S = sum(exp_vals)
        # So S is S_shifted.
        # w_i = exp_vals / S
        
        weights = exp_vals / S
        
        # Gradient of val w.r.t. pts
        # val = -m + (1/k) log(S_shifted)
        # This decomposition is tricky for gradient because m depends on pts.
        # Let's stick to the direct derivative of g(P) = (1/k) log( sum( exp(-k A_t) ) )
        # Let U = sum( exp(-k A_t) )
        # g = (1/k) log(U)
        # dg/dP = (1/k) * (1/U) * dU/dP
        # dU/dP = sum( -k * exp(-k A_t) * dA_t/dP )
        # dg/dP = (1/k) * (1/U) * sum( -k * exp(-k A_t) * dA_t/dP )
        # dg/dP = - sum( (exp(-k A_t) / U) * dA_t/dP )
        # Let w_t = exp(-k A_t) / U.
        # dg/dP = - sum( w_t * dA_t/dP )
        
        # We need to accumulate gradients for each point.
        grad_pts = np.zeros_like(pts)
        
        # dA_t/dP_i for each triangle t and point i in t
        # A_t = 0.5 * | cross |
        # dA_t/dP_i = 0.5 * sign(cross) * d(cross)/dP_i
        
        signs = np.sign(cross)
        # Handle zero cross (collinear) - sign is 0, gradient is 0? 
        # Actually if cross is 0, area is 0, gradient of |x| is undefined.
        # But with k large, these terms dominate. 
        # If we are near collinearity, the gradient pushes away.
        # Using sign(cross) is standard for subgradient of abs.
        
        for idx in range(n_tris):
            i, j, k_idx = tri_indices[idx]
            w = weights[idx]
            s = signs[idx]
            
            if w < 1e-12:
                continue
                
            p1 = pts[i]
            p2 = pts[j]
            p3 = pts[k_idx]
            
            # Gradients of cross product w.r.t. points
            # cross = (x2-x1)(y3-y1) - (y2-y1)(x3-x1)
            # d/dx1: -(y3-y1) + (y2-y1) = y2-y3
            # d/dy1: -(x3-x1) - (x2-x1) = 2x1 - x2 - x3 ? No.
            # Let's use perp vectors.
            # grad_p1 = 0.5 * perp(p3 - p2)
            # grad_p2 = 0.5 * perp(p1 - p3)
            # grad_p3 = 0.5 * perp(p2 - p1)
            # perp(v) = (-v_y, v_x)
            
            v32 = p3 - p2
            v13 = p1 - p3
            v21 = p2 - p1
            
            g1 = 0.5 * np.array([-v32[1], v32[0]])
            g2 = 0.5 * np.array([-v13[1], v13[0]])
            g3 = 0.5 * np.array([-v21[1], v21[0]])
            
            # Multiply by sign of area
            g1 *= s
            g2 *= s
            g3 *= s
            
            # Accumulate
            grad_pts[i] += w * g1
            grad_pts[j] += w * g2
            grad_pts[k_idx] += w * g3
            
        # The total gradient is - sum(w_t * grad_A_t)
        grad_flat = -grad_pts.flatten()
        
        return val, grad_flat

    x0 = points.flatten()
    
    # Optimize
    res = minimize(
        lambda x: objective_and_grad(x)[0],
        x0,
        jac=lambda x: objective_and_grad(x)[1],
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6}
    )
    
    final_points = res.x.reshape(N, 2)
    
    # Normalize to unit area convex hull
    # Compute convex hull area
    # Use scipy.spatial.ConvexHull
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(final_points)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area > 0:
            scale = 1.0 / np.sqrt(hull_area)
            final_points = final_points * scale
    except Exception:
        pass
        
    return final_points
