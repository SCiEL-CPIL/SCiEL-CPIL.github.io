"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import math

N = 11

def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
    angle_rad = math.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        vertices.append([x, y])
    return np.array(vertices)

def point_to_polygon_distance(px, py, vertices):
    """
    Compute the signed distance from point (px, py) to a convex polygon.
    Positive if outside, negative if inside.
    We use a smooth approximation via the minimum over edges of the distance to the line segment,
    but for optimization we can use a simpler approach: compute the minimum distance from the point
    to each edge (as infinite line) and check if the projection falls within the edge.
    
    For smoothness, we'll use a differentiable approximation.
    Actually, for a convex polygon, the distance from an interior point to the boundary is the minimum
    over all edges of the perpendicular distance to the edge line (if the projection is within the edge segment).
    
    A simpler differentiable approach: for each edge, compute the distance from the point to the line
    containing the edge, but only if the point is on the "outside" side. This is tricky.
    
    Alternative: use the fact that for a regular hexagon, we can compute the distance analytically.
    
    Let's use a different approach: for a regular hexagon with center (cx, cy), side s, and angle a,
    the distance from a point to the hexagon boundary can be computed by transforming the point into
    the hexagon's local coordinate system and then computing the distance to the hexagon in that frame.
    
    In the local frame (hexagon axis-aligned), a point (u, v) is inside the hexagon if:
    |u| <= s and |v| <= s * sqrt(3) / 2 and |u| + |v| / sqrt(3) <= s
    
    The distance from an interior point to the boundary is:
    min(s - |u|, s * sqrt(3)/2 - |v|, s - |u| - |v|/sqrt(3))
    
    But this is not smooth at the boundaries where the min switches. We can use a smooth min approximation.
    
    Let's use a soft-min: soft_min(a, b, c) = -1/t * log(exp(-t*a) + exp(-t*b) + exp(-t*c))
    """
    # Transform point to local hexagon coordinates
    dx = px - cx
    dy = py - cy
    angle_rad = math.radians(angle_deg)
    # Rotate by -angle to get to local frame
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    u = cos_a * dx + sin_a * dy
    v = -sin_a * dx + cos_a * dy
    
    # The hexagon in local frame has vertices at (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2),
    # (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
    # The constraints for being inside are:
    # |u| <= s
    # |v| <= s * sqrt(3) / 2
    # |u| + |v| / sqrt(3) <= s
    
    sqrt3 = math.sqrt(3)
    d1 = s - abs(u)
    d2 = s * sqrt3 / 2 - abs(v)
    d3 = s - abs(u) - abs(v) / sqrt3
    
    # Use soft-min for smoothness
    t = 10.0
    # Handle numerical stability
    vals = np.array([d1, d2, d3])
    max_val = np.max(vals)
    soft_min_val = -1.0 / t * math.log(np.sum(np.exp(-t * (vals - max_val)))) + max_val
    
    return soft_min_val

def objective(params, n_inner):
    """
    Objective function to minimize. We want to maximize the minimum clearance,
    so we minimize the negative of the minimum clearance.
    
    params: array of length 3*n_inner + 2
    First 3*n_inner: for each inner hex, (x, y, angle_deg)
    Last 2: outer hex center (ox, oy)
    The outer side length and angle are fixed for now, or we can optimize them too.
    
    Actually, let's fix the outer hex at origin with angle 0 for simplicity,
    and only optimize the inner hex positions. The outer side length is what we're minimizing.
    
    Let's restructure: we'll fix the outer hex center at (0,0) and angle at 0.
    We optimize the inner hex positions and the outer side length.
    
    params: [x1, y1, a1, x2, y2, a2, ..., xn, yn, an, S]
    where S is the outer side length.
    """
    n = n_inner
    S = params[3*n]
    if S < 1.0:
        return 1e6  # Penalty for too small S
    
    min_clearance = float('inf')
    
    for i in range(n):
        x = params[3*i]
        y = params[3*i + 1]
        a = params[3*i + 2]
        
        # Get vertices of the inner hexagon
        verts = hex_vertices(x, y, 1.0, a)
        
        for vx, vy in verts:
            # Compute distance from this vertex to the outer hexagon boundary
            # The outer hexagon is centered at (0,0) with side S and angle 0
            d = point_to_polygon_distance(vx, vy, 0.0, 0.0, S, 0.0)
            if d < min_clearance:
                min_clearance = d
    
    # We want to maximize the minimum clearance, so minimize -min_clearance
    # But we also want to minimize S, so we add a penalty term
    # Actually, we want to find the minimum S such that all inner hexes fit.
    # The constraint is that min_clearance >= 0 (all vertices are inside or on the boundary).
    # So we can minimize S subject to min_clearance >= 0.
    
    # For the optimizer, let's use a penalty method:
    # objective = S + lambda * max(0, -min_clearance)^2
    # But this is not smooth. Let's use a smooth penalty.
    
    penalty = 100.0 * math.exp(-10.0 * min_clearance) if min_clearance < 0 else 0.0
    
    return S + penalty

def construct():
    n_inner = N
    n_params = 3 * n_inner + 1  # 3*n_inner for inner hexes, 1 for outer side length
    
    # Initial guess: place inner hexes in a row
    x0 = []
    for k in range(-(n_inner // 2), n_inner - n_inner // 2):
        x0.append(2.0 * k)  # Spacing of 2 to avoid overlap
        x0.append(0.0)
        x0.append(0.0)
    x0.append(20.0)  # Initial outer side length
    
    x0 = np.array(x0)
    
    # Try multiple random starts
    best_S = float('inf')
    best_params = None
    
    np.random.seed(42)
    
    for trial in range(50):
        # Perturb the initial guess
        x_trial = x0.copy()
        for i in range(n_inner):
            x_trial[3*i] += np.random.uniform(-2, 2)
            x_trial[3*i+1] += np.random.uniform(-2, 2)
            x_trial[3*i+2] += np.random.uniform(-30, 30)
        x_trial[-1] = np.random.uniform(10, 30)
        
        result = minimize(objective, x_trial, args=(n_inner,), method='L-BFGS-B',
                         options={'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6})
        
        if result.success and result.fun < best_S:
            # Check if the solution is valid (min_clearance >= 0)
            S_trial = result.x[-1]
            min_clearance = float('inf')
            for i in range(n_inner):
                x = result.x[3*i]
                y = result.x[3*i+1]
                a = result.x[3*i+2]
                verts = hex_vertices(x, y, 1.0, a)
                for vx, vy in verts:
                    d = point_to_polygon_distance(vx, vy, 0.0, 0.0, S_trial, 0.0)
                    if d < min_clearance:
                        min_clearance = d
            if min_clearance >= -1e-3:  # Allow small numerical error
                best_S = S_trial
                best_params = result.x
    
    if best_params is None:
        # Fallback to a simple packing
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(n_inner // 2), n_inner - n_inner // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Extract the solution
    inner = []
    for i in range(n_inner):
        inner.append([best_params[3*i], best_params[3*i+1], best_params[3*i+2]])
    
    S = best_params[-1]
    
    return inner, (0.0, 0.0), S, 0.0

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
