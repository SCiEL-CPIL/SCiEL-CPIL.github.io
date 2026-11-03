import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 120.0

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    """Get vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _generate_hex_layout(n_hexes):
    """
    Generate a hexagonal arrangement of n unit hexagons.
    Returns list of (x, y) positions.
    """
    sqrt3 = math.sqrt(3)
    scale = 1.000001  # Slightly larger than touching
    
    # Hexagonal lattice vectors
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
    
    # Generate candidates in a hexagonal pattern
    candidates = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            dist_sq = x * x + y * y
            candidates.append((dist_sq, x, y))
    
    candidates.sort(key=lambda t: t[0])
    
    positions = []
    for _, x, y in candidates:
        positions.append((x, y))
        if len(positions) == n_hexes:
            break
    
    return positions

def _smooth_container_side(positions, inner_angle_deg, cx, cy, outer_angle_deg, beta=50.0):
    """
    Calculate the smooth approximation of the minimum enclosing hexagon side length.
    """
    # Precompute normals for the outer hexagon
    outer_angle_rad = math.radians(outer_angle_deg)
    normals = []
    for k in range(6):
        a = outer_angle_rad + math.pi / 6 + math.pi * k / 3
        normals.append((math.cos(a), math.sin(a)))
    
    # Precompute vertices of inner hexagons? 
    # No, we need the support function of the *set* of vertices.
    # The constraint is that ALL vertices of ALL inner hexagons must be inside the outer hexagon.
    # So we need to iterate over all vertices.
    
    all_vertices = []
    for px, py in positions:
        verts = _get_hex_vertices(px, py, inner_angle_deg)
        all_vertices.extend(verts)
    
    # all_vertices is a list of (vx, vy)
    # We want to compute soft_max of | (v - c) . n | for all v, n
    # To avoid memory issues, we can accumulate the log-sum-exp iteratively.
    
    # soft_max(x_1, ..., x_n) = (1/beta) * log( sum exp(beta * x_i) )
    # We want max over i,j of | dot(v_i - c, n_j) |
    # Let x_{i,j} = | dot(v_i - c, n_j) |
    # We compute log_sum_exp( beta * x_{i,j} )
    
    # Use the trick: log(sum exp(y)) = max(y) + log(sum exp(y - max(y)))
    # But we have absolute values, so we can't just take max of the raw dots.
    # We can iterate.
    
    max_val = 0.0
    sum_exp = 0.0
    
    # For numerical stability, we can do this in two passes or just use a large enough buffer
    # Since N=11, 66 vertices, 6 normals. 396 terms. This is small.
    
    terms = []
    for vx, vy in all_vertices:
        for nx, ny in normals:
            d = abs((vx - cx) * nx + (vy - cy) * ny)
            terms.append(d)
    
    if not terms:
        return 1e10
        
    max_term = max(terms)
    sum_exp = 0.0
    for t in terms:
        sum_exp += math.exp(beta * (t - max_term))
        
    soft_max_val = max_term + math.log(sum_exp) / beta
    
    # Side length = inradius / (sqrt(3)/2)
    # inradius = soft_max_val
    side = soft_max_val / (math.sqrt(3) / 2)
    
    return side

def _optimize_container(positions, inner_angle_deg, init_cx, init_cy, init_angle_deg, max_iter=200):
    """
    Optimize the outer container center and angle using L-BFGS-B.
    """
    def objective(params):
        cx, cy, angle_deg = params
        # Angle is periodic, but L-BFGS-B works on R.
        # We can wrap angle to [0, 30) if needed, but the function is smooth enough locally.
        return _smooth_container_side(positions, inner_angle_deg, cx, cy, angle_deg, beta=50.0)
    
    x0 = np.array([init_cx, init_cy, init_angle_deg])
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', 
                         options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 20})
        if result.success:
            cx, cy, angle_deg = result.x
            # Refine with exact LP?
            # For now, return the smooth result side length
            final_side = _smooth_container_side(positions, inner_angle_deg, cx, cy, angle_deg, beta=100.0)
            return cx, cy, final_side, angle_deg
        else:
            return None
    except Exception:
        return None

def _find_exact_side_lp(positions, inner_angle_deg, cx, cy, angle_deg):
    """
    Find the exact minimum side length given center and angle using LP.
    This is used to verify the smooth result.
    """
    all_vertices = []
    for px, py in positions:
        verts = _get_hex_vertices(px, py, inner_angle_deg)
        all_vertices.extend(verts)
    
    # We are fixing cx, cy, angle.
    # We just need to find the min R (inradius) such that all vertices are inside.
    # R >= max | (v - c) . n |
    # So R = max | (v - c) . n |
    # Side = R / (sqrt(3)/2)
    
    max_r = 0.0
    angle_rad = math.radians(angle_deg)
    for k in range(6):
        a = angle_rad + math.pi / 6 + math.pi * k / 3
        nx = math.cos(a)
        ny = math.sin(a)
        for vx, vy in all_vertices:
            d = abs((vx - cx) * nx + (vy - cy) * ny)
            if d > max_r:
                max_r = d
                
    side = max_r / (math.sqrt(3) / 2)
    return side

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Generate hexagonal layout
    positions = _generate_hex_layout(N)
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Try inner angles 0 and 30
    for inner_angle in [0.0, 30.0]:
        if time.time() > _DEADLINE - 20.0:
            break
            
        # Initial guess for center: centroid of positions
        cx_init = sum(p[0] for p in positions) / N
        cy_init = sum(p[1] for p in positions) / N
        
        # Coarse search for outer angle to get a good initial guess
        best_coarse_side = float('inf')
        best_coarse_angle = 0.0
        best_coarse_center = (cx_init, cy_init)
        
        # Symmetry: 0 to 30 degrees
        for outer_angle_deg in np.arange(0, 30.01, 1.0):
            if time.time() > _DEADLINE - 20.0:
                break
            # Use smooth approximation for speed in coarse search
            side = _smooth_container_side(positions, inner_angle, cx_init, cy_init, outer_angle_deg, beta=10.0)
            if side < best_coarse_side:
                best_coarse_side = side
                best_coarse_angle = outer_angle_deg
        
        # Polish with L-BFGS-B
        if best_coarse_side < float('inf'):
            result = _optimize_container(positions, inner_angle, cx_init, cy_init, best_coarse_angle, max_iter=100)
            if result:
                cx, cy, side, angle_deg = result
                # Verify with exact calculation
                exact_side = _find_exact_side_lp(positions, inner_angle, cx, cy, angle_deg)
                if exact_side < best_side:
                    best_side = exact_side
                    best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
                    best_center = (cx, cy)
                    best_outer_angle = angle_deg
    
    if best_inner is None:
        # Fallback if optimization failed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner, best_center, best_side, best_outer_angle
