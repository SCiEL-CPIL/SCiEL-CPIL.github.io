import math
import time
import numpy as np
from scipy.optimize import minimize, linprog
from itertools import combinations

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)
_INVSQRT3 = 1.0 / _SQRT3

def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _min_distance_hexagons(theta_diff_deg):
    """
    Calculate minimum distance between centers of two unit hexagons 
    with relative rotation theta_diff_deg to avoid overlap.
    """
    # Normalize angle to [0, 30]
    delta = theta_diff_deg % 60.0
    if delta > 30.0:
        delta = 60.0 - delta
    
    # If delta is 0, distance is sqrt(3) (inradius * 2)
    # If delta is 30, the "point" of one touches the "side" of the other?
    # Let's compute it numerically for accuracy
    # Vertices of hex1 (angle 0)
    v1 = _hex_vertices(0, 0, 0.0)
    # Vertices of hex2 (angle delta)
    v2 = _hex_vertices(0, 0, delta)
    
    min_dist = float('inf')
    # Check all pairs of vertices
    for p1 in v1:
        for p2 in v2:
            d = np.linalg.norm(p1 - p2)
            if d < min_dist:
                min_dist = d
                
    # Check vertex of 1 to edge of 2, and vertex of 2 to edge of 1
    # Edges of hex2
    edges2 = []
    for i in range(6):
        edges2.append((v2[i], v2[(i+1)%6]))
        
    for p1 in v1:
        for p_start, p_end in edges2:
            # Project p1 onto segment p_start-p_end
            v = p_end - p_start
            w = p1 - p_start
            c1 = np.dot(w, v)
            if c1 <= 0:
                d = np.linalg.norm(p1 - p_start)
            else:
                c2 = np.dot(v, v)
                if c2 <= c1:
                    d = np.linalg.norm(p1 - p_end)
                else:
                    b = c1 / c2
                    v_p = p_start + b * v
                    d = np.linalg.norm(p1 - v_p)
            if d < min_dist:
                min_dist = d
                
    # Edges of hex1
    edges1 = []
    for i in range(6):
        edges1.append((v1[i], v1[(i+1)%6]))
        
    for p2 in v2:
        for p_start, p_end in edges1:
            v = p_end - p_start
            w = p2 - p_start
            c1 = np.dot(w, v)
            if c1 <= 0:
                d = np.linalg.norm(p2 - p_start)
            else:
                c2 = np.dot(v, v)
                if c2 <= c1:
                    d = np.linalg.norm(p2 - p_end)
                else:
                    b = c1 / c2
                    v_p = p_start + b * v
                    d = np.linalg.norm(p2 - v_p)
            if d < min_dist:
                min_dist = d
                
    return min_dist

# Precompute minimum distances for a grid of angles
# Angles from 0 to 30 deg, step 0.5 deg
_ANGLE_GRID = np.arange(0, 30.25, 0.5)
_MIN_DIST_TABLE = np.array([_min_distance_hexagons(a) for a in _ANGLE_GRID])

def _get_min_dist(theta_diff_deg):
    delta = theta_diff_deg % 60.0
    if delta > 30.0:
        delta = 60.0 - delta
    # Interpolate
    idx = int(delta / 0.5)
    if idx >= len(_MIN_DIST_TABLE) - 1:
        return _MIN_DIST_TABLE[-1]
    frac = (delta - idx * 0.5) / 0.5
    return _MIN_DIST_TABLE[idx] * (1 - frac) + _MIN_DIST_TABLE[idx + 1] * frac

def _soft_max(x, beta=50.0):
    max_x = np.max(x)
    shifted = x - max_x
    return max_x + np.log(np.sum(np.exp(beta * shifted))) / beta

def _objective(params, beta=50.0, penalty_weight=1000.0):
    """
    Objective function for optimization.
    params: [x1, y1, t1, ..., x11, y11, t11, outer_angle]
    """
    if time.time() > _DEADLINE - 2.0:
        return 1e6
        
    # Extract positions and angles
    positions = params[:N*2].reshape(N, 2)
    angles = params[N*2:N*3]
    outer_angle = params[N*3]
    
    # 1. Calculate vertices of all inner hexagons
    all_vertices = []
    for i in range(N):
        v = _hex_vertices(positions[i, 0], positions[i, 1], angles[i])
        all_vertices.append(v)
    all_vertices = np.vstack(all_vertices)
    
    # 2. Calculate outer hexagon size (smooth approximation)
    base_angle = math.radians(outer_angle + 30)
    angles_n = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles_n), np.sin(angles_n)])
    
    vdots = all_vertices @ normals.T
    widths = np.max(vdots, axis=0) - np.min(vdots, axis=0)
    R_approx = _soft_max(widths, beta) / 2.0
    side_approx = 2.0 * R_approx * _INVSQRT3
    
    # 3. Calculate collision penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dist = np.linalg.norm(positions[i] - positions[j])
            theta_diff = angles[i] - angles[j]
            min_dist_req = _get_min_dist(theta_diff)
            if dist < min_dist_req:
                penalty += (min_dist_req - dist) ** 2
    
    return side_approx + penalty_weight * penalty

def _exact_side_length(positions, angles, outer_angle):
    """Calculate exact minimum side length using LP."""
    all_vertices = []
    for i in range(N):
        v = _hex_vertices(positions[i, 0], positions[i, 1], angles[i])
        all_vertices.append(v)
    all_vertices = np.vstack(all_vertices)
    
    base_angle = math.radians(outer_angle + 30)
    angles_n = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles_n), np.sin(angles_n)])
    
    M = len(all_vertices)
    vdots = all_vertices @ normals.T
    
    A_rows = []
    b_vals = []
    
    for j in range(6):
        nx, ny = normals[j]
        A_rows.append(np.tile([-nx, -ny, -1.0], (M, 1)))
        b_vals.append(-vdots[:, j])
        A_rows.append(np.tile([nx, ny, -1.0], (M, 1)))
        b_vals.append(vdots[:, j])
        
    A = np.vstack(A_rows)
    b = np.concatenate(b_vals)
    
    A = np.vstack([A, np.array([[0.0, 0.0, -1.0]])])
    b = np.concatenate([b, [0.0]])
    
    c_obj = np.array([0.0, 0.0, 1.0])
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs', options={'presolve': True, 'dual_feasibility_tolerance': 1e-12, 'primal_feasibility_tolerance': 1e-12})
        if res.success:
            cx, cy, R = res.x
            side = R * _INVSQRT3 * 2.0
            return side, cx, cy
    except Exception:
        pass
    return float('inf'), 0, 0

def _check_collision(positions, angles):
    """Check if configuration has collisions."""
    for i in range(N):
        for j in range(i+1, N):
            dist = np.linalg.norm(positions[i] - positions[j])
            theta_diff = angles[i] - angles[j]
            min_dist_req = _get_min_dist(theta_diff)
            if dist < min_dist_req - 1e-6:
                return False
    return True

def _optimize_layout(initial_positions, initial_angles, initial_outer_angle):
    """Run local optimization from a given start."""
    if time.time() > _DEADLINE - 10.0:
        return None
        
    # Fix first hexagon at (0,0) with angle 0 to remove translational and rotational symmetry
    # We will add these back later or just keep them fixed and optimize the rest relative to it.
    # Actually, let's just optimize all and then center them.
    
    x0 = np.concatenate([initial_positions.flatten(), initial_angles, [initial_outer_angle]])
    
    # Bounds: positions in [-10, 10], angles in [0, 60]
    bounds = []
    for i in range(N):
        bounds.append((-10, 10))
        bounds.append((-10, 10))
    for i in range(N):
        bounds.append((0, 60))
    bounds.append((0, 60))
    
    try:
        res = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50})
        
        if res.success:
            # Extract optimized parameters
            opt_positions = res.x[:N*2].reshape(N, 2)
            opt_angles = res.x[N*2:N*3]
            opt_outer_angle = res.x[N*3]
            
            # Check for collisions
            if not _check_collision(opt_positions, opt_angles):
                return None
                
            # Get exact side length
            side, cx, cy = _exact_side_length(opt_positions, opt_angles, opt_outer_angle)
            
            if side < float('inf'):
                # Center the configuration
                # The outer hexagon center is (cx, cy).
                # We should return the inner positions relative to the outer center?
                # No, the interface requires: inner rows (x, y, angle), outer center (x, y), side, angle.
                # The evaluator places inner hexagons at (x, y) and outer at (cx, cy).
                # So we should return the absolute coordinates.
                
                # However, we can translate the whole system so that the outer center is (0,0)
                # and return that, or keep it as is.
                # Let's normalize: shift so outer center is (0,0).
                opt_positions = opt_positions - np.array([cx, cy])
                cx, cy = 0.0, 0.0
                
                return {
                    'positions': opt_positions,
                    'angles': opt_angles,
                    'side': side,
                    'outer_angle': opt_outer_angle,
                    'center': (cx, cy)
                }
    except Exception as e:
        pass
    return None

def _generate_initial_layouts():
    """Generate diverse initial layouts for multi-start."""
    layouts = []
    
    # Layout 1: Flower (1 center, 6 ring, 4 more)
    pos1 = np.array([[0,0]])
    ang1 = np.array([0.0])
    for k in range(6):
        theta = k * 60.0
        x = _SQRT3 * math.cos(math.radians(theta))
        y = _SQRT3 * math.sin(math.radians(theta))
        pos1 = np.vstack([pos1, [[x, y]]])
        ang1 = np.append(ang1, 0.0)
    # Add 4 more from ring 2
    # Ring 2 positions: (2,0), (1,1), etc in axial coords
    # Let's just place them roughly
    for k in [0, 1, 3, 4]:
        theta = k * 60.0
        # Distance for ring 2 is 2 * sqrt(3) ? No, distance between centers in ring 2 is 2*sqrt(3) from center?
        # Center to ring 1 is sqrt(3). Ring 1 to Ring 2 is sqrt(3).
        # So ring 2 radius is 2*sqrt(3)? No.
        # Axial (2,0) -> x = 2*sqrt(3), y=0.
        x = 2 * _SQRT3 * math.cos(math.radians(theta))
        y = 2 * _SQRT3 * math.sin(math.radians(theta))
        pos1 = np.vstack([pos1, [[x, y]]])
        ang1 = np.append(ang1, 0.0)
    layouts.append((pos1, ang1, 0.0))
    
    # Layout 2: Strip
    pos2 = []
    ang2 = []
    for i in range(-2, 3):
        for j in range(0, 3):
            if len(pos2) >= 11: break
            x = i * _SQRT3 + (j % 2) * _SQRT3/2
            y = j * 1.5
            pos2.append([x, y])
            ang2.append(0.0)
    pos2 = np.array(pos2[:11])
    ang2 = np.array(ang2[:11])
    layouts.append((pos2, ang2, 0.0))
    
    # Layout 3: Random
    np.random.seed(42)
    pos3 = np.random.uniform(-5, 5, (11, 2))
    ang3 = np.random.uniform(0, 60, 11)
    layouts.append((pos3, ang3, 0.0))
    
    return layouts

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    best_result = None
    best_side = float('inf')
    
    initial_layouts = _generate_initial_layouts()
    
    for layout_idx, (pos, ang, outer_ang) in enumerate(initial_layouts):
        if time.time() > _DEADLINE - 15.0:
            break
            
        # Try multiple outer angles for each layout
        for start_outer_angle in [0.0, 15.0, 30.0]:
            if time.time() > _DEADLINE - 15.0:
                break
                
            result = _optimize_layout(pos, ang, start_outer_angle)
            if result:
                if result['side'] < best_side:
                    best_side = result['side']
                    best_result = result
                    
    if best_result is None:
        # Fallback to a valid but loose packing
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    # Prepare output
    inner = []
    for i in range(N):
        inner.append([best_result['positions'][i][0], best_result['positions'][i][1], best_result['angles'][i]])
        
    return inner, best_result['center'], best_result['side'], best_result['outer_angle']
