import numpy as np
import time
import random
import math
from scipy.optimize import minimize

N = 11
DEADLINE = 150.0
START_TIME = time.time()

def _remaining_time():
    return DEADLINE - (time.time() - START_TIME)

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with side length 'side'."""
    angle_rad = np.radians(angle_deg)
    # Vertices at angles 0, 60, 120, 180, 240, 300 relative to angle_deg
    # Side length s, circumradius = s
    # x = cx + s * cos(a), y = cy + s * sin(a)
    a = angle_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(a), cy + side * np.sin(a)])

def _get_outer_normals(angle_deg):
    """
    Return 6 unit outward normals for a regular hexagon centered at origin,
    rotated by angle_deg.
    For a hexagon with side s, inradius r = s * sqrt(3)/2.
    The normals point to the midpoints of the edges.
    Edge midpoints are at angles 30, 90, 150, 210, 270, 330 degrees relative to the hexagon's orientation.
    If the hexagon is rotated by angle_deg, the normals are also rotated.
    """
    base_angles = np.radians([30, 90, 150, 210, 270, 330])
    rot = np.radians(angle_deg)
    angles = base_angles + rot
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _min_enclosing_side(inner_positions, outer_center, outer_angle):
    """
    Compute the minimum side length of the outer hexagon to contain all inner hexagons.
    inner_positions: list of (x, y, angle_deg)
    outer_center: (cx, cy)
    outer_angle: degrees
    """
    cx, cy = outer_center
    normals = _get_outer_normals(outer_angle)
    
    max_inradius = 0.0
    for (ix, iy, iangle) in inner_positions:
        verts = _hex_vertices(ix, iy, 1.0, iangle)
        # Translate to outer center frame
        verts_local = verts - np.array([cx, cy])
        # Project onto each normal
        # projections = verts_local @ normals.T  (shape: 6x6)
        # For each normal, we need the max projection
        # We want the global max projection over all vertices and all normals
        # h = max_{v, n} dot(v, n)
        dots = verts_local @ normals.T
        max_dot = np.max(dots)
        if max_dot > max_inradius:
            max_inradius = max_dot
            
    # Inradius R = S * sqrt(3) / 2  => S = 2 * R / sqrt(3)
    return 2.0 * max_inradius / np.sqrt(3)

def _overlap_depth(verts1, verts2):
    """
    Compute the penetration depth between two convex hexagons.
    Returns 0 if disjoint, > 0 if overlapping.
    """
    # Get normals for both
    # For a regular hexagon, normals are perpendicular to edges.
    # We can just use the vertices to compute edge normals.
    
    def get_normals(verts):
        n = len(verts)
        normals = []
        for i in range(n):
            j = (i + 1) % n
            edge = verts[j] - verts[i]
            length = np.linalg.norm(edge)
            if length < 1e-10:
                continue
            # Perpendicular vector
            nx = -edge[1] / length
            ny = edge[0] / length
            normals.append([nx, ny])
        return np.array(normals)

    n1 = get_normals(verts1)
    n2 = get_normals(verts2)
    all_normals = np.vstack([n1, n2])
    
    min_overlap = float('inf')
    
    for n in all_normals:
        proj1 = verts1 @ n
        proj2 = verts2 @ n
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check for separation
        if max1 <= min2 or max2 <= min1:
            return 0.0
            
        # Overlap amount on this axis
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
            
    return min_overlap

def _check_validity(inner_positions, outer_center, outer_side, outer_angle, tol=1e-6):
    """
    Check if the configuration is valid.
    """
    # Check containment
    side_calc = _min_enclosing_side(inner_positions, outer_center, outer_angle)
    if side_calc > outer_side + tol:
        return False
        
    # Check overlaps
    verts_list = [_hex_vertices(ix, iy, 1.0, iangle) for (ix, iy, iangle) in inner_positions]
    for i in range(N):
        for j in range(i + 1, N):
            depth = _overlap_depth(verts_list[i], verts_list[j])
            if depth > tol:
                return False
    return True

def _objective(x, outer_angle):
    """
    Objective function for optimization.
    x: flattened array of [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN]
    Returns: Penalty + Side Length
    """
    pos = x.reshape(N, 3)
    
    # Calculate current enclosing side
    S = _min_enclosing_side([tuple(p) for p in pos], (0.0, 0.0), outer_angle)
    
    # Calculate overlap penalty
    overlap_penalty = 0.0
    verts_list = [_hex_vertices(p[0], p[1], 1.0, p[2]) for p in pos]
    for i in range(N):
        for j in range(i + 1, N):
            depth = _overlap_depth(verts_list[i], verts_list[j])
            if depth > 0:
                overlap_penalty += depth * depth
                
    # We want to minimize S, but heavily penalize overlaps.
    # If overlaps are 0, we just minimize S.
    # If overlaps > 0, the penalty dominates.
    return S + 100.0 * overlap_penalty

def _optimize_from_start(start_pos, outer_angle, time_budget):
    """
    Run L-BFGS-B optimization from a starting position.
    """
    x0 = np.array([p for pos in start_pos for p in pos])
    
    # Normalize angles to [0, 360)
    for i in range(0, len(x0), 3):
        x0[i+2] = x0[i+2] % 360.0
        
    def obj(x):
        if _remaining_time() < 2.0:
            return 1e9
        return _objective(x, outer_angle)
        
    # Options
    max_iter = 200
    max_fun = 1000
    
    try:
        res = minimize(obj, x0, method='L-BFGS-B', 
                       options={'maxiter': max_iter, 'maxfun': max_fun, 'ftol': 1e-10})
        best_x = res.x
        best_val = res.fun
    except Exception:
        best_x = x0
        best_val = obj(x0)
        
    # Extract positions
    best_pos = best_x.reshape(N, 3)
    # Normalize angles again
    for i in range(N):
        best_pos[i, 2] = best_pos[i, 2] % 360.0
        
    # Calculate true side length
    true_side = _min_enclosing_side([tuple(p) for p in best_pos], (0.0, 0.0), outer_angle)
    
    # Check validity
    if _check_validity([tuple(p) for p in best_pos], (0.0, 0.0), true_side, outer_angle, tol=1e-5):
        return [tuple(p) for p in best_pos], true_side
    else:
        return None, float('inf')

def _initial_lattice():
    """Hexagonal lattice arrangement."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    d = 2.0 * np.sqrt(3)
    for k in range(6):
        a = np.radians(60 * k)
        positions.append([d * np.cos(a), d * np.sin(a), 0.0])
    d2 = d * np.sqrt(3)
    for k in range(4):
        a = np.radians(30 + 60 * k)
        positions.append([d2 * np.cos(a), d2 * np.sin(a), 0.0])
    return positions

def _initial_random():
    """Random positions."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    for i in range(1, N):
        r = random.uniform(1.0, 4.0)
        a = random.uniform(0, 2 * np.pi)
        angle = random.uniform(0, 360)
        positions.append([r * np.cos(a), r * np.sin(a), angle])
    return positions

def construct():
    """
    Main entry point.
    Returns: (inner, center, side, angle)
    inner: list of (x, y, angle_degrees)
    center: (x, y)
    side: float
    angle: float (outer hexagon rotation)
    """
    random.seed(42)
    np.random.seed(42)
    
    best_result = None
    best_score = float('inf')
    
    # Outer angles to try
    # The optimal packing might depend on the orientation of the outer hexagon relative to the inner ones.
    # We try a range of angles.
    outer_angles = [0.0, 15.0, 30.0, 45.0, 10.0, 20.0, 25.0, 35.0]
    
    # Initial configurations
    inits = [
        _initial_lattice(),
        _initial_random(),
        _initial_random(),
        _initial_random()
    ]
    
    # Add some perturbed lattices
    for _ in range(3):
        lat = _initial_lattice()
        perturbed = []
        for p in lat:
            perturbed.append([
                p[0] + random.uniform(-0.5, 0.5),
                p[1] + random.uniform(-0.5, 0.5),
                p[2] + random.uniform(-10, 10)
            ])
        inits.append(perturbed)
        
    # Main search loop
    for oa in outer_angles:
        if _remaining_time() < 10.0:
            break
            
        for init in inits:
            if _remaining_time() < 5.0:
                break
                
            # Copy init to avoid modification
            init_copy = [list(p) for p in init]
            
            try:
                res_pos, res_side = _optimize_from_start(init_copy, oa, 10.0)
                if res_pos is not None and res_side < best_score:
                    best_score = res_side
                    best_result = (res_pos, (0.0, 0.0), res_side, oa)
            except Exception:
                pass

    # If we have a good result, try to refine it with a finer angle search
    if best_result is not None and _remaining_time() > 15.0:
        best_angle = best_result[3]
        # Fine tune angle
        for da in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            if _remaining_time() < 5.0:
                break
            oa = best_angle + da
            init_copy = [list(p) for p in best_result[0]]
            # Add slight perturbation to escape local minima
            perturbed = []
            for p in init_copy:
                perturbed.append([
                    p[0] + random.uniform(-0.1, 0.1),
                    p[1] + random.uniform(-0.1, 0.1),
                    p[2] + random.uniform(-5, 5)
                ])
            try:
                res_pos, res_side = _optimize_from_start(perturbed, oa, 5.0)
                if res_pos is not None and res_side < best_score:
                    best_score = res_side
                    best_result = (res_pos, (0.0, 0.0), res_side, oa)
            except Exception:
                pass

    if best_result is not None:
        return best_result
    
    # Fallback: return the lattice configuration
    pos = _initial_lattice()
    side = _min_enclosing_side(pos, (0.0, 0.0), 0.0)
    return pos, (0.0, 0.0), side, 0.0
