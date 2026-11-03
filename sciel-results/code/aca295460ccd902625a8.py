import math
import time
import numpy as np
from scipy.optimize import linprog, minimize, minimize_scalar, differential_evolution

# Constants
N_HEX = 11
APOTHEM = math.sqrt(3) / 2.0
DIAMETER = 2.0
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9
START_TIME = time.time()

# Unit hexagon vertices (centered at origin, flat-top orientation)
def get_unit_hexagon_vertices():
    """Get vertices of a unit regular hexagon (side=1, flat-top) centered at origin."""
    verts = []
    for i in range(6):
        angle = math.pi / 3.0 * i
        verts.append([math.cos(angle), math.sin(angle)])
    return np.array(verts)

UNIT_VERTS = get_unit_hexagon_vertices()

def time_left():
    """Return remaining time in seconds."""
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)

def check_time():
    """Check if we should stop."""
    return time_left() <= 0

def hexagon_vertices(cx, cy, angle_deg, scale=1.0):
    """Get vertices of a regular hexagon with center (cx, cy), rotation angle_deg, scale factor."""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return scale * (UNIT_VERTS @ rot.T) + np.array([cx, cy])

def hexagon_in_hexagon_test(inner_hexes, outer_center, outer_side, outer_angle_deg):
    """
    Test if all inner hexagons fit inside the outer hexagon.
    inner_hexes: list of (cx, cy, angle_deg)
    Returns True if valid.
    """
    # Get outer hexagon vertices and normals
    outer_verts = hexagon_vertices(outer_center[0], outer_center[1], outer_angle_deg, outer_side)
    
    # For each side of the outer hexagon, check that all inner hex vertices are on the correct side
    for i in range(6):
        v1 = outer_verts[i]
        v2 = outer_verts[(i + 1) % 6]
        # Edge vector
        edge = v2 - v1
        # Inward normal (pointing toward center)
        normal = np.array([-edge[1], edge[0]])
        # Ensure normal points inward
        center_vec = np.array(outer_center) - v1
        if np.dot(normal, center_vec) < 0:
            normal = -normal
        # Normalize
        normal = normal / np.linalg.norm(normal)
        
        # Check all inner hexagon vertices
        for (icx, icy, iangle) in inner_hexes:
            inner_verts = hexagon_vertices(icx, icy, iangle)
            for iv in inner_verts:
                # Project onto normal
                proj = np.dot(iv - v1, normal)
                # The distance from edge should be <= the apothem of outer hexagon
                # Actually, we need proj <= 0 (inside)
                if proj > 1e-8:
                    return False
    return True

def compute_outer_side_for_config(inner_hexes, outer_center, outer_angle_deg):
    """
    For a given configuration of inner hexagons (positions and orientations) 
    and outer hexagon center/orientation, compute the minimum outer side length
    that contains all inner hexagons.
    
    This is a linear program: minimize D subject to all vertices being inside.
    Returns (side_length, center, angle_deg) or None if infeasible.
    """
    # Collect all vertices from all inner hexagons
    all_verts = []
    for (cx, cy, angle) in inner_hexes:
        verts = hexagon_vertices(cx, cy, angle)
        all_verts.extend(verts.tolist())
    all_verts = np.array(all_verts)
    
    if len(all_verts) == 0:
        return 1.0, outer_center, outer_angle_deg
    
    # For a hexagon with center (hx, hy), rotation angle_deg, and side length S:
    # The constraint is that for each of the 6 directions (normals), the projection
    # of every vertex onto the normal direction from the center is <= S * APOTHEM
    # (the apothem of the outer hexagon).
    
    # Actually, let's think about this differently.
    # A regular hexagon with center (hx, hy), side length S, and rotation angle 
    # can be described as the intersection of 6 half-planes.
    # The normal directions are at angles: outer_angle_deg + 30° + k*60° for k=0..5
    # (since the normals point to the midpoints of the edges)
    
    # For each normal direction, the constraint is:
    #   max_i (dot(v_i - center, n_k)) <= S * APOTHEM
    #   min_i (dot(v_i - center, n_k)) >= -S * APOTHEM
    # Combined: for each k, both max and min projections must satisfy |proj| <= S * APOTHEM
    # Which means: max_i |dot(v_i - center, n_k)| <= S * APOTHEM for all k
    # So: S >= max_k (max_i |dot(v_i - center, n_k)|) / APOTHEM
    
    # We want to minimize S by choosing center (hx, hy) and S, with fixed angle.
    # This is a linear program in variables (hx, hy, S):
    #   Minimize S
    #   Subject to: for each k in 0..5:
    #       For each vertex v_i:
    #           dot(v_i - (hx,hy), n_k) <= S * APOTHEM
    #           dot((hx,hy) - v_i, n_k) <= S * APOTHEM  (equivalently)
    #       S >= 0
    
    angle_rad = math.radians(outer_angle_deg)
    
    # Normal directions
    normals = []
    for k in range(6):
        ang = angle_rad + math.pi / 6.0 + k * math.pi / 3.0
        normals.append([math.cos(ang), math.sin(ang)])
    normals = np.array(normals)
    
    n_verts = len(all_verts)
    
    # Variables: [hx, hy, S]
    # Objective: minimize S -> [0, 0, 1]
    c = np.array([0.0, 0.0, 1.0])
    
    A_ub = []
    b_ub = []
    
    for k in range(6):
        nx, ny = normals[k]
        for i in range(n_verts):
            vx, vy = all_verts[i]
            # dot(v_i - center, n_k) <= S * APOTHEM
            # vx*nx + vy*ny - hx*nx - hy*ny <= S * APOTHEM
            # -nx*hx - ny*hy - APOTHEM*S <= -(vx*nx + vy*ny)
            A_ub.append([-nx, -ny, -APOTHEM])
            b_ub.append(-(vx * nx + vy * ny))
            
            # dot(center - v_i, n_k) <= S * APOTHEM
            # hx*nx + hy*ny - vx*nx - vy*ny <= S * APOTHEM
            # nx*hx + ny*hy - APOTHEM*S <= vx*nx + vy*ny
            A_ub.append([nx, ny, -APOTHEM])
            b_ub.append(vx * nx + vy * ny)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = [(None, None), (None, None), (0.1, None)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            hx, hy, S = res.x
            return S, (float(hx), float(hy)), outer_angle_deg
        else:
            return None
    except Exception:
        return None

def outer_side_direct(inner_hexes, outer_center, outer_angle_deg):
    """Direct computation of required outer side length for given configuration."""
    all_verts = []
    for (cx, cy, angle) in inner_hexes:
        verts = hexagon_vertices(cx, cy, angle)
        all_verts.extend(verts.tolist())
    all_verts = np.array(all_verts)
    
    if len(all_verts) == 0:
        return 1.0
    
    angle_rad = math.radians(outer_angle_deg)
    max_proj = 0.0
    
    for k in range(6):
        ang = angle_rad + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (all_verts[:, 0] - outer_center[0]) * nx + (all_verts[:, 1] - outer_center[1]) * ny
        max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
    
    return (max_proj + APOTHEM) / APOTHEM

def non_overlap_constraint_test(inner_hexes, tol=1e-7):
    """
    Test if all inner hexagons are pairwise non-overlapping (disjoint interiors).
    Uses separating axis theorem for convex polygons.
    Returns True if non-overlapping.
    """
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if polygons_overlap(inner_hexes[i], inner_hexes[j], tol):
                return False
    return True

def polygons_overlap(hex1, hex2, tol=1e-7):
    """
    Check if two hexagons overlap using SAT (Separating Axis Theorem).
    hex1, hex2: (cx, cy, angle_deg)
    Returns True if they overlap.
    """
    verts1 = hexagon_vertices(hex1[0], hex1[1], hex1[2])
    verts2 = hexagon_vertices(hex2[0], hex2[1], hex2[2])
    
    # Get all edge normals from both polygons
    edges = []
    for verts in [verts1, verts2]:
        for i in range(6):
            v1 = verts[i]
            v2 = verts[(i + 1) % 6]
            edge = v2 - v1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                normal = normal / norm
                edges.append(normal)
    
    # Test each edge as potential separating axis
    for normal in edges:
        # Project both polygons onto this axis
        proj1 = np.dot(verts1, normal)
        proj2 = np.dot(verts2, normal)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check for separation
        if max1 < min2 - tol or max2 < min1 - tol:
            return False  # Found a separating axis, no overlap
    
    return True  # No separating axis found, polygons overlap

def random_initial_config():
    """Generate a random initial configuration of 11 unit hexagons."""
    # Place them in a rough grid pattern with some randomness
    config = []
    # Try a few different initial patterns
    patterns = [
        # Pattern 0: hexagonal packing
        [(0, 0), (2, 0), (4, 0), (-2, 0), (-4, 0),
         (1, 1.732), (3, 1.732), (-1, 1.732), (-3, 1.732),
         (1, -1.732), (3, -1.732)],
        # Pattern 1: linear
        [(-4, 0), (-2, 0), (0, 0), (2, 0), (4, 0),
         (-4, 2), (-2, 2), (0, 2), (2, 2), (4, 2), (0, -2)],
        # Pattern 2: compact
        [(0, 0), (2, 0), (4, 0), (-2, 0), (-4, 0),
         (1, 1.732), (3, 1.732), (-1, 1.732), (-3, 1.732),
         (1, -1.732), (3, -1.732)],
        # Pattern 3: random-ish
        [(i * 2.01 - 4.0, (i % 2) * 1.74 - 0.87) for i in range(11)],
    ]
    
    pat_idx = np.random.randint(0, len(patterns))
    base = patterns[pat_idx]
    
    for i in range(11):
        cx = base[i][0] + np.random.randn() * 0.1
        cy = base[i][1] + np.random.randn() * 0.1
        angle = np.random.choice([0, 30, 60, 90, 120, 150])
        config.append((float(cx), float(cy), float(angle)))
    
    return config

def compute_overlap_penalty(config, tol=1e-3):
    """
    Compute a penalty for overlapping hexagons.
    Returns (total_penalty, max_overlap).
    """
    total_penalty = 0.0
    max_overlap = 0.0
    n = len(config)
    for i in range(n):
        for j in range(i + 1, n):
            # Check for overlap
            if polygons_overlap(config[i], config[j], -tol):  # negative tol means "overlap with margin"
                # Estimate overlap amount
                verts1 = hexagon_vertices(config[i][0], config[i][1], config[i][2])
                verts2 = hexagon_vertices(config[j][0], config[j][1], config[j][2])
                
                # Simple overlap metric: average distance between closest points
                min_dist = float('inf')
                for v1 in verts1:
                    for v2 in verts2:
                        d = np.linalg.norm(v1 - v2)
                        min_dist = min(min_dist, d)
                
                # If they overlap, the "depth" is roughly related to how much they interpenetrate
                # Use a simple penalty
                penalty = max(0, tol - min_dist)
                total_penalty += penalty
                max_overlap = max(max_overlap, penalty)
    
    return total_penalty, max_overlap

def optimize_positions_for_fixed_angles(config, outer_angle_deg, max_iters=50):
    """
    For fixed orientations, optimize positions and outer hexagon center/size.
    Uses iterative LP-based approach.
    """
    # This is a complex problem. For now, use a simpler approach:
    # Given fixed angles, find the best outer hexagon that contains all inner hexagons,
    # then check for overlaps.
    
    # First, find a reasonable outer hexagon
    all_verts = []
    for (cx, cy, angle) in config:
        verts = hexagon_vertices(cx, cy, angle)
        all_verts.extend(verts.tolist())
    all_verts = np.array(all_verts)
    
    # Center of mass
    cx_mean = np.mean(all_verts[:, 0])
    cy_mean = np.mean(all_verts[:, 1])
    
    # Compute required side length
    side = outer_side_direct(config, (cx_mean, cy_mean), outer_angle_deg)
    
    return config, (cx_mean, cy_mean), side

def full_optimization(max_time=80.0):
    """
    Full optimization: optimize positions, orientations, and outer hexagon.
    """
    best_config = None
    best_outer = None
    best_side = float('inf')
    
    # Try multiple random restarts
    n_restarts = 5
    time_per_restart = max_time / n_restarts
    
    for restart in range(n_restarts):
        if check_time():
            break
        
        # Generate initial config
        config = random_initial_config()
        
        # Iterative optimization
        for iteration in range(30):
            if check_time() or (time.time() - START_TIME) > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            # Step 1: Optimize outer hexagon for current config
            all_verts = []
            for (cx, cy, angle) in config:
                verts = hexagon_vertices(cx, cy, angle)
                all_verts.extend(verts.tolist())
            all_verts = np.array(all_verts)
            
            cx_mean = np.mean(all_verts[:, 0])
            cy_mean = np.mean(all_verts[:, 1])
            
            # Try different outer angles
            best_outer_angle = 0
            best_side_this = float('inf')
            best_center_this = None
            
            for outer_angle in [0, 15, 30, 45, 60, 75, 90]:
                side = outer_side_direct(config, (cx_mean, cy_mean), outer_angle)
                if side < best_side_this:
                    best_side_this = side
                    best_outer_angle = outer_angle
                    best_center_this = (cx_mean, cy_mean)
            
            # Refine outer angle
            def outer_angle_obj(a):
                return outer_side_direct(config, best_center_this, a)
            
            try:
                res = minimize_scalar(outer_angle_obj, bounds=(max(0, best_outer_angle - 15), min(90, best_outer_angle + 15)), 
                                       method='bounded', options={'maxiter': 100})
                if res.fun < best_side_this:
                    best_side_this = res.fun
                    best_outer_angle = res.x
                    best_center_this = (cx_mean, cy_mean)
            except Exception:
                pass
            
            # Refine center
            def center_obj(params):
                return outer_side_direct(config, (params[0], params[1]), best_outer_angle)
            
            try:
                res = minimize(center_obj, [cx_mean, cy_mean], method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
                if res.fun < best_side_this:
                    best_side_this = res.fun
                    best_center_this = (res.x[0], res.x[1])
            except Exception:
                pass
            
            # Step 2: Check for overlaps
            has_overlap = not non_overlap_constraint_test(config, tol=1e-6)
            
            if not has_overlap:
                # Valid configuration
                if best_side_this < best_side:
                    best_side = best_side_this
                    best_config = config[:]
                    best_outer = (best_center_this, best_outer_angle)
            else:
                # Try to resolve overlaps by adjusting positions
                # Simple gradient-free approach: push overlapping hexagons apart
                overlap_penalty, max_overlap = compute_overlap_penalty(config, tol=1e-4)
                
                if overlap_penalty > 0:
                    # Perturb overlapping hexagons
                    for i in range(11):
                        for j in range(i + 1, 11):
                            if polygons_overlap(config[i], config[j], -1e-4):
                                # Push apart
                                dx = config[j][0] - config[i][0]
                                dy = config[j][1] - config[i][1]
                                dist = math.sqrt(dx*dx + dy*dy)
                                if dist > 1e-6:
                                    push = 0.05
                                    config[i] = (config[i][0] - push*dx/dist, 
                                                 config[i][1] - push*dy/dist, 
                                                 config[i][2])
                                    config[j] = (config[j][0] + push*dx/dist, 
                                                 config[j][1] + push*dy/dist, 
                                                 config[j][2])
                            else:
                                # Try random perturbation
                                if np.random.random() < 0.3:
                                    config[i] = (config[i][0] + np.random.randn()*0.1,
                                                 config[i][1] + np.random.randn()*0.1,
                                                 config[i][2])
                
                # Re-check
                if not non_overlap_constraint_test(config, tol=1e-6):
                    continue
                
                # If no overlap after adjustment, evaluate
                side = outer_side_direct(config, best_center_this, best_outer_angle)
                if side < best_side:
                    best_side = side
                    best_config = config[:]
                    best_outer = (best_center_this, best_outer_angle)
        
        # After restart, do a local refinement
        if best_config is not None and best_outer is not None:
            # Refine each hexagon's angle
            for i in range(11):
                if check_time():
                    break
                
                def angle_obj_i(a, idx=i):
                    test_config = best_config[:]
                    test_config[idx] = (test_config[idx][0], test_config[idx][1], a)
                    if not non_overlap_constraint_test(test_config, tol=1e-6):
                        return float('inf')
                    return outer_side_direct(test_config, best_outer[0], best_outer[1])
                
                try:
                    res = minimize_scalar(angle_obj_i, bounds=(0, 60), method='bounded',
                                           options={'maxiter': 100})
                    if res.fun < best_side - 1e-6:
                        best_config[i] = (best_config[i][0], best_config[i][1], res.x)
                        best_side = res.fun
                except Exception:
                    pass
            
            # Refine positions
            for i in range(11):
                if check_time():
                    break
                
                def pos_obj_i(params, idx=i):
                    test_config = best_config[:]
                    test_config[idx] = (params[0], params[1], test_config[idx][2])
                    if not non_overlap_constraint_test(test_config, tol=1e-6):
                        return float('inf')
                    return outer_side_direct(test_config, best_outer[0], best_outer[1])
                
                try:
                    res = minimize(pos_obj_i, [best_config[i][0], best_config[i][1]],
                                   method='Nelder-Mead',
                                   options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5})
                    if res.fun < best_side - 1e-6:
                        best_config[i] = (res.x[0], res.x[1], best_config[i][2])
                        best_side = res.fun
                except Exception:
                    pass
            
            # Refine outer hexagon
            def outer_obj(params):
                cx, cy, angle = params
                return outer_side_direct(best_config, (cx, cy), angle)
            
            try:
                res = minimize(outer_obj, [best_outer[0][0], best_outer[0][1], best_outer[1]],
                               method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
                if res.fun < best_side - 1e-6:
                    best_outer = ((res.x[0], res.x[1]), res.x[2])
                    best_side = res.fun
            except Exception:
                pass
    
    return best_config, best_outer, best_side

def construct():
    """
    Main entry point. Returns (inner, center, side, angle) where:
    - inner: list of [x, y, angle_degrees] for each of the 11 unit hexagons
    - center: [x, y] of the outer hexagon
    - side: side length of the outer hexagon
    - angle: rotation angle of the outer hexagon in degrees
    """
    if time_left() <= 0:
        # Emergency fallback
        inner = [[float(i * 2.0), 0.0, 0.0] for i in range(11)]
        return inner, [0.0, 0.0], 25.0, 0.0
    
    # Run the optimization
    config, outer, side = full_optimization(max_time=TIME_LIMIT - SAFETY_MARGIN - 5.0)
    
    if config is None or outer is None or side == float('inf'):
        # Fallback: simple linear arrangement
        config = [(float(i * 2.0 - 10.0), 0.0, 0.0) for i in range(11)]
        outer = ((0.0, 0.0), 0.0)
        side = outer_side_direct(config, outer[0], outer[1])
    
    # Build the result
    inner = []
    for (cx, cy, angle) in config:
        inner.append([float(cx), float(cy), float(angle)])
    
    center = [float(outer[0][0]), float(outer[0][1])]
    side = float(side)
    angle = float(outer[1])
    
    return inner, center, side, angle
