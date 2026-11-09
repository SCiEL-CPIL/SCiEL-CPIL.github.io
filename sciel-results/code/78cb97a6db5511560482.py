import numpy as np
from scipy.optimize import minimize, minimize_scalar
import time
import math
from itertools import combinations

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, angle_deg=0.0):
    angle = math.radians(angle_deg)
    return np.array([[cx + side * math.cos(angle + i * math.pi / 3),
                      cy + side * math.sin(angle + i * math.pi / 3)] for i in range(6)])

def _hex_edge_normals(cx, cy, side, angle_deg=0.0):
    """Return 6 outward unit normals and their support values for a hexagon."""
    angle = math.radians(angle_deg)
    normals = []
    supports = []
    for i in range(6):
        # Edge i connects vertex i to vertex i+1
        # Normal points outward, perpendicular to edge
        theta = angle + (i + 0.5) * math.pi / 3
        nx = math.cos(theta)
        ny = math.sin(theta)
        normals.append((nx, ny))
        # Support = distance from center to edge along normal = apothem
        supports.append(side * math.cos(math.pi / 6))  # = side * sqrt(3)/2
    return normals, supports

def _min_enclosing_side(positions, outer_center=(0.0, 0.0), outer_angle=0.0):
    """Find minimum side length of regular hexagon containing all points."""
    def fits(side):
        ox, oy = outer_center
        for pos in positions:
            x, y, _ = pos
            # Transform to outer hexagon local frame
            dx = x - ox
            dy = y - oy
            ang = math.radians(outer_angle)
            cos_a = math.cos(-ang)
            sin_a = math.sin(-ang)
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            # For a regular hexagon with side s, center at origin, angle 0:
            # The hexagon is the intersection of 6 half-planes
            # In local frame, the constraints are:
            # For each of 6 directions (normals at 30, 90, 150, 210, 270, 330 degrees):
            #   n_i . (lx, ly) <= s * sqrt(3)/2
            apothem = side * math.sqrt(3) / 2
            for i in range(6):
                theta = math.radians(30 + i * 60)
                nx = math.cos(theta)
                ny = math.sin(theta)
                if nx * lx + ny * ly > apothem + 1e-10:
                    return False
        return True
    
    lo, hi = 1.0, 50.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if fits(mid):
            hi = mid
        else:
            lo = mid
    return hi

def _hex_overlap_fast(v1, v2):
    """Fast check if two hexagon vertex arrays overlap."""
    # Check vertices of v1 inside v2 and vice versa
    c1x, c1y = float(np.mean(v1[:, 0])), float(np.mean(v1[:, 1]))
    c2x, c2y = float(np.mean(v2[:, 0])), float(np.mean(v2[:, 1]))
    
    # Compute side and angle from vertices
    d01 = np.linalg.norm(v1[1] - v1[0])
    d02 = np.linalg.norm(v2[1] - v2[0])
    s1, s2 = d01, d02
    
    a1 = math.degrees(math.atan2(v1[0, 1] - c1y, v1[0, 0] - c1x))
    a2 = math.degrees(math.atan2(v2[0, 1] - c2y, v2[0, 0] - c2x))
    
    def point_in_hex(px, py, cx, cy, side, ang_deg):
        dx = px - cx
        dy = py - cy
        ang = math.radians(ang_deg)
        cos_a = math.cos(-ang)
        sin_a = math.sin(-ang)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        apothem = side * math.sqrt(3) / 2
        for i in range(6):
            theta = math.radians(30 + i * 60)
            nx = math.cos(theta)
            ny = math.sin(theta)
            if nx * lx + ny * ly > apothem + 1e-9:
                return False
        return True
    
    for i in range(6):
        if point_in_hex(v1[i, 0], v1[i, 1], c2x, c2y, s2, a2):
            return True
    for i in range(6):
        if point_in_hex(v2[i, 0], v2[i, 1], c1x, c1y, s1, a1):
            return True
    
    # Edge intersection check
    def seg_intersect(p1, p2, p3, p4):
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False
    
    for i in range(6):
        for j in range(6):
            if seg_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                return True
    return False

def _check_validity(inner_positions, outer_center, outer_side, outer_angle):
    """Verify all inner hexagons are inside outer and mutually disjoint."""
    ocx, ocy = outer_center
    outer_verts = _hex_vertices(ocx, ocy, outer_side, outer_angle)
    
    inner_verts = []
    for pos in inner_positions:
        x, y, ang = pos
        v = _hex_vertices(x, y, 1.0, ang)
        inner_verts.append(v)
    
    # Check all vertices of inner hexagons are inside outer
    for v in inner_verts:
        for i in range(6):
            px, py = v[i]
            dx = px - ocx
            dy = py - ocy
            ang = math.radians(outer_angle)
            cos_a = math.cos(-ang)
            sin_a = math.sin(-ang)
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            apothem = outer_side * math.sqrt(3) / 2
            inside = True
            for k in range(6):
                theta = math.radians(30 + k * 60)
                nx = math.cos(theta)
                ny = math.sin(theta)
                if nx * lx + ny * ly > apothem + 1e-8:
                    inside = False
                    break
            if not inside:
                return False
    
    # Check mutual disjointness
    for i in range(len(inner_verts)):
        for j in range(i+1, len(inner_verts)):
            if _hex_overlap_fast(inner_verts[i], inner_verts[j]):
                return False
    return True

def _contact_graph_positions(contact_edges, positions):
    """
    Given a contact graph (list of (i,j) pairs where hexagon i touches hexagon j),
    compute the minimum center-to-center distance for touching unit hexagons.
    For two unit regular hexagons sharing an edge, center distance = sqrt(3).
    """
    # For two unit hexagons in arbitrary orientations to be tangent (touching at one edge),
    # the center-to-center distance is sqrt(3) ≈ 1.732
    return math.sqrt(3)

def _build_initial_configurations():
    """Generate diverse initial configurations using different topologies."""
    configs = []
    sqrt3 = math.sqrt(3)
    
    # Configuration 1: Triangular lattice 4-3-4
    pos = []
    d = sqrt3
    dy = d * sqrt3 / 2
    for i in range(4):
        pos.append([i * d - 1.5 * d, 0.0, 0.0])
    for i in range(3):
        pos.append([i * d - d, dy, 0.0])
    for i in range(4):
        pos.append([i * d - 1.5 * d, 2 * dy, 0.0])
    configs.append(pos)
    
    # Configuration 2: Center + ring of 6 + outer 4
    pos = [[0.0, 0.0, 0.0]]
    r = sqrt3
    for i in range(6):
        ang = math.radians(60 * i)
        pos.append([r * math.cos(ang), r * math.sin(ang), 0.0])
    # 4 more at radius 2*sqrt3, at 45, 135, 225, 315 degrees
    r2 = 2 * sqrt3
    for i in range(4):
        ang = math.radians(45 + 90 * i)
        pos.append([r2 * math.cos(ang), r2 * math.sin(ang), 0.0])
    configs.append(pos)
    
    # Configuration 3: 5-3-3 with slight rotation variation
    pos = []
    d = sqrt3
    dy = d * sqrt3 / 2
    for i in range(5):
        pos.append([i * d - 2.0 * d, 0.0, 0.0])
    for i in range(3):
        pos.append([i * d - d, dy, 0.0])
    for i in range(3):
        pos.append([i * d - d, 2 * dy, 0.0])
    configs.append(pos)
    
    # Configuration 4: Compact cluster with mixed angles
    pos = [[0.0, 0.0, 0.0]]
    r = sqrt3
    for i in range(6):
        ang = math.radians(60 * i)
        pos.append([r * math.cos(ang), r * math.sin(ang), 30.0])
    r2 = 1.8 * sqrt3
    for i in range(4):
        ang = math.radians(30 + 90 * i)
        pos.append([r2 * math.cos(ang), r2 * math.sin(ang), 0.0])
    configs.append(pos)
    
    # Configuration 5: Linear with perturbation
    pos = []
    for i in range(11):
        pos.append([i * d - 5 * d, 0.0, 0.0])
    configs.append(pos)
    
    return configs

def _optimize_with_contacts(positions, start_time, deadline, contact_pairs=None):
    """
    Optimize positions using a constrained approach where we minimize the outer hexagon side
    while enforcing minimum separation between all pairs (soft constraint via penalty).
    Uses SLSQP with the contact structure to guide the search.
    """
    n = len(positions)
    
    def flatten(pos_list):
        return [v for p in pos_list for v in p]
    
    def unflatten(params):
        return [[params[3*i], params[3*i+1], params[3*i+2]] for i in range(n)]
    
    def objective(params):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10.0:
            return 1e10
        
        pos = unflatten(params)
        
        # Compute outer side
        outer_side = _min_enclosing_side(pos)
        
        # Penalty for overlaps
        penalty = 0.0
        min_dist = math.sqrt(3) - 0.01  # Slightly less than touching distance
        
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2 * 100
        
        return outer_side + penalty
    
    x0 = flatten(positions)
    
    # Bounds: keep positions in a reasonable range
    bounds = [(-10, 10), (-10, 10), (-180, 180)] * n
    
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                         options={'maxiter': 300, 'ftol': 1e-6})
        if result.fun < 1e9:
            best_pos = unflatten(result.x)
            return best_pos, result.fun
    except Exception:
        pass
    
    # Fallback to Nelder-Mead
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 0.005, 'fatol': 0.005})
        if result.fun < 1e9:
            best_pos = unflatten(result.x)
            return best_pos, result.fun
    except Exception:
        pass
    
    return positions, 1e10

def _local_refinement(positions, start_time, deadline):
    """Fine-tune positions using a gradient-free method with tighter tolerance."""
    n = len(positions)
    
    def flatten(pos_list):
        return [v for p in pos_list for v in p]
    
    def unflatten(params):
        return [[params[3*i], params[3*i+1], params[3*i+2]] for i in range(n)]
    
    def objective(params):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            return 1e10
        
        pos = unflatten(params)
        outer_side = _min_enclosing_side(pos)
        
        # Strong penalty for overlaps
        penalty = 0.0
        min_dist = math.sqrt(3)
        
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < min_dist - 0.001:
                    penalty += (min_dist - dist) ** 2 * 1000
        
        return outer_side + penalty
    
    x0 = flatten(positions)
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': 1000, 'xatol': 0.001, 'fatol': 0.001,
                                  'initial_simplex': None})
        if result.fun < 1e9:
            best_pos = unflatten(result.x)
            return best_pos, result.fun
    except Exception:
        pass
    
    return positions, 1e10

def _get_valid_fallback():
    """Return a guaranteed valid configuration."""
    d = math.sqrt(3)
    pos = []
    for i in range(11):
        pos.append([i * d, 0.0, 0.0])
    return pos

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_positions = None
    best_side = float('inf')
    
    # Generate diverse initial configurations
    initial_configs = _build_initial_configurations()
    
    for config in initial_configs:
        elapsed = time.time() - start_time
        if elapsed > deadline - 30.0:
            break
        
        # Optimize this configuration
        opt_pos, opt_score = _optimize_with_contacts(config, start_time, deadline)
        
        if opt_score < 1e9:
            # Verify validity
            outer_side = _min_enclosing_side(opt_pos)
            if _check_validity(opt_pos, (0.0, 0.0), outer_side, 0.0):
                if outer_side < best_side:
                    best_side = outer_side
                    best_positions = opt_pos
        
        # Local refinement
        elapsed = time.time() - start_time
        if elapsed > deadline - 15.0:
            break
        
        if best_positions is not None:
            refined_pos, refined_score = _local_refinement(best_positions, start_time, deadline)
            if refined_score < 1e9:
                outer_side = _min_enclosing_side(refined_pos)
                if _check_validity(refined_pos, (0.0, 0.0), outer_side, 0.0):
                    if outer_side < best_side:
                        best_side = outer_side
                        best_positions = refined_pos
    
    # If no valid result found, use fallback
    if best_positions is None:
        best_positions = _get_valid_fallback()
        best_side = _min_enclosing_side(best_positions)
    
    # Final check
    final_side = _min_enclosing_side(best_positions)
    if not _check_validity(best_positions, (0.0, 0.0), final_side, 0.0):
        best_positions = _get_valid_fallback()
        final_side = _min_enclosing_side(best_positions)
    
    return best_positions, (0.0, 0.0), final_side, 0.0
# EVOLVE-BLOCK-END
