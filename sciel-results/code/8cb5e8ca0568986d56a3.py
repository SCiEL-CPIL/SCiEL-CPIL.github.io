"""
Hexagon packing: 11 unit hexagons inside a regular hexagon.
Strategy: Rigid-body optimization.
We assume the 11 hexagons form a rigid cluster (specifically, a central hexagon 
surrounded by 6, plus 4 more from the next ring, arranged symmetrically).
We optimize the global rotation of this cluster and the orientation of the outer hexagon.
This reduces the search space significantly compared to optimizing 33 variables independently.
"""
import math
import time
import itertools
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
DEADLINE_MARGIN = 10.0  # seconds before the 150s limit

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for k in range(6):
        a = math.radians(angle_deg + 60 * k)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts

def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon."""
    dx = px - cx
    dy = py - cy
    a = -math.radians(angle_deg)
    lx = dx * math.cos(a) - dy * math.sin(a)
    ly = dx * math.sin(a) + dy * math.cos(a)
    apothem = side * math.sqrt(3) / 2
    for k in range(6):
        a_norm = math.radians(30 + 60 * k)
        nx = math.cos(a_norm)
        ny = math.sin(a_norm)
        if lx * nx + ly * ny > apothem + 1e-9:
            return False
    return True

def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 properly intersects segment p3-p4."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def _hexagons_disjoint(hex1, hex2, eps=1e-9):
    """Check if two hexagons have disjoint interiors."""
    cx1, cy1, s1, a1 = hex1
    cx2, cy2, s2, a2 = hex2
    r1 = s1
    r2 = s2
    dx = cx2 - cx1
    dy = cy2 - cy1
    if dx * dx + dy * dy > (r1 + r2) ** 2 + 1e-12:
        return True
    verts1 = _hex_vertices(cx1, cy1, s1, a1)
    for vx, vy in verts1:
        if _point_in_hex(vx, vy, cx2, cy2, s2, a2):
            return False
    verts2 = _hex_vertices(cx2, cy2, s2, a2)
    for vx, vy in verts2:
        if _point_in_hex(vx, vy, cx1, cy1, s1, a1):
            return False
    for i in range(6):
        v1a = verts1[i]
        v1b = verts1[(i + 1) % 6]
        for j in range(6):
            v2a = verts2[j]
            v2b = verts2[(j + 1) % 6]
            if _segments_intersect(v1a, v1b, v2a, v2b):
                return False
    return True

def _all_disjoint(inner_hexes, eps=1e-9):
    """Check if all pairs of hexagons are disjoint."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _hexagons_disjoint(inner_hexes[i], inner_hexes[j], eps):
                return False
    return True

def _inner_in_outer(inner_hex, outer_hex):
    """Check if inner hexagon is fully inside outer hexagon."""
    cx, cy, s, a = inner_hex
    ox, oy, oside, oangle = outer_hex
    verts = _hex_vertices(cx, cy, s, a)
    for vx, vy in verts:
        if not _point_in_hex(vx, vy, ox, oy, oside, oangle):
            return False
    return True

def _compute_outer_lp(positions, angles, oangle=0.0):
    """
    Find the minimum side length and center of an outer hexagon with orientation oangle
    that contains all inner hexagons defined by positions and angles.
    Returns: (center_x, center_y, side_length)
    """
    n = len(positions)
    c = [0, 0, 1] # minimize t (apothem)
    A_ub = []
    b_ub = []
    
    # Precompute normals for the outer hexagon
    normals = []
    for k in range(6):
        theta = math.radians(oangle + 30 + 60 * k)
        normals.append((math.cos(theta), math.sin(theta)))
    
    for i in range(n):
        cx_h, cy_h = positions[i]
        ang_h = angles[i]
        verts = _hex_vertices(cx_h, cy_h, 1.0, ang_h)
        for vx, vy in verts:
            for nx, ny in normals:
                # (vx - cx)*nx + (vy - cy)*ny <= t
                # -nx*cx - ny*cy - t <= -vx*nx - vy*ny
                A_ub.append([-nx, -ny, -1])
                b_ub.append(-vx * nx - vy * ny)
    
    # t >= 0
    A_ub.append([0, 0, -1])
    b_ub.append(0)
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None), (0, None)], method='highs')
    
    if res.success:
        cx = res.x[0]
        cy = res.x[1]
        t = res.x[2]
        side = t * 2.0 / math.sqrt(3)
        return cx, cy, side
    else:
        # Fallback: centroid and large side
        cx = sum(p[0] for p in positions) / n
        cy = sum(p[1] for p in positions) / n
        return cx, cy, float('inf')

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    sqrt3 = math.sqrt(3)
    
    # Define the rigid cluster configuration in axial coordinates
    # Center: (0,0)
    # Ring 1: 6 hexagons
    # Ring 2: We need 4 more. To maintain symmetry and compactness, 
    # we pick 4 from ring 2 that form a stable shape.
    # Common optimal packing for 11 is often related to the "flower" + 4 corners or similar.
    # Let's use the standard hexagonal grid coordinates.
    
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return (x, y)
    
    # Base positions (axial)
    base_axial = [(0, 0)]
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    base_axial.extend(ring1)
    
    # Ring 2 candidates. We want to pick 4 that are "tight".
    # The 12 positions in ring 2.
    # Let's try a few symmetric configurations of 4 extra hexagons.
    # Config 1: 4 hexagons at "corners" of the hexagon (indices 0, 3, 6, 9 in a 12-element ring?)
    # Ring 2 axial coords:
    ring2_all = [
        (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
        (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
        (0, 2), (1, 2)
    ]
    # Note: The standard ring 2 has 12 hexagons. The list above has 14? 
    # Let's generate ring 2 properly.
    ring2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2 and abs(q) <= 2 and abs(r) <= 2:
                if abs(q) + abs(r) + abs(q+r) == 4: # Distance 2
                    ring2.append((q, r))
    # Deduplicate and sort for consistency
    ring2 = sorted(set(ring2))
    # There should be 12 hexagons in ring 2.
    
    # We will try different sets of 4 from ring 2.
    # To save time, we pre-select a few promising symmetric sets.
    # Set A: (2,0), (0,2), (-2,2), (0,-2) -> Not symmetric
    # Let's just iterate through combinations of 4 from ring 2 if time permits, 
    # but start with a few known good ones or just one symmetric one.
    # For robustness, let's pick the 4 that are furthest apart in angular distribution.
    
    # Actually, the most compact packing of 11 hexagons is known to be the "flower" (7) + 4 from the next layer.
    # The 4 should be placed at 4 of the 6 "corners" of the 7-hexagon flower? No, the flower is a hexagon of side 2.
    # The 11 hexagons usually form a shape close to a hexagon of side 2, but with 2 missing from the corners?
    # A hexagon of side 2 has 7 + 12 = 19 hexagons.
    # A hexagon of side 1 has 7.
    # 11 is 7 + 4.
    # The 4 extra should be placed to minimize the bounding box.
    # Placing them at 4 of the 6 "directions" of the ring 2 seems good.
    
    # Let's define a few candidate sets of 4 indices from ring 2.
    # Index 0: (2,0) - East
    # Index 3: (1,-2) - South-East
    # Index 6: (-1,-2) - South-West
    # Index 9: (-2,0) - West
    # This set {(2,0), (1,-2), (-1,-2), (-2,0)} is symmetric (180 deg).
    
    candidate_sets = [
        [0, 3, 6, 9], # Symmetric horizontal
        [0, 2, 6, 8], # Another symmetric set?
        [1, 4, 7, 10], # Rotated
    ]
    
    best_side = float('inf')
    best_config = None
    best_global_rot = 0.0
    best_oangle = 0.0
    best_center = (0.0, 0.0)
    
    # Pre-compute xy for base
    base_xy = [axial_to_xy(q, r) for q, r in base_axial]
    
    # Search over candidate sets of 4 extra hexagons
    for indices in candidate_sets:
        if time.time() > deadline:
            break
            
        extra_axial = [ring2[i] for i in indices]
        full_axial = base_axial + extra_axial
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]
        
        # Now optimize global rotation of this rigid cluster and outer hexagon orientation.
        # Variables: [global_rot, oangle]
        # We can fix the cluster center at (0,0) by subtracting the mean, or let the LP handle translation.
        # The LP handles translation automatically.
        
        def objective_rigid(params):
            g_rot = params[0]
            o_ang = params[1]
            
            # Rotate all positions around (0,0) by g_rot
            cos_g = math.cos(math.radians(g_rot))
            sin_g = math.sin(math.radians(g_rot))
            
            rotated_positions = []
            for x, y in full_xy:
                rx = x * cos_g - y * sin_g
                ry = x * sin_g + y * cos_g
                rotated_positions.append((rx, ry))
            
            # All inner hexagons rotate by g_rot as well
            rotated_angles = [g_rot] * 11
            
            # Compute optimal outer hexagon
            cx, cy, side = _compute_outer_lp(rotated_positions, rotated_angles, o_ang)
            
            if not math.isfinite(side):
                return 1e6
                
            return side
        
        # Initial guess: g_rot = 0, o_ang = 0
        # We can try a few starting points for o_ang
        best_local_side = float('inf')
        best_local_params = None
        
        for o_init in [0.0, 15.0, 30.0]:
            if time.time() > deadline:
                break
            try:
                res = minimize(objective_rigid, [0.0, o_init], method='Nelder-Mead', 
                               options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
                if res.fun < best_local_side:
                    best_local_side = res.fun
                    best_local_params = res.x
            except Exception:
                pass
        
        if best_local_params is not None and best_local_side < best_side:
            best_side = best_local_side
            best_global_rot = best_local_params[0]
            best_oangle = best_local_params[1]
            best_config = full_xy[:] # Store the base config (unrotated)
            
    # If no rigid config found, fallback to simple lattice
    if best_config is None:
        best_config = base_xy[:7] # Just the flower
        best_side, best_global_rot, best_oangle = 3.0, 0.0, 0.0
        
    # Final Polish:
    # Now that we have a good rigid configuration, we can try to slightly "relax" the rigidity
    # or just re-verify and compute the final values precisely.
    # For now, we stick to the rigid body result as it is robust and fast.
    
    # Calculate final positions
    cos_g = math.cos(math.radians(best_global_rot))
    sin_g = math.sin(math.radians(best_global_rot))
    
    final_positions = []
    for x, y in best_config:
        rx = x * cos_g - y * sin_g
        ry = x * sin_g + y * cos_g
        final_positions.append((rx, ry))
    
    final_angles = [best_global_rot] * 11
    
    # Compute final outer hexagon
    cx, cy, side = _compute_outer_lp(final_positions, final_angles, best_oangle)
    
    # Ensure validity
    inner_hexes = [(final_positions[i][0], final_positions[i][1], 1.0, final_angles[i]) for i in range(11)]
    outer_hex = (cx, cy, side, best_oangle)
    
    if not _all_disjoint(inner_hexes):
        # Should not happen for rigid lattice, but check just in case
        return None # Or fallback
        
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            # Expand side slightly to ensure containment due to numerical precision
            side *= 1.001
            outer_hex = (cx, cy, side, best_oangle)
            break

    inner = [[inner_hexes[i][0], inner_hexes[i][1], inner_hexes[i][3]] for i in range(11)]
    
    return inner, (cx, cy), side, best_oangle

# Entry point
if __name__ == "__main__":
    construct()
