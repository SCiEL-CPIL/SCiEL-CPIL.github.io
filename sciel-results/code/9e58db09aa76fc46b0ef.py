"""Optimized packing of 12 unit hexagons in minimal regular hexagon."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, angle_deg, side=1.0):
    """Get vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    a = np.radians(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    # Unit hexagon vertices (flat-top orientation at angle=0)
    base = np.array([[1, 0], [0.5, np.sqrt(3)/2], [-0.5, np.sqrt(3)/2],
                     [-1, 0], [-0.5, -np.sqrt(3)/2], [0.5, -np.sqrt(3)/2]])
    # Rotate
    R = np.array([[ca, -sa], [sa, ca]])
    verts = (base @ R.T) + np.array([cx, cy])
    return verts

def outer_hex_vertices(cx, cy, side, angle_deg):
    """Get vertices of outer regular hexagon."""
    a = np.radians(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    base = np.array([[side, 0], [side/2, side*np.sqrt(3)/2], [-side/2, side*np.sqrt(3)/2],
                     [-side, 0], [-side/2, -side*np.sqrt(3)/2], [side/2, -side*np.sqrt(3)/2]])
    R = np.array([[ca, -sa], [sa, ca]])
    verts = (base @ R.T) + np.array([cx, cy])
    return verts

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside regular hexagon centered at (cx,cy)."""
    # Transform to hexagon's local frame
    a = np.radians(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    dx, dy = px - cx, py - cy
    # Rotate point into hexagon frame
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    # In local frame, hexagon has vertices at angles 0, 60, 120, 180, 240, 300
    # The hexagon is defined by |lx| <= side, |ly| <= side*sqrt(3)/2, |lx/2 + ly*sqrt(3)/2| <= side
    # Actually for a regular hexagon with circumradius = side:
    # The six supporting lines are at distance side*cos(30) = side*sqrt(3)/2 from center
    # Let me use the proper containment test
    # A regular hexagon with circumradius R has the property that a point (x,y) is inside
    # iff max over k of (x*cos(60k) + y*sin(60k)) <= R
    R = side
    max_proj = 0
    for k in range(6):
        ang = np.radians(60 * k)
        proj = lx * np.cos(ang) + ly * np.sin(ang)
        max_proj = max(max_proj, abs(proj))
    return max_proj <= R + 1e-10

def all_vertices_inside(inner_hexes, outer_center, outer_side, outer_angle):
    """Check all vertices of all inner hexagons are inside outer hexagon."""
    ocx, ocy = outer_center
    for (ix, iy, iang) in inner_hexes:
        verts = hex_vertices(ix, iy, iang)
        for v in verts:
            if not point_in_hexagon(v[0], v[1], ocx, ocy, outer_side, outer_angle):
                return False
    return True

def max_violation(inner_hexes, outer_center, outer_side, outer_angle):
    """Return maximum distance by which any vertex violates the outer hexagon constraint."""
    ocx, ocy = outer_center
    max_v = 0.0
    for (ix, iy, iang) in inner_hexes:
        verts = hex_vertices(ix, iy, iang)
        for v in verts:
            a = np.radians(outer_angle)
            ca, sa = np.cos(a), np.sin(a)
            dx, dy = v[0] - ocx, v[1] - ocy
            lx = ca * dx + sa * dy
            ly = -sa * dx + ca * dy
            for k in range(6):
                ang = np.radians(60 * k)
                proj = abs(lx * np.cos(ang) + ly * np.sin(ang))
                if proj > outer_side:
                    max_v = max(max_v, proj - outer_side)
    return max_v

def compute_min_enclosing_hex(inner_hexes, outer_angle=0.0):
    """Compute minimal side length of enclosing regular hexagon with given rotation."""
    # Collect all vertices
    all_v = []
    for (ix, iy, iang) in inner_hexes:
        verts = hex_vertices(ix, iy, iang)
        all_v.extend(verts.tolist())
    all_v = np.array(all_v)
    
    # Center the points
    center = all_v.mean(axis=0)
    centered = all_v - center
    
    # For a regular hexagon with circumradius R and rotation angle,
    # the constraint is: for each vertex p, max_k |p · n_k| <= R
    # where n_k are the 6 unit normals at angles outer_angle + 60k
    # So R = max over all vertices and all k of |p · n_k|
    
    a = np.radians(outer_angle)
    max_R = 0
    for k in range(6):
        ang = a + np.radians(60 * k)
        nx, ny = np.cos(ang), np.sin(ang)
        projs = np.abs(centered[:, 0] * nx + centered[:, 1] * ny)
        max_R = max(max_R, np.max(projs))
    
    return max_R, center

def try_arrangement(positions, angles, outer_angle=0.0):
    """Try a specific arrangement and return the minimal outer side."""
    inner_hexes = [(positions[i][0], positions[i][1], angles[i]) for i in range(N)]
    R, center = compute_min_enclosing_hex(inner_hexes, outer_angle)
    return R, center, inner_hexes

def optimize_arrangement(init_positions, init_angles, outer_angle=0.0, max_iter=200):
    """Optimize positions and angles to minimize enclosing hexagon size."""
    x0 = np.concatenate([init_positions.flatten(), init_angles])
    
    def objective(x):
        pos = x[:2*N].reshape(N, 2)
        ang = x[2*N:]
        inner_hexes = [(pos[i,0], pos[i,1], ang[i]) for i in range(N)]
        R, _ = compute_min_enclosing_hex(inner_hexes, outer_angle)
        return R
    
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10})
    return result.x, result.fun

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_R = float('inf')
    best_result = None
    
    # Strategy 1: Grid-like arrangements
    # Hexagonal packing: rows offset
    # Try a 3x4 grid with hexagonal offset
    
    # Initial guess: hexagonal cluster arrangement
    # For 12 hexagons, a good starting arrangement:
    # Row 0: 4 hexagons
    # Row 1: 4 hexagons (offset)
    # Row 2: 4 hexagons
    
    # Hexagon spacing: for flat-top hexagons, horizontal spacing = 1.5*side = 1.5
    # Vertical spacing = sqrt(3)*side = sqrt(3)
    # Offset rows shift by 0.75 horizontally
    
    sqrt3 = np.sqrt(3)
    
    # Arrangement 1: 3 rows of 4
    positions1 = []
    angles1 = []
    for row in range(3):
        for col in range(4):
            x = col * 1.5 + (0.75 if row % 2 == 1 else 0)
            y = row * sqrt3
            positions1.append([x, y])
            angles1.append(0.0)
    
    # Arrangement 2: 4 rows of 3
    positions2 = []
    angles2 = []
    for row in range(4):
        for col in range(3):
            x = col * 1.5 + (0.75 if row % 2 == 1 else 0)
            y = row * sqrt3
            positions2.append([x, y])
            angles2.append(0.0)
    
    # Arrangement 3: Hexagonal cluster (center + rings)
    # 1 center, 6 in first ring, 5 more... actually 1+6+5=12
    # First ring at distance 2 (touching)
    positions3 = []
    angles3 = []
    # Center
    positions3.append([0, 0])
    angles3.append(0.0)
    # First ring: 6 hexagons at distance 2
    for k in range(6):
        ang = np.radians(60 * k)
        x = 2 * np.cos(ang)
        y = 2 * np.sin(ang)
        positions3.append([x, y])
        angles3.append(0.0)
    # Second ring partial: add 5 more at distance 4
    for k in range(5):
        ang = np.radians(60 * k + 30)  # offset
        x = 4 * np.cos(ang)
        y = 4 * np.sin(ang)
        positions3.append([x, y])
        angles3.append(0.0)
    
    arrangements = [
        (np.array(positions1), np.array(angles1)),
        (np.array(positions2), np.array(angles2)),
        (np.array(positions3), np.array(angles3)),
    ]
    
    # Try different outer rotations
    outer_angles = [0.0, 30.0, 15.0, 45.0]
    
    for arr_idx, (pos, ang) in enumerate(arrangements):
        if time.time() > deadline:
            break
        
        for oa in outer_angles:
            if time.time() > deadline:
                break
            
            # First, evaluate the initial arrangement
            R0, center0, inner0 = try_arrangement(pos.tolist(), ang.tolist(), oa)
            if R0 < best_R:
                best_R = R0
                best_result = (inner0, center0, R0, oa)
            
            # Optimize
            try:
                x_opt, R_opt = optimize_arrangement(pos, ang, oa, max_iter=300)
                if time.time() > deadline:
                    break
                pos_opt = x_opt[:2*N].reshape(N, 2)
                ang_opt = x_opt[2*N:]
                inner_opt = [(pos_opt[i,0], pos_opt[i,1], ang_opt[i]) for i in range(N)]
                R_final, center_final = compute_min_enclosing_hex(inner_opt, oa)
                if R_final < best_R:
                    best_R = R_final
                    best_result = (inner_opt, center_final, R_final, oa)
            except Exception:
                pass
    
    if best_result is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner, center, R, oa = best_result
    return inner, (center[0], center[1]), R, oa

# EVOLVE-BLOCK-END

