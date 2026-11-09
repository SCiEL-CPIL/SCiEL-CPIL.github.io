"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

def _hex_vertices(cx, cy, side, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = np.radians(angle_deg)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees (pointy-top by default)
    # For flat-top: offset by 30 degrees
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300])) + a
    # Actually, let's use the standard: vertices at angle a + k*60 for k=0..5
    # A regular hexagon of side s has vertices at distance s from center
    verts = np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])
    return verts

def _point_in_hexagon(px, py, cx, cy, side, angle_deg=0.0):
    """Check if point (px,py) is inside regular hexagon centered at (cx,cy)."""
    # Transform to hexagon's local frame
    dx, dy = px - cx, py - cy
    a = np.radians(angle_deg)
    ca, sa = np.cos(-a), np.sin(-a)
    lx = dx * ca - dy * sa
    ly = dx * sa + dy * ca
    # Regular hexagon with side s: |lx| <= s*sqrt(3)/2 and |ly| <= s and |ly| + |lx|/sqrt(3) <= s
    # More precisely, the hexagon is intersection of 6 half-planes
    # For a hexagon with vertices at angles 0,60,...: 
    # Normal directions at 30,90,150,210,270,330
    # Distance from center to each flat side = s * sqrt(3)/2
    # Condition: for each normal direction n_k, dot((lx,ly), n_k) <= s*sqrt(3)/2
    normals = np.array([[np.cos(np.radians(30)), np.sin(np.radians(30))],
                        [np.cos(np.radians(90)), np.sin(np.radians(90))],
                        [np.cos(np.radians(150)), np.sin(np.radians(150))],
                        [np.cos(np.radians(210)), np.sin(np.radians(210))],
                        [np.cos(np.radians(270)), np.sin(np.radians(270))],
                        [np.cos(np.radians(330)), np.sin(np.radians(330))]])
    dists = normals @ np.array([lx, ly])
    return np.all(dists <= side * np.sqrt(3) / 2 + 1e-9)

def _min_enclosing_hexagon(points):
    """Find minimal side length of a regular hexagon containing all points.
    We try all 6 orientations (0, 60, 120, 180, 240, 300) and for each, 
    find the minimal side and center."""
    best_side = float('inf')
    best_center = (0, 0)
    best_angle = 0.0
    
    for angle_deg in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
        a = np.radians(angle_deg)
        ca, sa = np.cos(-a), np.sin(-a)
        # Rotate points
        rx = points[:, 0] * ca - points[:, 1] * sa
        ry = points[:, 0] * sa + points[:, 1] * ca
        
        # For a regular hexagon with this orientation, the constraints are:
        # For each of 6 sides with normal n_k, max_k dot(p, n_k) - min_k dot(p, n_k) <= side * sqrt(3)
        # Wait, that's not quite right. Let me think again.
        
        # A regular hexagon centered at c with side s and orientation a has 6 half-plane constraints:
        # dot(p - c, n_k) <= s * sqrt(3)/2 for k=0..5
        # where n_k are the outward normals at angles a + 30 + k*60
        
        # To contain all points, we need:
        # For each k: max_i dot(p_i, n_k) - dot(c, n_k) <= s*sqrt(3)/2
        # So: dot(c, n_k) >= max_i dot(p_i, n_k) - s*sqrt(3)/2
        # And: dot(c, n_k) <= min_i dot(p_i, n_k) + s*sqrt(3)/2
        
        # This gives: max_i dot(p_i, n_k) - min_i dot(p_i, n_k) <= s*sqrt(3)
        # So s >= (max_i dot(p_i, n_k) - min_i dot(p_i, n_k)) / sqrt(3) for all k
        # Thus s_min = max_k (range_k) / sqrt(3)
        
        normals = np.array([[np.cos(np.radians(30 + k*60)), np.sin(np.radians(30 + k*60))] for k in range(6)])
        projections = rx * normals[:, 0] + ry * normals[:, 1]  # shape (6, n_points)
        ranges = projections.max(axis=1) - projections.min(axis=1)
        s_candidate = ranges.max() / np.sqrt(3)
        
        if s_candidate < best_side:
            best_side = s_candidate
            # Find center: average of the midpoint constraints
            # c should satisfy: for each k, dot(c, n_k) = (max_k + min_k)/2 (for tight fit)
            # Solve least squares
            midpoints = (projections.max(axis=1) + projections.min(axis=1)) / 2
            # normals @ c = midpoints
            c_local, _, _, _ = np.linalg.lstsq(normals, midpoints, rcond=None)
            # Rotate back
            cx = c_local[0] * np.cos(a) - c_local[1] * np.sin(a)
            cy = c_local[0] * np.sin(a) + c_local[1] * np.cos(a)
            best_center = (cx, cy)
            best_angle = angle_deg
    
    return best_side, best_center, best_angle

def _check_validity(inner, center, side, angle):
    """Check if all inner hexagons are disjoint and inside the outer hexagon."""
    cx, cy = center
    outer_verts = _hex_vertices(cx, cy, side, angle)
    
    # Check all inner hexagon vertices are inside outer hexagon
    for (ix, iy, iangle) in inner:
        iverts = _hex_vertices(ix, iy, 1.0, iangle)
        for v in iverts:
            if not _point_in_hexagon(v[0], v[1], cx, cy, side, angle):
                return False
    
    # Check disjointness: no two inner hexagons overlap
    # For convex polygons, check edge-edge intersection or vertex-in-polygon
    for i in range(len(inner)):
        for j in range(i+1, len(inner)):
            ix1, iy1, ia1 = inner[i]
            ix2, iy2, ia2 = inner[j]
            v1 = _hex_vertices(ix1, iy1, 1.0, ia1)
            v2 = _hex_vertices(ix2, iy2, 1.0, ia2)
            # Check if any vertex of v1 is inside v2 or vice versa
            for p in v1:
                if _point_in_hexagon(p[0], p[1], ix2, iy2, 1.0, ia2):
                    return False
            for p in v2:
                if _point_in_hexagon(p[0], p[1], ix1, iy1, 1.0, ia1):
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Strategy: Use a 4x3 grid arrangement on hexagonal lattice
    # This is known to be near-optimal for 12 hexagons
    
    sqrt3 = np.sqrt(3)
    
    # Generate hexagonal lattice positions for 4 columns x 3 rows
    # Hex lattice: odd rows offset by sqrt3/2 in x, 3/2 in y
    positions = []
    for row in range(3):
        for col in range(4):
            x = col * sqrt3 + (row % 2) * (sqrt3 / 2)
            y = row * 1.5
            positions.append((x, y, 0.0))
    
    # Center the arrangement
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx_mean = (max(xs) + min(xs)) / 2
    cy_mean = (max(ys) + min(ys)) / 2
    
    inner = [[x - cx_mean, y - cy_mean, ang] for (x, y, ang) in positions]
    
    # Get all vertices of all inner hexagons
    all_verts = []
    for (ix, iy, ia) in inner:
        verts = _hex_vertices(ix, iy, 1.0, ia)
        all_verts.extend(verts.tolist())
    all_verts = np.array(all_verts)
    
    # Find minimal enclosing regular hexagon
    side, center, angle = _min_enclosing_hexagon(all_verts)
    
    # Verify validity
    if not _check_validity(inner, center, side, angle):
        # If invalid, try to fix by slightly expanding
        for expansion in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
            if _check_validity(inner, center, side + expansion, angle):
                side += expansion
                break
    
    # Try to improve with local optimization
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        # Try different orientations for inner hexagons
        best_config = (inner, center, side, angle)
        best_score = side
        
        for trial in range(50):
            if time.time() > deadline - 10:
                break
            # Random small perturbations
            inner_trial = [[x + np.random.uniform(-0.05, 0.05),
                          y + np.random.uniform(-0.05, 0.05),
                          ang] for (x, y, ang) in inner]
            
            all_verts_trial = []
            for (ix, iy, ia) in inner_trial:
                verts = _hex_vertices(ix, iy, 1.0, ia)
                all_verts_trial.extend(verts.tolist())
            all_verts_trial = np.array(all_verts_trial)
            
            side_t, center_t, angle_t = _min_enclosing_hexagon(all_verts_trial)
            
            if side_t < best_score and _check_validity(inner_trial, center_t, side_t, angle_t):
                best_score = side_t
                best_config = (inner_trial, center_t, side_t, angle_t)
        
        inner, center, side, angle = best_config
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END

