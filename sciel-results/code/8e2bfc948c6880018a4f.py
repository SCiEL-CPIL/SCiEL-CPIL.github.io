import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 12
START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill

def hexagon_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon with side s, center (cx,cy), rotation angle_deg."""
    a = np.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = a + k * np.pi / 3.0
        x = cx + s * np.cos(theta)
        y = cy + s * np.sin(theta)
        verts.append((x, y))
    return np.array(verts)

def get_honeycomb_cluster():
    """
    Returns a list of (x, y, angle) for 12 unit hexagons in a non-overlapping honeycomb arrangement.
    Arrangement: 1 center, 6 in first ring, 5 in second ring (partial).
    Distances:
    Center: (0,0)
    Ring 1: distance sqrt(3)
    Ring 2: distance 2*sqrt(3) for some, or sqrt(3)*sqrt(7) for others?
    Let's use a standard 3x3 grid of hexagons minus corners? No, 12 is specific.
    A compact shape:
    Row 0: 4 hexagons
    Row 1: 4 hexagons (shifted)
    Row 2: 4 hexagons
    This is a parallelogram.
    
    Better: A "flower" of 7 (1+6) plus 5 more attached to the outer ring to make it more circular.
    Or simply a 3x4 rectangular grid of hexagons.
    Hexagon side 1.
    Horizontal spacing: 3/2 = 1.5? No, distance between centers in same row is 3/2 * side * 2? 
    Distance between centers of touching hexagons is 2 * apothem = sqrt(3).
    If we align them in a grid:
    dx = 3/2 * 1 * 2? No.
    Let's use the standard hexagonal lattice vectors.
    u = (sqrt(3), 0)
    v = (sqrt(3)/2, 3/2)
    
    Coordinates (i, j) in lattice:
    pos = i*u + j*v
    We need 12 points.
    i in -1..1, j in -1..1 is 9 points.
    Add 3 more?
    
    Let's just use a 3x4 grid:
    i = 0, 1, 2
    j = 0, 1, 2, 3
    Total 12.
    This shape is a parallelogram.
    """
    pts = []
    # Lattice vectors for unit hexagons (side 1)
    # Distance between centers is sqrt(3)
    dx = 3.0 / 2.0 # Horizontal distance between centers in same row? 
    # Actually, for flat-topped hexagons:
    # Width = 2, Height = sqrt(3).
    # Horizontal spacing = 3/2. Vertical spacing = sqrt(3).
    # Let's use pointy-topped for easier math? No, let's stick to flat-topped.
    # Center (0,0).
    # Neighbor right: (1.5, 0.866)? No.
    # For flat top:
    # Vertices at angles 0, 60, 120, 180, 240, 300.
    # Apothem = sqrt(3)/2.
    # Distance between centers of horizontally adjacent hexagons = 2 * apothem = sqrt(3).
    # Wait, if flat top, the "width" is 2 (vertex to vertex). The "height" is sqrt(3) (flat to flat).
    # If we pack them, we can place them in rows.
    # Row 0: y=0. x = 0, sqrt(3), 2sqrt(3)...
    # Row 1: y=sqrt(3)/2? No, vertical distance is sqrt(3)/2?
    # Let's verify.
    # Hex 1 at (0,0). Vertices: (1,0), (0.5, 0.866), (-0.5, 0.866), (-1,0), (-0.5, -0.866), (0.5, -0.866).
    # Hex 2 at (1.5, 0.866).
    # Distance between (0.5, 0.866) and (0.5, 0.866) is 0? They share a vertex? No, they share an edge.
    # Edge from (0.5, 0.866) to (1, 0) for Hex 1.
    # Edge from (0.5, 0.866) to (1, 0) for Hex 2?
    # Hex 2 center (1.5, 0.866). Vertices:
    # (1.5+1, 0.866) = (2.5, 0.866)
    # (1.5+0.5, 0.866+0.866) = (2, 1.732)
    # (1.5-0.5, 0.866+0.866) = (1, 1.732)
    # (1.5-1, 0.866) = (0.5, 0.866)
    # (1.5-0.5, 0.866-0.866) = (1, 0)
    # (1.5+0.5, 0.866-0.866) = (2, 0)
    # Yes, Hex 1 and Hex 2 share edge (0.5, 0.866)-(1,0).
    # So dx = 1.5, dy = 0.866 (sqrt(3)/2).
    
    u = np.array([1.5, np.sqrt(3)/2])
    v = np.array([1.5, -np.sqrt(3)/2])
    
    # 3x4 grid
    for i in range(3):
        for j in range(4):
            pos = i * u + j * v
            pts.append((pos[0], pos[1], 0.0))
            
    return pts

def calc_min_enclosing_hexagon(points, outer_angle_deg):
    """
    Calculate the minimum side length and center of a regular hexagon with orientation outer_angle_deg
    that encloses all points.
    """
    n_pts = len(points)
    if n_pts == 0:
        return 0.0, 0.0, 0.0
        
    a = np.radians(outer_angle_deg)
    normals = np.array([[np.cos(a + k * np.pi / 3.0), np.sin(a + k * np.pi / 3.0)] for k in range(6)])
    
    # Project points onto normals
    proj = points @ normals.T # (N, 6)
    min_proj = proj.min(axis=0) # (6,)
    max_proj = proj.max(axis=0) # (6,)
    
    # Widths
    widths = max_proj - min_proj # (6,)
    
    # Required apothem is max(width / 2)
    apothem = np.max(widths) / 2.0
    side = 2.0 * apothem / np.sqrt(3.0)
    
    # Find center (cx, cy)
    # We want cx * nx + cy * ny = mid_proj[k] for all k
    # mid_proj = (min + max) / 2
    mid_proj = (min_proj + max_proj) / 2.0
    
    # Least squares solution for [cx, cy]
    # A * [cx, cy]^T = b
    A = normals # (6, 2)
    b = mid_proj # (6,)
    
    try:
        res = np.linalg.lstsq(A, b, rcond=None)
        cx, cy = res[0]
    except:
        cx, cy = 0.0, 0.0
        
    return side, cx, cy

def construct():
    global START_TIME
    
    # 1. Generate fixed cluster
    cluster = get_honeycomb_cluster()
    cluster_arr = np.array(cluster)
    
    best_side = np.inf
    best_rot = 0.0
    best_center = (0.0, 0.0)
    
    # 2. Search over rotation angle of the cluster
    # The outer hexagon orientation is fixed to 0 degrees (or we can fix cluster and rotate outer, equivalent).
    # Let's fix outer hexagon to 0 degrees and rotate the cluster.
    # Symmetry is 60 degrees.
    
    # Coarse search
    angles = np.arange(0, 60, 1.0)
    for ang in angles:
        rad = np.radians(ang)
        c, s = np.cos(rad), np.sin(rad)
        R = np.array([[c, -s], [s, c]])
        
        # Rotate cluster centers
        centers = cluster_arr[:, :2]
        rotated_centers = centers @ R.T
        
        # The inner hexagons also rotate. But for the enclosing hexagon size, 
        # the orientation of the inner hexagons relative to the outer matters.
        # If we rotate the whole cluster by ang, the inner hexagons are also rotated by ang.
        # So we need to compute vertices of rotated hexagons.
        
        all_pts = []
        for i in range(N):
            cx, cy = rotated_centers[i]
            # Inner hexagon angle is ang (since we rotated the whole cluster)
            v = hexagon_vertices(cx, cy, 1.0, ang)
            all_pts.append(v)
            
        pts = np.vstack(all_pts)
        
        # Calculate min enclosing hexagon with fixed orientation 0
        side, cx, cy = calc_min_enclosing_hexagon(pts, 0.0)
        
        if side < best_side:
            best_side = side
            best_rot = ang
            best_center = (cx, cy)
            
    # Fine search around best_rot
    if best_rot > 0 and best_rot < 60:
        fine_angles = np.arange(best_rot - 1.0, best_rot + 1.0, 0.1)
    else:
        fine_angles = np.arange(0, 1.0, 0.1)
        
    for ang in fine_angles:
        rad = np.radians(ang)
        c, s = np.cos(rad), np.sin(rad)
        R = np.array([[c, -s], [s, c]])
        
        centers = cluster_arr[:, :2]
        rotated_centers = centers @ R.T
        
        all_pts = []
        for i in range(N):
            cx, cy = rotated_centers[i]
            v = hexagon_vertices(cx, cy, 1.0, ang)
            all_pts.append(v)
            
        pts = np.vstack(all_pts)
        
        side, cx, cy = calc_min_enclosing_hexagon(pts, 0.0)
        
        if side < best_side:
            best_side = side
            best_rot = ang
            best_center = (cx, cy)

    # 3. Construct final result
    # The outer hexagon has center best_center, side best_side, angle 0.0
    # The inner hexagons are the rotated cluster.
    
    rad = np.radians(best_rot)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[c, -s], [s, c]])
    
    centers = cluster_arr[:, :2]
    rotated_centers = centers @ R.T
    
    inner_list = []
    for i in range(N):
        inner_list.append([float(rotated_centers[i, 0]), float(rotated_centers[i, 1]), float(best_rot)])
        
    return inner_list, (float(best_center[0]), float(best_center[1])), float(best_side), 0.0
