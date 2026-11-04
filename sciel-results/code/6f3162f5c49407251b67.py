import numpy as np
import time

N = 12
TIME_LIMIT = 140.0  # Safety margin for 150s limit

def _get_cluster_vertices():
    """
    Generate vertices for a 3x4 block of unit hexagons.
    The block is arranged in a honeycomb lattice.
    Hexagon i is at (row, col) in the block.
    """
    # Honeycomb lattice vectors
    # For pointy-top hexagons (vertices at 0, 60, ...):
    # Width = sqrt(3), Height = 2
    # Horizontal spacing = sqrt(3)
    # Vertical spacing = 1.5
    # Odd rows shifted by sqrt(3)/2
    
    centers = []
    for r in range(3):
        for c in range(4):
            x = c * np.sqrt(3) + (r % 2) * (np.sqrt(3) / 2)
            y = r * 1.5
            centers.append((x, y))
            
    # Center the cluster at (0,0) for symmetry
    cx = np.mean([c[0] for c in centers])
    cy = np.mean([c[1] for c in centers])
    centers = [(x - cx, y - cy) for x, y in centers]
    
    # Generate vertices for each hexagon
    # Hexagon side = 1
    # Vertices at angles 0, 60, 120, 180, 240, 300 relative to center
    # This is "flat-top" if we rotate by 30, or "pointy-top" if 0.
    # Let's use standard flat-top: vertices at 0, 60... is actually pointy-top?
    # cos(0)=1, sin(0)=0 -> (1,0). This is a vertex on the right.
    # So the hexagon has a vertex at (1,0). This is "pointy" on the right/left.
    # The sides are flat on top/bottom?
    # Distance from center to vertex = 1.
    # Distance from center to side midpoint = cos(30) = sqrt(3)/2.
    
    all_verts = []
    for cx, cy in centers:
        for k in range(6):
            theta = np.pi / 3 * k
            vx = cx + np.cos(theta)
            vy = cy + np.sin(theta)
            all_verts.append([vx, vy])
            
    return np.array(all_verts)

def _min_enclosing_hex_side(verts, angle_deg):
    """
    Calculate the minimum side length of a regular hexagon centered at origin
    with rotation angle_deg that contains all points in 'verts'.
    
    A regular hexagon with side S and rotation alpha has 6 supporting lines
    with normals at angles alpha + k*pi/3.
    The distance from origin to each supporting line is the apothem A = S * sqrt(3) / 2.
    
    For the hexagon to contain the points, for each normal direction n_k,
    the maximum projection of any point onto n_k must be <= A.
    
    So A_min = max_k ( max_p ( dot(p, n_k) ) )
    And S_min = A_min * 2 / sqrt(3)
    
    Note: We must also consider the negative side?
    A regular hexagon is centrally symmetric.
    The condition is |dot(p, n_k)| <= A for all k?
    Actually, the hexagon is the intersection of 6 slabs.
    The slabs are defined by -A <= dot(p, n_k) <= A.
    So yes, we need max_p |dot(p, n_k)| <= A.
    """
    alpha = np.deg2rad(angle_deg)
    normals = []
    for k in range(6):
        theta = alpha + k * np.pi / 3
        normals.append([np.cos(theta), np.sin(theta)])
    normals = np.array(normals)
    
    # Project all vertices onto all normals
    # verts: (M, 2)
    # normals: (6, 2)
    # projections: (M, 6)
    projections = verts @ normals.T
    
    # Find max absolute projection for each normal
    max_abs_proj = np.max(np.abs(projections), axis=0)
    
    # The required apothem is the maximum of these
    apothem = np.max(max_abs_proj)
    
    # Side length
    side = apothem * 2.0 / np.sqrt(3.0)
    
    return side

def construct():
    start_time = time.time()
    
    # Pre-compute the rigid cluster vertices
    cluster_verts = _get_cluster_vertices()
    
    # We need to find the optimal relative rotation between the cluster and the outer hexagon.
    # The outer hexagon rotation is a global rotation. The cluster rotation is relative.
    # Since the system is rotationally symmetric, we can fix the outer hexagon to angle 0
    # and rotate the cluster, or vice versa.
    # Let's fix outer hexagon at 0 degrees and rotate the cluster from 0 to 60 degrees.
    # Due to symmetry of the hexagon (60 deg) and the cluster (likely no full symmetry, but let's check),
    # we only need to search 0 to 60 degrees.
    
    # Actually, the cluster is a 3x4 block. It does not have 60-degree symmetry.
    # It might have 180-degree symmetry (point symmetry).
    # Let's do a fine scan over 0 to 60 degrees.
    
    best_side = float('inf')
    best_angle = 0.0
    
    # Coarse scan
    num_coarse = 100
    angles = np.linspace(0, 60, num_coarse)
    
    for ang in angles:
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        side = _min_enclosing_hex_side(cluster_verts, ang)
        if side < best_side:
            best_side = side
            best_angle = ang
            
    # Fine scan around best
    if best_angle > 0 and best_angle < 60:
        num_fine = 100
        angles_fine = np.linspace(best_angle - 60/num_coarse, best_angle + 60/num_coarse, num_fine)
        for ang in angles_fine:
            if time.time() - start_time > TIME_LIMIT - 2:
                break
            side = _min_enclosing_hex_side(cluster_verts, ang)
            if side < best_side:
                best_side = side
                best_angle = ang

    # Construct the result
    # The inner hexagons are at the centers defined in _get_cluster_vertices, rotated by best_angle.
    # Wait, _get_cluster_vertices returns vertices. I need centers.
    # Let's re-derive centers or store them.
    
    # Re-derive centers for output
    centers_orig = []
    for r in range(3):
        for c in range(4):
            x = c * np.sqrt(3) + (r % 2) * (np.sqrt(3) / 2)
            y = r * 1.5
            centers_orig.append((x, y))
            
    cx = np.mean([c[0] for c in centers_orig])
    cy = np.mean([c[1] for c in centers_orig])
    centers_orig = [(x - cx, y - cy) for x, y in centers_orig]
    
    theta = np.deg2rad(best_angle)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    inner_hexes = []
    for x, y in centers_orig:
        # Rotate center
        rx = x * cos_t - y * sin_t
        ry = x * sin_t + y * cos_t
        # The inner hexagon angle is also rotated by best_angle
        # The base hexagon had vertices at 0, 60...
        # So the angle is best_angle
        inner_hexes.append([rx, ry, best_angle])
        
    # Outer hexagon
    # Center at (0,0)
    # Side length is best_side
    # Angle is 0 (we rotated the cluster to fit a 0-degree outer hex)
    outer_center = [0.0, 0.0]
    outer_side = best_side
    outer_angle = 0.0
    
    return (inner_hexes, outer_center, outer_side, outer_angle)
