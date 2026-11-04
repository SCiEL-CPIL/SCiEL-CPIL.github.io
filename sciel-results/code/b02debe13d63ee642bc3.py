"""
Optimized packing of 12 unit hexagons in a minimal outer hexagon.
Structural change: Model the 12 hexagons as a rigid cluster (fixed relative positions)
and optimize only the cluster's global rotation and the outer hexagon's parameters.
This reduces the 36-dimensional non-convex problem to a low-dimensional smooth one.
We use a grid search over cluster rotations combined with analytic computation of
the minimal enclosing hexagon for each rotation.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize_scalar

N = 12
TIME_LIMIT = 140.0  # Safety margin below 150s

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        verts.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return verts

def _compute_minimal_hexagon_side(points, outer_rotation_deg):
    """
    Compute the side length of the smallest regular hexagon centered at the origin
    with given rotation that encloses all given points.
    """
    a = math.radians(outer_rotation_deg)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    max_need = 0.0
    for (vx, vy) in points:
        for (nx, ny) in normals:
            val = (nx * vx + ny * vy) / apothem_unit
            if val > max_need:
                max_need = val
    return max_need

def _rigid_cluster_score(cluster_rotation_deg, base_positions, base_rotations):
    """
    Given a global cluster rotation and base positions/rotations (relative to cluster center),
    compute the minimal outer hexagon side length.
    The outer hexagon is centered at the centroid of the cluster and can rotate freely.
    We optimize the outer hexagon rotation for a given cluster rotation.
    """
    c = math.cos(math.radians(cluster_rotation_deg))
    s = math.sin(math.radians(cluster_rotation_deg))
    
    all_points = []
    for i in range(N):
        px, py, pr = base_positions[i], base_rotations[i]
        # Rotate position
        rx = c * px - s * py
        ry = s * px + c * py
        # Rotate hexagon orientation
        new_rot = pr + cluster_rotation_deg
        verts = _hex_vertices(rx, ry, new_rot)
        all_points.extend(verts)
    
    # Center the points at origin for symmetry (outer hexagon centered at centroid)
    cx = np.mean([p[0] for p in all_points])
    cy = np.mean([p[1] for p in all_points])
    centered_points = [(p[0] - cx, p[1] - cy) for p in all_points]
    
    # Optimize outer hexagon rotation
    def outer_obj(outer_rot):
        return _compute_minimal_hexagon_side(centered_points, outer_rot)
    
    # Grid search over outer rotation
    best_side = float('inf')
    best_outer_rot = 0.0
    for rot in np.arange(0, 60, 5):
        side = outer_obj(rot)
        if side < best_side:
            best_side = side
            best_outer_rot = rot
    
    # Refine
    res = minimize_scalar(outer_obj, bounds=(best_outer_rot - 5, best_outer_rot + 5), method='bounded', options={'xatol': 1e-4})
    if res.success and res.fun < best_side:
        best_side = res.fun
        best_outer_rot = res.x
    
    return best_side, best_outer_rot, (cx, cy)

def construct():
    t0 = time.time()
    
    # Define several candidate rigid cluster configurations.
    # Each is a list of (x, y, rot_deg) relative to the cluster center.
    
    def make_flower_cluster():
        """1 center + 6 ring + 5 outer ring"""
        pts = []
        pts.append((0.0, 0.0, 0.0))
        r1 = 2 * math.sqrt(3)
        for i in range(6):
            angle = math.radians(60 * i)
            x = r1 * math.cos(angle)
            y = r1 * math.sin(angle)
            pts.append((x, y, 0.0))
        r2 = 2 * math.sqrt(3) * 1.732
        angles = [30, 90, 150, 210, 270]
        for deg in angles:
            rad = math.radians(deg)
            x = r2 * math.cos(rad)
            y = r2 * math.sin(rad)
            pts.append((x, y, 0.0))
        return pts[:12]
    
    def make_lattice_cluster():
        """3x4 honeycomb-like arrangement"""
        pts = []
        row_height = math.sqrt(3)
        col_width = 2.0
        # Row 0: 4 hexagons
        for c in range(4):
            x = c * col_width - 1.5 * col_width
            y = -row_height
            pts.append((x, y, 0.0))
        # Row 1: 4 hexagons offset
        for c in range(4):
            x = c * col_width - 1.5 * col_width + col_width/2
            y = 0.0
            pts.append((x, y, 0.0))
        # Row 2: 4 hexagons
        for c in range(4):
            x = c * col_width - 1.5 * col_width
            y = row_height
            pts.append((x, y, 0.0))
        # Take first 12
        return pts[:12]
    
    def make_compact_cluster():
        """Tighter packing: 2 rows of 3 + 1 row of 6? No, 12 total."""
        pts = []
        # Try a 3x4 grid with hexagonal offset
        sqrt3 = math.sqrt(3)
        # Row 0: 4 hexes at y=0
        for i in range(4):
            x = i * 2.0 - 3.0
            pts.append((x, 0.0, 0.0))
        # Row 1: 4 hexes at y=sqrt(3), offset by 1.0
        for i in range(4):
            x = i * 2.0 - 3.0 + 1.0
            pts.append((x, sqrt3, 0.0))
        # Row 2: 4 hexes at y=2*sqrt(3)
        for i in range(4):
            x = i * 2.0 - 3.0
            pts.append((x, 2*sqrt3, 0.0))
        return pts[:12]
    
    def make_ring_cluster():
        """11 in a ring + 1 center"""
        pts = [(0.0, 0.0, 0.0)]
        r = 2 * math.sqrt(3)
        for i in range(11):
            angle = math.radians(360 * i / 11 + 15)
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            pts.append((x, y, 0.0))
        return pts
    
    def make_two_row_cluster():
        """Two staggered rows of 6"""
        pts = []
        sqrt3 = math.sqrt(3)
        for i in range(6):
            x = i * 2.0 - 5.0
            pts.append((x, 0.0, 0.0))
        for i in range(6):
            x = i * 2.0 - 5.0 + 1.0
            pts.append((x, sqrt3, 0.0))
        return pts
    
    def make_honeycomb_cluster():
        """Optimal honeycomb: 1+6+5 arrangement but tightened"""
        pts = []
        pts.append((0.0, 0.0, 0.0))
        d1 = math.sqrt(3) * 1.9  # Slightly less than 2*sqrt(3) to allow rotation
        for i in range(6):
            angle = math.radians(60 * i)
            x = d1 * math.cos(angle)
            y = d1 * math.sin(angle)
            pts.append((x, y, 0.0))
        d2 = d1 * 1.7
        for i in range(5):
            angle = math.radians(60 * i + 30)
            x = d2 * math.cos(angle)
            y = d2 * math.sin(angle)
            pts.append((x, y, 0.0))
        return pts[:12]
    
    clusters = [
        make_flower_cluster,
        make_lattice_cluster,
        make_compact_cluster,
        make_ring_cluster,
        make_two_row_cluster,
        make_honeycomb_cluster,
    ]
    
    best_side = float('inf')
    best_cluster_idx = 0
    best_cluster_rot = 0.0
    best_outer_rot = 0.0
    best_center = (0.0, 0.0)
    best_base_positions = None
    best_base_rotations = None
    
    for idx, cluster_func in enumerate(clusters):
        if time.time() - t0 > TIME_LIMIT - 20:
            break
        
        base_pts = cluster_func()
        if len(base_pts) != N:
            continue
        
        base_positions = np.array([(p[0], p[1]) for p in base_pts])
        base_rotations = np.array([p[2] for p in base_pts])
        
        # Center the cluster
        centroid = np.mean(base_positions, axis=0)
        base_positions -= centroid
        
        # Search over cluster rotation
        def cluster_obj(rot_deg):
            side, outer_rot, center = _rigid_cluster_score(rot_deg, base_positions, base_rotations)
            return side
        
        # Coarse grid search
        best_rot = 0.0
        best_s = float('inf')
        for rot in np.arange(0, 60, 5):
            s = cluster_obj(rot)
            if s < best_s:
                best_s = s
                best_rot = rot
        
        # Refine
        res = minimize_scalar(cluster_obj, bounds=(best_rot - 5, best_rot + 5), method='bounded', options={'xatol': 1e-3})
        if res.success and res.fun < best_s:
            best_s = res.fun
            best_rot = res.x
        
        if best_s < best_side:
            best_side = best_s
            best_cluster_idx = idx
            best_cluster_rot = best_rot
            # Get the best outer rot and center from the refined result
            _, best_outer_rot, best_center = _rigid_cluster_score(best_rot, base_positions, base_rotations)
            best_base_positions = base_positions.copy()
            best_base_rotations = base_rotations.copy()
    
    # If no valid configuration found (shouldn't happen), fallback
    if best_base_positions is None:
        base_pts = make_lattice_cluster()
        best_base_positions = np.array([(p[0], p[1]) for p in base_pts])
        best_base_rotations = np.array([p[2] for p in base_pts])
        centroid = np.mean(best_base_positions, axis=0)
        best_base_positions -= centroid
        best_cluster_rot = 0.0
        best_outer_rot = 0.0
        best_center = (0.0, 0.0)
        best_side = _rigid_cluster_score(0.0, best_base_positions, best_base_rotations)[0]
    
    # Construct final inner hexagons
    c = math.cos(math.radians(best_cluster_rot))
    s = math.sin(math.radians(best_cluster_rot))
    
    inner = []
    for i in range(N):
        px, py = best_base_positions[i]
        pr = best_base_rotations[i]
        # Rotate position
        rx = c * px - s * py
        ry = s * px + c * py
        # Rotate hexagon orientation
        new_rot = pr + best_cluster_rot
        inner.append([rx, ry, new_rot])
    
    # The outer hexagon is centered at best_center (which is the centroid of the rotated cluster)
    # But we centered the base positions at origin, so the centroid is at origin.
    # However, _rigid_cluster_score returns the centroid in the rotated frame.
    # Since we centered base_positions at origin, the centroid after rotation is still origin.
    # So best_center should be (0,0).
    
    return inner, (0.0, 0.0), best_side, best_outer_rot

# EVOLVE-BLOCK-START
# (The construct function is defined above)
# EVOLVE-BLOCK-END
