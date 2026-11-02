"""Radial Slicing formulation for packing 12 unit hexagons in a regular hexagon."""
import numpy as np
from scipy.optimize import minimize
import itertools

N = 12

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy) with side length and rotation."""
    theta = np.radians(angle_deg)
    # Regular hexagon vertices: for a regular hexagon with 'flat top' orientation at angle 0,
    # vertices are at angles 0, 60, 120, 180, 240, 300 degrees from center
    # But we need to be careful about orientation. Let's use the standard:
    # A regular hexagon with side s has circumradius = s.
    # Vertices at angles 0, 60, ..., 300 from center.
    angles = theta + np.linspace(0, 2 * np.pi, 7)[:-1]
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack([x, y])

def _outer_hexagon_support(cx, cy, side, angle_deg, direction):
    """
    Compute the support function of the outer hexagon in a given direction.
    The outer hexagon is centered at (cx, cy) with side length 'side' and rotation angle_deg.
    direction is a unit vector (dx, dy).
    Returns the maximum dot product of vertices with direction, i.e., the support value.
    """
    verts = _hex_vertices(cx, cy, side, angle_deg)
    return np.max(verts @ direction)

def _inner_hexagon_vertices_list(inner, center_offset=(0.0, 0.0)):
    """
    Get all vertices of all inner hexagons.
    inner: list of [x, y, angle_degrees]
    Returns list of arrays of vertices.
    """
    all_verts = []
    for row in inner:
        x, y, ang = row
        # Translate by center_offset if needed (but for support function, we work in absolute coords)
        verts = _hex_vertices(x, y, 1.0, ang)
        all_verts.append(verts)
    return all_verts

def _compute_min_outer_side(inner, outer_center, outer_angle):
    """
    Given inner hexagons (fixed positions/rotations), find the minimum side length of an outer
    regular hexagon centered at outer_center with rotation outer_angle that contains all inner hexagons.
    
    The outer hexagon with center (cx, cy), side s, rotation theta contains a point p if and only if
    for all 6 edge normals n_i, (p - c) · n_i <= s * apothem_factor... 
    
    Actually, let's think differently. A regular hexagon with side s has inradius (apothem) = s * sqrt(3)/2.
    The hexagon can be defined as the intersection of 6 half-planes.
    For a hexagon centered at origin with rotation 0 (flat top), the edges are at:
    - Top: y <= s*sqrt(3)/2
    - Bottom: y >= -s*sqrt(3)/2
    - Top-right: (x*sqrt(3)/2 - y/2) <= s*sqrt(3)/2 * ... 
    
    Let me use a cleaner formulation. A regular hexagon centered at c with circumradius R (= side length) 
    and rotation theta can be described by its 6 edge normals.
    
    For a regular hexagon, the distance from center to each edge (apothem) is R * cos(pi/6) = R * sqrt(3)/2.
    
    The 6 edge normals (outward) for a hexagon rotated by theta are at angles:
    theta + 0, theta + 60, theta + 120, theta + 180, theta + 240, theta + 300 degrees? 
    No, the edges are perpendicular to directions at theta + 30, theta + 90, etc.
    
    Let's use: for a regular hexagon with vertices at angles theta + k*60, the edge normals are at 
    theta + 30 + k*60 for k=0..5.
    
    The constraint for a point p to be inside is: (p - c) · n_k <= apothem = R * sqrt(3)/2 for all k.
    
    So R >= max_k ((p - c) · n_k) / (sqrt(3)/2) for all points p in the inner hexagons.
    
    The maximum of (p - c) · n_k over a convex polygon is achieved at a vertex.
    """
    # Edge normals for outer hexagon
    theta = np.radians(outer_angle)
    # Edge normals are perpendicular to edges. For vertices at theta + k*pi/3, 
    # the outward normals to edges are at theta + pi/6 + k*pi/3
    normal_angles = theta + np.pi/6 + np.arange(6) * np.pi/3
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    apothem_factor = np.sqrt(3) / 2  # apothem = R * apothem_factor
    
    max_val = 0.0
    for row in inner:
        x, y, ang = row
        verts = _hex_vertices(x, y, 1.0, ang)
        # Translate by -center
        verts_rel = verts - np.array(outer_center)
        # Project onto each normal
        projections = verts_rel @ normals.T  # shape (6, 6)
        max_proj = np.max(projections, axis=0)  # shape (6,)
        max_val = max(max_val, np.max(max_proj))
    
    R = max_val / apothem_factor
    return R

def construct():
    """
    Radial Slicing: 1 center, 5 inner ring, 6 outer ring.
    Optimize ring radii and global rotation.
    """
    # Ring structure:
    # Ring 0: 1 hexagon at center
    # Ring 1: 5 hexagons at radius r1, angles offset
    # Ring 2: 6 hexagons at radius r2, angles offset
    
    # For 5 hexagons in a ring, they are at angles 0, 72, 144, 216, 288 degrees (regular pentagon)
    # For 6 hexagons in a ring, they are at angles 0, 60, 120, 180, 240, 300 degrees (regular hexagon)
    
    # Parameters to optimize:
    # r1: radius of inner ring
    # r2: radius of outer ring
    # global_angle: global rotation of the entire configuration
    # outer_angle: rotation of the outer hexagon
    
    def objective(params):
        r1, r2, global_angle, outer_angle = params
        global_angle_rad = np.radians(global_angle)
        
        inner = []
        
        # Ring 0: center
        inner.append([0.0, 0.0, 0.0])
        
        # Ring 1: 5 hexagons
        for k in range(5):
            angle = global_angle_rad + 2 * np.pi * k / 5
            x = r1 * np.cos(angle)
            y = r1 * np.sin(angle)
            # Rotate each hexagon to align radially? Or keep fixed?
            # Let's rotate each hexagon so its "flat" side faces outward, or keep at 0
            # For simplicity, keep hexagon rotation at 0 degrees initially
            inner.append([x, y, 0.0])
        
        # Ring 2: 6 hexagons
        for k in range(6):
            angle = global_angle_rad + 2 * np.pi * k / 6
            x = r2 * np.cos(angle)
            y = r2 * np.sin(angle)
            inner.append([x, y, 0.0])
        
        # Compute minimum outer side length
        R = _compute_min_outer_side(inner, (0.0, 0.0), outer_angle)
        
        # Check for overlaps (disjointness)
        # For now, add a penalty for overlaps
        overlap_penalty = 0.0
        for i in range(len(inner)):
            for j in range(i + 1, len(inner)):
                xi, yi, ai = inner[i]
                xj, yj, aj = inner[j]
                dist = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                # Two unit hexagons are disjoint if their distance is greater than 2 * apothem = sqrt(3)
                # Actually, the minimum distance between centers of two disjoint unit hexagons depends on orientation
                # For worst case, they can be closer. Let's use a conservative bound.
                # The inradius of a unit hexagon is sqrt(3)/2, so if centers are at least 2*sqrt(3)/2 = sqrt(3) apart,
                # they are definitely disjoint when aligned. But with rotation, they can be closer.
                # A safe lower bound: if dist < 2 * (sqrt(3)/2) * cos(pi/6) ... this is getting complex.
                # Let's use: if dist < 2.0, there might be overlap (since max extent is 2*side=2 in any direction for a hexagon? No, circumradius is 1, so diameter is 2)
                # Actually, the maximum distance from center to any point is 1 (circumradius). So if center distance > 2, definitely disjoint.
                # If center distance < 2, they might overlap.
                # Let's add a penalty if dist < 1.8 (conservative)
                if dist < 1.8:
                    overlap_penalty += (1.8 - dist) ** 2
        
        return R + 100 * overlap_penalty
    
    # Initial guess
    # For 12 hexagons in a hexagonal arrangement, a good starting point:
    # r1 ~ 2.0, r2 ~ 3.5, global_angle ~ 0, outer_angle ~ 0
    x0 = [2.0, 3.5, 0.0, 0.0]
    
    # Try multiple initial points
    best_R = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_side = float('inf')
    best_outer_angle = 0.0
    
    for r1_init in [1.8, 2.0, 2.2]:
        for r2_init in [3.0, 3.5, 4.0]:
            for ga_init in [0.0, 30.0, 60.0]:
                for oa_init in [0.0, 30.0, 60.0]:
                    x0 = [r1_init, r2_init, ga_init, oa_init]
                    result = minimize(objective, x0, method='Nelder-Mead', 
                                     options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
                    if result.fun < best_R:
                        best_R = result.fun
                        r1, r2, global_angle, outer_angle = result.x
                        global_angle_rad = np.radians(global_angle)
                        
                        inner = []
                        # Ring 0: center
                        inner.append([0.0, 0.0, 0.0])
                        
                        # Ring 1: 5 hexagons
                        for k in range(5):
                            angle = global_angle_rad + 2 * np.pi * k / 5
                            x = r1 * np.cos(angle)
                            y = r1 * np.sin(angle)
                            inner.append([x, y, 0.0])
                        
                        # Ring 2: 6 hexagons
                        for k in range(6):
                            angle = global_angle_rad + 2 * np.pi * k / 6
                            x = r2 * np.cos(angle)
                            y = r2 * np.sin(angle)
                            inner.append([x, y, 0.0])
                        
                        best_inner = inner
                        best_outer_center = (0.0, 0.0)
                        best_outer_side = _compute_min_outer_side(inner, (0.0, 0.0), outer_angle)
                        best_outer_angle = outer_angle
    
    # Fallback if optimization fails
    if best_inner is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_outer_center, best_outer_side, best_outer_angle

# EVOLVE-BLOCK-END
