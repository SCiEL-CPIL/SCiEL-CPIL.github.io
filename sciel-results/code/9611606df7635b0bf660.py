"""Radial Slicing formulation for 12 unit hexagons in a regular hexagon."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import math

N = 12

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append((x, y))
    return verts

def get_inner_hex_positions(ring1_r, ring2_r, global_rot_deg):
    """
    Generate positions and angles for 12 unit hexagons in concentric rings.
    Ring 0: 1 hexagon at center
    Ring 1: 5 hexagons at radius ring1_r
    Ring 2: 6 hexagons at radius ring2_r
    """
    positions = []
    angles = []
    
    # Center hexagon
    positions.append((0.0, 0.0))
    angles.append(0.0)
    
    # Inner ring: 5 hexagons
    for i in range(5):
        theta = math.radians(global_rot_deg) + i * 2 * math.pi / 5.0
        x = ring1_r * math.cos(theta)
        y = ring1_r * math.sin(theta)
        positions.append((x, y))
        angles.append(global_rot_deg)
    
    # Outer ring: 6 hexagons
    for i in range(6):
        theta = math.radians(global_rot_deg) + i * 2 * math.pi / 6.0 + math.pi / 6.0
        x = ring2_r * math.cos(theta)
        y = ring2_r * math.sin(theta)
        positions.append((x, y))
        angles.append(global_rot_deg)
    
    return positions, angles

def compute_convex_hull_extent(positions, angles, side=1.0):
    """
    Compute all vertices of all inner hexagons, find convex hull,
    and return the maximum distance from origin to any hull vertex.
    """
    all_verts = []
    for (cx, cy), ang in zip(positions, angles):
        verts = hexagon_vertices(cx, cy, side, ang)
        all_verts.extend(verts)
    
    all_verts = np.array(all_verts)
    
    try:
        hull = ConvexHull(all_verts)
        hull_verts = all_verts[hull.vertices]
        # Maximum radial extent
        max_r = np.max(np.sqrt(np.sum(hull_verts**2, axis=1)))
        return max_r
    except Exception:
        # Fallback: just max distance
        return np.max(np.sqrt(np.sum(all_verts**2, axis=1)))

def outer_hex_side_from_radial_extent(max_r, outer_rot_deg):
    """
    Given the maximum radial extent of the cluster, find the minimum side length
    of a regular hexagon (with given rotation) that contains it.
    For a regular hexagon with side s and rotation phi, the support function in direction theta is:
    h(theta) = s * max over edges... Actually, for a regular hexagon centered at origin,
    the distance from center to each side is s * sqrt(3)/2 (apothem).
    The hexagon contains a point at distance r in direction theta iff r <= h(theta).
    
    For a regular hexagon with side s, the support function is:
    h(theta) = s * sqrt(3) / 2 * max(1, ...) 
    Actually, the support function of a regular hexagon with side s is:
    h(theta) = s * sqrt(3)/2 * max_{k=0..5} cos(theta - k*pi/3)  ... no.
    
    Let me think more carefully. A regular hexagon with side s has vertices at distance s from center.
    The apothem (distance from center to side) is s * sqrt(3)/2.
    
    The support function h(theta) = max_{v in hex} v . (cos theta, sin theta)
    For a regular hexagon, h(theta) = s * sqrt(3)/2 * max(1, 2*cos(6*theta mod 2pi/3 - ...)) 
    
    Actually, for a regular hexagon with side s centered at origin, the support function is:
    h(theta) = s * sqrt(3)/2 * max_{k=0,1,2} |cos(theta - k*pi/3)|  ... no, that's not right either.
    
    Let me just compute it numerically. The hexagon vertices are at angles 0, 60, 120, 180, 240, 300 degrees (for rotation 0), at distance s.
    The edges are between consecutive vertices. The support function in direction theta is the max of dot products with all vertices (since it's convex).
    
    So h(theta) = s * max_{i=0..5} cos(theta - i*pi/3 - rot)
    
    The minimum side s such that the hexagon contains all points with max radial extent R in various directions...
    
    Actually, the problem is: we have a set of points (the cluster). We need to find the minimum s such that there exists a regular hexagon of side s (with some rotation) containing all these points.
    
    But the task says we can choose the outer hexagon's center, side, and rotation. So we need to minimize s over center, side, rotation such that all inner hexagons are contained.
    
    This is complex. Let me simplify: assume the outer hexagon is centered at origin (since the cluster is roughly centered). Then we need to find min s and rotation phi such that all cluster vertices are inside the hexagon.
    
    For a hexagon centered at origin with side s and rotation phi, a point p is inside iff for all 6 edges, the point is on the correct side.
    Equivalently, the support function condition: for all directions, the projection of p onto that direction <= h_hex(direction).
    
    A simpler approach: for a given rotation phi of the outer hexagon, the required s is:
    s >= max over all cluster vertices v of: |v| / (sqrt(3)/2 * max_{k=0..5} cos(angle(v) - phi - k*pi/3))
    
    Wait, the support function of the hexagon in direction u (unit vector) is:
    h(u) = s * sqrt(3)/2 * max_{k=0..5} max(0, cos(angle(u) - phi - k*pi/3))
    
    Hmm, this is getting complicated. Let me just do a 2D optimization: optimize ring1_r, ring2_r, global_rot_deg, outer_rot_deg, and outer_center_x, outer_center_y to minimize outer side s.
    
    For a given configuration of inner hexagons and outer hexagon parameters, check containment and compute required s.
    """
    pass

def objective(params):
    """
    Objective function: returns the outer hexagon side length needed to contain all inner hexagons.
    params = [ring1_r, ring2_r, global_rot_deg, outer_rot_deg, cx, cy]
    """
    ring1_r, ring2_r, global_rot_deg, outer_rot_deg, cx, cy = params
    
    # Ensure positive radii
    ring1_r = max(ring1_r, 1.0)
    ring2_r = max(ring2_r, ring1_r)
    
    positions, angles = get_inner_hex_positions(ring1_r, ring2_r, global_rot_deg)
    
    # Get all vertices of inner hexagons
    all_verts = []
    for (px, py), ang in zip(positions, angles):
        verts = hexagon_vertices(px, py, 1.0, ang)
        all_verts.extend(verts)
    all_verts = np.array(all_verts)
    
    # For the outer hexagon centered at (cx, cy) with rotation outer_rot_deg,
    # we need to find minimum s such that all all_verts are inside.
    # 
    # A regular hexagon with side s centered at (cx,cy) with rotation phi has 6 edges.
    # Each edge is a line at distance s*sqrt(3)/2 from center.
    # The 6 edge normals are at angles phi + k*pi/3 for k=0..5.
    # 
    # For a point p to be inside, for each edge normal n_k:
    #   (p - center) . n_k <= s * sqrt(3)/2
    # 
    # So s >= max over all points p and all k of: (p - center) . n_k / (sqrt(3)/2)
    # 
    # But we also need (p - center) . n_k >= -s * sqrt(3)/2, which is automatically satisfied
    # if we consider all 6 directions (since -n_k is also one of the normals for k+3).
    
    phi = math.radians(outer_rot_deg)
    center = np.array([cx, cy])
    
    max_val = 0.0
    for k in range(6):
        theta = phi + k * math.pi / 3.0
        n = np.array([math.cos(theta), math.sin(theta)])
        projections = np.dot(all_verts - center, n)
        max_val = max(max_val, np.max(projections))
    
    s = max_val / (math.sqrt(3) / 2.0)
    
    # Add small penalty if s is too small (shouldn't happen)
    if s < 1.0:
        s = 1.0
    
    return s

def construct():
    """
    Construct 12 unit hexagons inside a regular hexagon, minimizing the outer side length.
    """
    # Initial guess
    x0 = [2.0, 3.5, 0.0, 0.0, 0.0, 0.0]
    
    # Bounds
    bounds = [
        (1.0, 5.0),   # ring1_r
        (1.0, 6.0),   # ring2_r
        (0.0, 360.0), # global_rot_deg
        (0.0, 360.0), # outer_rot_deg
        (-5.0, 5.0),  # cx
        (-5.0, 5.0),  # cy
    ]
    
    # Optimize
    result = minimize(objective, x0, method='Nelder-Mead', 
                      bounds=bounds,
                      options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    best_params = result.x
    ring1_r, ring2_r, global_rot_deg, outer_rot_deg, cx, cy = best_params
    
    ring1_r = max(ring1_r, 1.0)
    ring2_r = max(ring2_r, ring1_r)
    
    positions, angles = get_inner_hex_positions(ring1_r, ring2_r, global_rot_deg)
    
    # Build inner list
    inner = []
    for (px, py), ang in zip(positions, angles):
        inner.append([px, py, ang])
    
    # Compute side length
    s = objective(best_params)
    
    return inner, (cx, cy), s, outer_rot_deg

# EVOLVE-BLOCK-START
# (The construct function is defined above)
# EVOLVE-BLOCK-END
