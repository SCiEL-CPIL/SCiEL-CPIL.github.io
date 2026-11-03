"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize_scalar

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, angle_deg):
    """Get vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        verts.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
    return np.array(verts)

def _min_hexagon_side(vertices, angle_deg):
    """Given a set of vertices and the outer hexagon rotation angle,
    compute the minimum side length of a regular hexagon (centered at origin, rotated by angle_deg)
    that contains all vertices."""
    a = np.radians(angle_deg)
    # For a regular hexagon centered at origin with rotation a, the support function in direction d
    # gives the required half-width. The side length s satisfies:
    # For each of the 6 normal directions (at angles a + k*pi/3 + pi/6 for k=0..5, 
    # actually the outward normals are at a + k*pi/3 for the flat sides... 
    # Let me think more carefully.
    # A regular hexagon with "pointy-top" (vertex at angle a) has vertices at a + k*pi/3.
    # The 6 face normals are at a + pi/6 + k*pi/3.
    # The constraint is: for each face normal n_k, max_i (v_i . n_k) <= s * cos(pi/6) = s*sqrt(3)/2
    # Wait, for a regular hexagon with side s, the apothem (distance from center to face) is s*sqrt(3)/2.
    # So we need: for each of 6 face normals, the max projection of all vertices <= apothem = s*sqrt(3)/2
    # Thus s >= 2/sqrt(3) * max_k (max_i v_i . n_k)
    
    normals = []
    for k in range(6):
        theta = a + np.pi / 6 + k * np.pi / 3
        normals.append(np.array([np.cos(theta), np.sin(theta)]))
    
    max_proj = 0
    for n in normals:
        projs = vertices @ n
        max_proj = max(max_proj, np.max(projs))
    
    side = 2.0 / np.sqrt(3.0) * max_proj
    return side

def _total_extent(vertices, angle_deg):
    """Compute the side length of the minimum enclosing regular hexagon for given rotation."""
    return _min_hexagon_side(vertices, angle_deg)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    # Create 11 hexagons in a linear chain on triangular lattice
    # Center-to-center distance for touching hexagons = 2 * apothem = 2 * (1*sqrt(3)/2) = sqrt(3)
    spacing = np.sqrt(3.0)
    gap = 1.001  # small gap to ensure strict disjointness
    
    centers = []
    all_vertices = []
    for k in range(N):
        offset = (k - N // 2) * spacing * gap
        cx = offset
        cy = 0.0
        centers.append((cx, cy))
        verts = _hex_vertices(cx, cy, 1.0, 0.0)
        all_vertices.append(verts)
    
    all_vertices = np.vstack(all_vertices)  # shape (66, 2)
    
    # Optimize the outer hexagon rotation angle
    # The inner hexagons are all axis-aligned (angle=0), so the problem is symmetric.
    # We search over the outer hexagon rotation from 0 to 60 degrees (periodicity).
    
    best_side = float('inf')
    best_angle = 0.0
    
    # Coarse search
    angles = np.linspace(0, 60, 121)
    for ang in angles:
        if time.time() > deadline:
            break
        s = _total_extent(all_vertices, ang)
        if s < best_side:
            best_side = s
            best_angle = ang
    
    # Fine search around best
    if time.time() < deadline - 5:
        result = minimize_scalar(
            lambda a: _total_extent(all_vertices, a),
            bounds=(best_angle - 3, best_angle + 3),
            method='bounded',
            options={'xatol': 1e-6}
        )
        if result.success and result.fun < best_side:
            best_side = result.fun
            best_angle = result.x
    
    # Even finer polish
    if time.time() < deadline - 2:
        result2 = minimize_scalar(
            lambda a: _total_extent(all_vertices, a),
            bounds=(best_angle - 0.5, best_angle + 0.5),
            method='bounded',
            options={'xatol': 1e-9}
        )
        if result2.success and result2.fun < best_side:
            best_side = result2.fun
            best_angle = result2.x
    
    # Build inner hexagon positions
    inner = []
    for k in range(N):
        offset = (k - N // 2) * spacing * gap
        cx = offset
        cy = 0.0
        inner.append([cx, cy, 0.0])
    
    outer_center = (0.0, 0.0)
    outer_side = best_side
    outer_angle = best_angle
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END

