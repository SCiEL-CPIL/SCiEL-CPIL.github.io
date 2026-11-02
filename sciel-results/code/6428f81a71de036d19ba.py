import math
import time
import numpy as np
from scipy.optimize import linprog

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0

def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    return np.array([(cx + math.cos(a0 + math.pi*k/3), cy + math.sin(a0 + math.pi*k/3)) for k in range(6)])

def _min_enclosing_hex_side(verts, outer_angle_deg):
    """
    Find minimum side length of a regular hexagon (with vertex at outer_angle_deg)
    that contains all points in verts (numpy array of shape (M,2)).
    Returns (cx, cy, side) or None.
    """
    normals = np.array([(math.cos(math.radians(outer_angle_deg + 30 + 60*k)),
                         math.sin(math.radians(outer_angle_deg + 30 + 60*k))) for k in range(6)])
    
    M = len(verts)
    A = []
    b = []
    for i in range(M):
        vx, vy = verts[i]
        for nx, ny in normals:
            vd = vx*nx + vy*ny
            A.append([-nx, -ny, -1.0])
            b.append(-vd)
            A.append([nx, ny, -1.0])
            b.append(vd)
    A.append([0.0, 0.0, -1.0])
    b.append(0.0)
    
    c_obj = [0.0, 0.0, 1.0]
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs')
        if res.success:
            cx, cy, R = res.x
            side = R / (math.sqrt(3)/2)
            return cx, cy, side
    except Exception:
        pass
    return None

def _generate_hex_lattice_positions(n_points):
    """
    Generate n_points non-overlapping unit hexagon positions on a hexagonal lattice.
    Uses the 11 closest centers to the origin.
    """
    # Basis vectors for hex lattice of unit hexagons (pointy-right orientation)
    # Distance between centers is sqrt(3)
    # Neighbors are at angles 30, 90, 150, 210, 270, 330 degrees
    a = 1.5
    b = math.sqrt(3) / 2
    # Vector to neighbor at 30 deg: (1.5, sqrt(3)/2)
    v1 = np.array([a, b])
    # Vector to neighbor at 90 deg: (0, sqrt(3))
    v2 = np.array([0, math.sqrt(3)])
    
    candidates = []
    # Search in a range large enough to contain at least n_points
    # Radius in lattice coords ~ sqrt(n_points)
    max_k = 5
    for i in range(-max_k, max_k + 1):
        for j in range(-max_k, max_k + 1):
            pos = i * v1 + j * v2
            dist_sq = pos[0]**2 + pos[1]**2
            candidates.append((dist_sq, pos[0], pos[1]))
            
    candidates.sort(key=lambda x: x[0])
    positions = [(p[1], p[2]) for p in candidates[:n_points]]
    return positions

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        # Fallback: widely spaced hexagons
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Use the robust lattice-based positions
    positions = _generate_hex_lattice_positions(N)
    
    # Try inner angle 0.0 (aligned with lattice)
    # The lattice is built for angle 0.0 hexagons.
    # If we rotate inner hexagons, they might overlap unless we scale up or adjust.
    # Since we want tight packing, we should stick to the lattice orientation (0.0)
    # or potentially 30.0 if the lattice was defined differently.
    # Our lattice vectors v1, v2 correspond to neighbors of a 0-degree hexagon.
    # So inner_angle should be 0.0.
    
    inner_angle = 0.0
    
    all_vertices = []
    for px, py in positions:
        all_vertices.extend(_hex_vertices(px, py, inner_angle))
    all_vertices = np.array(all_vertices)
    
    # Coarse search over outer angle
    best_coarse_side = float('inf')
    best_coarse_angle = 0.0
    best_coarse_center = (0.0, 0.0)
    
    # The optimal outer angle is likely aligned with the cluster symmetry.
    # For a hexagonal cluster, 0, 30, 60... are candidates.
    # Since the cluster is not perfectly symmetric (11 points), we search.
    
    for outer_angle_deg in np.arange(0, 30.1, 1.0):
        if time.time() > _DEADLINE - 10.0:
            break
        res = _min_enclosing_hex_side(all_vertices, outer_angle_deg)
        if res:
            cx, cy, side = res
            if side < best_coarse_side:
                best_coarse_side = side
                best_coarse_angle = outer_angle_deg
                best_coarse_center = (cx, cy)
    
    # Fine search around best coarse angle
    if best_coarse_side < float('inf'):
        for outer_angle_deg in np.arange(max(0, best_coarse_angle - 1.5),
                                         min(30.1, best_coarse_angle + 1.5), 0.05):
            if time.time() > _DEADLINE - 10.0:
                break
            res = _min_enclosing_hex_side(all_vertices, outer_angle_deg)
            if res:
                cx, cy, side = res
                if side < best_coarse_side:
                    best_coarse_side = side
                    best_coarse_angle = outer_angle_deg
                    best_coarse_center = (cx, cy)
    
    if best_coarse_side < best_side:
        best_side = best_coarse_side
        best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
        best_center = best_coarse_center
        best_outer_angle = best_coarse_angle

    if best_inner is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_outer_angle
