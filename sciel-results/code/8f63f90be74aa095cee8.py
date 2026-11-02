import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

# Precompute unit hexagon vertices for a unit side length
_HEX_VERTS = np.array([
    [1.0, 0.0],
    [0.5, math.sqrt(3)/2],
    [-0.5, math.sqrt(3)/2],
    [-1.0, 0.0],
    [-0.5, -math.sqrt(3)/2],
    [0.5, -math.sqrt(3)/2]
])

def get_all_vertices(params):
    """
    params: (N, 3) array of [x, y, angle_deg]
    Returns: (N*6, 2) array of vertices
    """
    N = len(params)
    # angles in radians
    angles = np.deg2rad(params[:, 2])
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    
    # Rotation matrices: (N, 2, 2)
    # R = [[cos, -sin], [sin, cos]]
    rot = np.zeros((N, 2, 2))
    rot[:, 0, 0] = cos_a
    rot[:, 0, 1] = -sin_a
    rot[:, 1, 0] = sin_a
    rot[:, 1, 1] = cos_a
    
    # _HEX_VERTS is (6, 2)
    # We want to apply rotation to each vertex for each hexagon
    # vertices = _HEX_VERTS @ rot.T ?
    # _HEX_VERTS (6, 2) @ rot.T (2, 2) -> (6, 2) for one hexagon.
    # For all hexagons:
    # We can reshape _HEX_VERTS to (1, 6, 2) and broadcast?
    # Or simpler:
    # verts = np.einsum('ij,klm->kjm', _HEX_VERTS, rot) ?
    # _HEX_VERTS: (6, 2)
    # rot: (N, 2, 2)
    # Result: (N, 6, 2)
    verts = np.einsum('ab,cbd->cad', _HEX_VERTS, rot)
    
    # Add centers
    centers = params[:, :2] # (N, 2)
    verts += centers[:, np.newaxis, :]
    
    return verts.reshape(-1, 2)

def calc_min_enclosing_hex_side_fast(points):
    """
    Vectorized calculation of min enclosing hexagon side.
    """
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    
    # Thetas: 0 to pi/6
    thetas = np.linspace(0, math.pi/6, 73)
    
    # Normals: (73, 6, 2)
    # theta + k*pi/3
    ks = np.arange(6) * math.pi/3
    angles = thetas[:, np.newaxis] + ks[np.newaxis, :]
    normals = np.stack([np.cos(angles), np.sin(angles)], axis=-1) # (73, 6, 2)
    
    # Projections: (73, N_pts, 6)
    # centered_points: (N_pts, 2)
    # normals: (73, 6, 2)
    # projs = np.einsum('ab,cdb->acd', centered_points, normals)
    projs = np.einsum('ab,cdb->acd', centered_points, normals)
    
    # Widths: (73, 6)
    max_projs = np.max(projs, axis=1)
    min_projs = np.min(projs, axis=1)
    widths = max_projs - min_projs
    
    # Max width for each theta: (73,)
    max_widths = np.max(widths, axis=1)
    
    # Side length: (73,)
    sides = max_widths / math.sqrt(3)
    
    return np.min(sides)

def check_overlap_fast(params):
    """
    Fast overlap check using bounding boxes first.
    """
    N = len(params)
    # Bounding boxes
    bboxes = []
    for i in range(N):
        x, y, ang = params[i]
        # Approximate bbox for hexagon
        # Radius is 1.0
        # Bbox is roughly [-1, 1] rotated
        # To be safe, use a square of size 2.2
        bboxes.append((x-1.2, y-1.2, x+1.2, y+1.2))
        
    for i in range(N):
        for j in range(i+1, N):
            # Bbox check
            if bboxes[i][2] < bboxes[j][0] or bboxes[j][2] < bboxes[i][0] or \
               bboxes[i][3] < bboxes[j][1] or bboxes[j][3] < bboxes[i][1]:
                continue
            # SAT Check
            if polygons_intersect_slow(params[i], params[j]):
                return True
    return False

def polygons_intersect_slow(p1, p2):
    # p1, p2 are [x, y, angle]
    v1 = get_hexagon_vertices(p1[0], p1[1], p1[2])
    v2 = get_hexagon_vertices(p2[0], p2[1], p2[2])
    return polygons_intersect(v1, v2)

def get_hexagon_vertices(cx, cy, angle_deg):
    angle_rad = math.radians(angle_deg)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    rot = np.array([[c, -s], [s, c]])
    return (_HEX_VERTS) @ rot.T + np.array([cx, cy])

def polygons_intersect(poly1, poly2):
    def get_axes(poly):
        axes = []
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i+1)%len(poly)]
            edge = p2 - p1
            n = np.array([-edge[1], edge[0]])
            n = n / (np.linalg.norm(n) + 1e-9)
            axes.append(n)
        return axes

    axes = get_axes(poly1) + get_axes(poly2)
    
    for axis in axes:
        proj1 = poly1 @ axis
        min1, max1 = np.min(proj1), np.max(proj1)
        proj2 = poly2 @ axis
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            return False
            
    return True

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Initial configurations
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    axial_2 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    axial_3 = [
        (0, 0), (1, 1), (2, 2), (3, 3),
        (0, 1), (1, 2), (2, 3),
        (-1, 0), (0, -1), (1, -2)
    ]
    
    candidates = [axial_1, axial_2, axial_3]
    
    best_params = None
    best_side = float('inf')
    
    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        pts = axial_to_cartesian(axial)
        
        init_params = np.zeros((N, 3))
        init_params[:, 0] = pts[:, 0]
        init_params[:, 1] = pts[:, 1]
        np.random.seed(42)
        init_params[:, 0] += np.random.uniform(-0.1, 0.1, N)
        init_params[:, 1] += np.random.uniform(-0.1, 0.1, N)
        init_params[:, 2] += np.random.uniform(-5, 5, N)
        
        def objective(params):
            p = params.reshape(N, 3)
            all_verts = get_all_vertices(p)
            side = calc_min_enclosing_hex_side_fast(all_verts)
            
            # Overlap penalty
            if check_overlap_fast(p):
                return side + 100.0
            return side
            
        try:
            result = minimize(objective, init_params.ravel(), method='Nelder-Mead',
                              options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            
            if result.fun < best_side:
                final_params = result.x.reshape(N, 3)
                if not check_overlap_fast(final_params):
                    best_side = result.fun
                    best_params = final_params
        except Exception:
            pass
            
    if best_params is None:
        pts = axial_to_cartesian(candidates[0])
        best_params = np.zeros((N, 3))
        best_params[:, 0] = pts[:, 0]
        best_params[:, 1] = pts[:, 1]
        all_verts = get_all_vertices(best_params)
        best_side = calc_min_enclosing_hex_side_fast(all_verts)
        
    all_verts = get_all_vertices(best_params)
    min_side = calc_min_enclosing_hex_side_fast(all_verts)
    
    # Find best theta for reporting
    # Re-calc theta
    centroid = np.mean(all_verts, axis=0)
    centered_points = all_verts - centroid
    thetas = np.linspace(0, math.pi/6, 73)
    ks = np.arange(6) * math.pi/3
    angles = thetas[:, np.newaxis] + ks[np.newaxis, :]
    normals = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    projs = np.einsum('ab,cdb->acd', centered_points, normals)
    widths = np.max(projs, axis=1) - np.min(projs, axis=1)
    max_widths = np.max(widths, axis=1)
    sides = max_widths / math.sqrt(3)
    best_theta = thetas[np.argmin(sides)]
    
    inner = []
    for i in range(N):
        x, y, ang = best_params[i]
        inner.append([float(x), float(y), float(ang)])
        
    center = [float(centroid[0]), float(centroid[1])]
    side = float(min_side)
    rotation = float(math.degrees(best_theta))
    
    return inner, center, side, rotation
