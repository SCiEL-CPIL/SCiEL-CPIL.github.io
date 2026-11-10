# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, a_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side s, rotation a_deg."""
    a = np.radians(a_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _hex_face_normals(a_deg):
    """Return 6 outward face normals (unit vectors) for a hexagon with rotation a_deg."""
    a = np.radians(a_deg)
    angles = a + np.radians(30) + np.radians(60) * np.arange(6)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _pair_separated(v1, n1, v2, n2, tol=1e-6):
    """Check if two hexagons are separated using all 12 edge normals."""
    for i in range(6):
        n = n1[i]
        proj1 = v1 @ n
        proj2 = v2 @ n
        if proj1.max() - proj2.min() < -tol or proj2.max() - proj1.min() < -tol:
            return True
    for i in range(6):
        n = n2[i]
        proj1 = v1 @ n
        proj2 = v2 @ n
        if proj1.max() - proj2.min() < -tol or proj2.max() - proj1.min() < -tol:
            return True
    return False


def _check_inner_separation(inner_data, tol=1e-6):
    """Check all pairs of inner hexagons are separated."""
    n = len(inner_data)
    vertices = []
    normals = []
    for i in range(n):
        cx, cy, a = inner_data[i]
        v = _hex_vertices(cx, cy, 1.0, a)
        nm = _hex_face_normals(a)
        vertices.append(v)
        normals.append(nm)
    for i in range(n):
        for j in range(i + 1, n):
            if not _pair_separated(vertices[i], normals[i], vertices[j], normals[j], tol):
                return False
    return True


def _check_inside_outer_full(inner_data, outer_data, side, tol=1e-6):
    """Check all inner hexagons are inside the outer hexagon with given side length."""
    cx, cy, a = outer_data
    outer_verts = _hex_vertices(cx, cy, side, a)
    n_outer = 6
    for i in range(len(inner_data)):
        ix, iy, ia = inner_data[i]
        inner_verts = _hex_vertices(ix, iy, 1.0, ia)
        for p in inner_verts:
            for e in range(n_outer):
                p1 = outer_verts[e]
                p2 = outer_verts[(e + 1) % n_outer]
                edge = p2 - p1
                vec = p - p1
                cross_val = edge[0] * vec[1] - edge[1] * vec[0]
                if cross_val < -tol:
                    return False
    return True


def _validate(inner_data, outer_data, side, tol=1e-6):
    """Full validation."""
    if not (np.all(np.isfinite(inner_data)) and np.all(np.isfinite(outer_data)) and np.isfinite(side)):
        return False
    if side < np.sqrt(11) - 1e-9:
        return False
    if not _check_inner_separation(inner_data, tol):
        return False
    if not _check_inside_outer_full(inner_data, outer_data, side, tol):
        return False
    outer_center = np.array([outer_data[0], outer_data[1]])
    for i in range(len(inner_data)):
        ix, iy, ia = inner_data[i]
        shifted = np.array([ix, iy]) - outer_center
        v = _hex_vertices(shifted[0], shifted[1], 1.0, ia)
        for e in range(6):
            d = v[(e + 1) % 6] - v[e]
            length = np.linalg.norm(d)
            if abs(length - 1.0) > 1e-8:
                return False
    return True


def _hex_dist_to_origin(px, py):
    """Hex distance of point (px,py) from origin for hexagon with side s, angle 0."""
    return abs(px) + abs(py) / np.sqrt(3)


def _compute_required_side(inner_data):
    """Compute minimum side length of outer hexagon (centered at origin, angle 0) to contain all inner hexagons."""
    required_side = 0.0
    for i in range(len(inner_data)):
        ix, iy, ia = inner_data[i]
        v = _hex_vertices(ix, iy, 1.0, ia)
        for p in v:
            hex_dist = _hex_dist_to_origin(p[0], p[1])
            required_side = max(required_side, hex_dist)
    return required_side


def _cluster_objective(params, cluster_def):
    """
    Objective for rigid cluster optimization.
    params: [x_offset, y_offset, cluster_rotation_deg]
    cluster_def: list of (rel_x, rel_y, rel_angle_deg) for each hexagon in the cluster
    Returns the required outer side length, or large penalty if invalid.
    """
    x_off, y_off, rot_deg = params
    rot_rad = np.radians(rot_deg)
    cos_r = np.cos(rot_rad)
    sin_r = np.sin(rot_rad)
    
    inner_data = np.zeros((len(cluster_def), 3))
    for i, (rx, ry, ra) in enumerate(cluster_def):
        # Rotate relative position
        x_rot = rx * cos_r - ry * sin_r
        y_rot = rx * sin_r + ry * cos_r
        inner_data[i, 0] = x_rot + x_off
        inner_data[i, 1] = y_rot + y_off
        inner_data[i, 2] = ra + rot_deg
    
    # Check separation
    if not _check_inner_separation(inner_data, tol=1e-6):
        # Compute overlap penalty
        n = len(inner_data)
        vertices = []
        normals = []
        for i in range(n):
            cx, cy, a = inner_data[i]
            v = _hex_vertices(cx, cy, 1.0, a)
            nm = _hex_face_normals(a)
            vertices.append(v)
            normals.append(nm)
        
        overlap_penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                min_gap = float('inf')
                # Check all 12 normals
                for k in range(6):
                    n_vec = normals[i][k]
                    proj1 = vertices[i] @ n_vec
                    proj2 = vertices[j] @ n_vec
                    gap = min(proj1.min() - proj2.max(), proj2.min() - proj1.max())
                    min_gap = min(min_gap, -gap)
                for k in range(6):
                    n_vec = normals[j][k]
                    proj1 = vertices[i] @ n_vec
                    proj2 = vertices[j] @ n_vec
                    gap = min(proj1.min() - proj2.max(), proj2.min() - proj1.max())
                    min_gap = min(min_gap, -gap)
                if min_gap > 0:
                    overlap_penalty += min_gap ** 2
        
        req_side = _compute_required_side(inner_data)
        return 1000.0 + req_side + overlap_penalty
    
    req_side = _compute_required_side(inner_data)
    return req_side


def _local_refine(inner_data, max_iter=5000):
    """Local refinement using Nelder-Mead on the full 33-DOF problem."""
    def obj(params):
        inner = params.reshape(11, 3)
        if not _check_inner_separation(inner, tol=1e-6):
            return 1000.0 + _compute_required_side(inner)
        return _compute_required_side(inner)
    
    result = minimize(
        obj,
        inner_data.flatten(),
        method='Nelder-Mead',
        options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10}
    )
    return result.x.reshape(11, 3)


def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 600.0
    
    outer_data = np.array([0.0, 0.0, 0.0])
    
    # Define several cluster configurations to try
    # Each cluster is a list of (rel_x, rel_y, rel_angle_deg)
    
    # Cluster 1: Hexagonal grid - center + 6 neighbors + 4 in second ring
    # For unit hexagons (vertex at 0 deg), adjacent centers are sqrt(3) apart
    # in directions 0, 60, 120, 180, 240, 300 degrees
    s3 = np.sqrt(3)
    
    cluster1 = [
        (0, 0, 0),  # center
        (s3, 0, 0),  # right
        (s3 * np.cos(np.radians(60)), s3 * np.sin(np.radians(60)), 0),  # upper-right
        (s3 * np.cos(np.radians(120)), s3 * np.sin(np.radians(120)), 0),  # upper-left
        (-s3, 0, 0),  # left
        (s3 * np.cos(np.radians(240)), s3 * np.sin(np.radians(240)), 0),  # lower-left
        (s3 * np.cos(np.radians(300)), s3 * np.sin(np.radians(300)), 0),  # lower-right
        # Second ring: 4 hexagons at distance 2*sqrt(3) in diagonal directions
        (2 * s3 * np.cos(np.radians(30)), 2 * s3 * np.sin(np.radians(30)), 0),
        (2 * s3 * np.cos(np.radians(90)), 2 * s3 * np.sin(np.radians(90)), 0),
        (2 * s3 * np.cos(np.radians(150)), 2 * s3 * np.sin(np.radians(150)), 0),
        (2 * s3 * np.cos(np.radians(210)), 2 * s3 * np.sin(np.radians(210)), 0),
    ]
    
    # Cluster 2: Linear arrangement with stagger
    cluster2 = [
        (0, 0, 0),
        (2.0, 0, 0),
        (-2.0, 0, 0),
        (1.0, s3, 0),
        (-1.0, s3, 0),
        (1.0, -s3, 0),
        (-1.0, -s3, 0),
        (3.0, s3, 0),
        (-3.0, s3, 0),
        (3.0, -s3, 0),
        (-3.0, -s3, 0),
    ]
    
    # Cluster 3: Compact 3-4-4 arrangement
    dy = 1.5  # vertical spacing for staggered rows
    cluster3 = [
        (0, dy, 0),
        (-1.5, 0, 0),
        (0, 0, 0),
        (1.5, 0, 0),
        (-3.0, -dy, 0),
        (-1.5, -dy, 0),
        (0, -dy, 0),
        (1.5, -dy, 0),
        (-4.5, -2*dy, 0),
        (-3.0, -2*dy, 0),
        (-1.5, -2*dy, 0),
    ]
    
    # Cluster 4: Another compact arrangement
    cluster4 = [
        (0, 0, 0),
        (2.0, 0, 0),
        (-2.0, 0, 0),
        (1.0, 1.732, 0),
        (-1.0, 1.732, 0),
        (1.0, -1.732, 0),
        (-1.0, -1.732, 0),
        (3.0, 0.866, 0),
        (-3.0, 0.866, 0),
        (3.0, -0.866, 0),
        (-3.0, -0.866, 0),
    ]
    
    best_inner = None
    best_side = float('inf')
    
    clusters = [cluster1, cluster2, cluster3, cluster4]
    
    for cluster_def in clusters:
        if time.time() > deadline - 120:
            break
        
        # Optimize the rigid cluster: [x_offset, y_offset, rotation]
        bounds = [(-5, 5), (-5, 5), (0, 60)]
        
        try:
            result = minimize(
                _cluster_objective,
                [0, 0, 0],
                args=(cluster_def,),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if result.fun < 500:  # Valid solution
                x_off, y_off, rot_deg = result.x
                rot_rad = np.radians(rot_deg)
                cos_r = np.cos(rot_rad)
                sin_r = np.sin(rot_rad)
                
                inner_data = np.zeros((len(cluster_def), 3))
                for i, (rx, ry, ra) in enumerate(cluster_def):
                    x_rot = rx * cos_r - ry * sin_r
                    y_rot = rx * sin_r + ry * cos_r
                    inner_data[i, 0] = x_rot + x_off
                    inner_data[i, 1] = y_rot + y_off
                    inner_data[i, 2] = ra + rot_deg
                
                req_side = _compute_required_side(inner_data)
                
                if _validate(inner_data, outer_data, req_side):
                    if req_side < best_side:
                        best_side = req_side
                        best_inner = inner_data.copy()
        except Exception:
            pass
    
    # If we have a valid solution, do local refinement on the full 33-DOF problem
    if best_inner is not None and time.time() < deadline - 60:
        try:
            refined = _local_refine(best_inner, max_iter=5000)
            req_side = _compute_required_side(refined)
            if _validate(refined, outer_data, req_side):
                if req_side < best_side:
                    best_side = req_side
                    best_inner = refined
        except Exception:
            pass
    
    # Fallback if no valid solution found
    if best_inner is None:
        # Use the original seed arrangement
        inner_data = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ])
        req_side = _compute_required_side(inner_data)
        if _validate(inner_data, outer_data, req_side):
            best_inner = inner_data
            best_side = req_side
        else:
            # Last resort
            best_inner = inner_data
            best_side = req_side
    
    return best_inner, outer_data, best_side


# EVOLVE-BLOCK-END
