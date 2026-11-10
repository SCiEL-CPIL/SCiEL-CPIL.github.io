# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

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
    # Vectorized check for speed
    # Project v1 and v2 onto all normals of n1
    proj1_n1 = v1 @ n1.T # (6, 6)
    proj2_n1 = v2 @ n1.T # (6, 6)
    
    # For each normal k, check if intervals are apart
    # Interval 1: [min(proj1_n1[:,k]), max(proj1_n1[:,k])]
    # Interval 2: [min(proj2_n1[:,k]), max(proj2_n1[:,k])]
    # Separated if max1 < min2 - tol OR max2 < min1 - tol
    
    min1 = proj1_n1.min(axis=0)
    max1 = proj1_n1.max(axis=0)
    min2 = proj2_n1.min(axis=0)
    max2 = proj2_n1.max(axis=0)
    
    sep1 = (max1 < min2 - tol)
    sep2 = (max2 < min1 - tol)
    
    if np.any(sep1 | sep2):
        return True

    # Check normals of n2
    proj1_n2 = v1 @ n2.T
    proj2_n2 = v2 @ n2.T
    
    min1 = proj1_n2.min(axis=0)
    max1 = proj1_n2.max(axis=0)
    min2 = proj2_n2.min(axis=0)
    max2 = proj2_n2.max(axis=0)
    
    sep1 = (max1 < min2 - tol)
    sep2 = (max2 < min1 - tol)
    
    if np.any(sep1 | sep2):
        return True
        
    return False

def _check_inner_separation_fast(inner_data, tol=1e-6):
    """Check all pairs of inner hexagons are separated. Returns True if valid."""
    n = len(inner_data)
    if n < 2: return True
    
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

def _check_inside_outer_fast(inner_data, outer_data, side, tol=1e-6):
    """Check all inner hexagons are inside the outer hexagon."""
    cx, cy, a = outer_data
    outer_verts = _hex_vertices(cx, cy, side, a)
    n_outer = 6
    outer_normals = []
    # Precompute outer edge normals and offsets for faster point-in-poly check
    # Edge from v[i] to v[i+1]
    for i in range(n_outer):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % n_outer]
        edge = p2 - p1
        # Outward normal
        n = np.array([-edge[1], edge[0]])
        n = n / np.linalg.norm(n)
        # Ensure it points outward (away from center)
        if np.dot(n, p1 - np.array([cx, cy])) < 0:
            n = -n
        outer_normals.append((n, p1))
        
    for i in range(len(inner_data)):
        ix, iy, ia = inner_data[i]
        inner_verts = _hex_vertices(ix, iy, 1.0, ia)
        for p in inner_verts:
            for n, p1 in outer_normals:
                # cross(p2-p1, p-p1) >= -tol is equivalent to dot(n, p-p1) >= -tol 
                # if n is the inward normal? 
                # Let's stick to the cross product definition in the prompt to be safe
                pass
            
    # Re-implementing standard check for correctness, it's fast enough for 11*6 points
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
    if not _check_inner_separation_fast(inner_data, tol):
        return False
    if not _check_inside_outer_fast(inner_data, outer_data, side, tol):
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

def _generate_templates():
    """Generate candidate rigid body templates for 11 hexagons."""
    s3 = np.sqrt(3)
    templates = []
    
    # Template 1: Hexagonal grid subset (Center + 6 neighbors + 4 from 2nd ring)
    # This is the most compact symmetric arrangement
    t1 = [
        (0, 0),
        (s3, 0),
        (s3 * np.cos(np.radians(60)), s3 * np.sin(np.radians(60))),
        (s3 * np.cos(np.radians(120)), s3 * np.sin(np.radians(120))),
        (-s3, 0),
        (s3 * np.cos(np.radians(240)), s3 * np.sin(np.radians(240))),
        (s3 * np.cos(np.radians(300)), s3 * np.sin(np.radians(300))),
        # 2nd ring: pick 4 that fit well
        (2 * s3 * np.cos(np.radians(30)), 2 * s3 * np.sin(np.radians(30))),
        (2 * s3 * np.cos(np.radians(90)), 2 * s3 * np.sin(np.radians(90))),
        (2 * s3 * np.cos(np.radians(150)), 2 * s3 * np.sin(np.radians(150))),
        (2 * s3 * np.cos(np.radians(210)), 2 * s3 * np.sin(np.radians(210))),
    ]
    templates.append(t1)
    
    # Template 2: Linear-ish compact
    t2 = [
        (0, 0),
        (2, 0),
        (-2, 0),
        (1, s3),
        (-1, s3),
        (1, -s3),
        (-1, -s3),
        (3, s3),
        (-3, s3),
        (3, -s3),
        (-3, -s3),
    ]
    templates.append(t2)
    
    # Template 3: "Tetris" like blocks
    # 3 rows of 4,4,3
    dy = 1.5
    t3 = [
        (-1.5, dy), (0, dy), (1.5, dy), (3.0, dy),
        (-3.0, 0), (-1.5, 0), (0, 0), (1.5, 0),
        (-1.5, -dy), (0, -dy), (1.5, -dy),
    ]
    templates.append(t3)
    
    # Template 4: Center + 10 in a ring? (Too big, but let's try a compact ring)
    # Actually, just add a few random-ish valid configurations from grid
    # Grid coords: (col, row) -> x = col * 2 + (row % 2), y = row * 1.5
    # Let's generate all subsets of size 11 from a 5x5 grid region and filter by connectivity/compactness? 
    # Too slow. Just stick to the hand-crafted ones and maybe perturb them.
    
    return templates

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580.0 # Safety margin
    
    outer_data = np.array([0.0, 0.0, 0.0])
    best_inner = None
    best_side = float('inf')
    
    templates = _generate_templates()
    
    for t_idx, template in enumerate(templates):
        if time.time() > deadline - 30:
            break
            
        # We optimize the rigid body: x_offset, y_offset, rotation
        # Initial guess: 0, 0, 0
        x0 = [0, 0, 0]
        
        def rigid_obj(params):
            x_off, y_off, rot_deg = params
            rot_rad = np.radians(rot_deg)
            cos_r = np.cos(rot_rad)
            sin_r = np.sin(rot_rad)
            
            inner_data = np.zeros((len(template), 3))
            for i, (rx, ry) in enumerate(template):
                x_rot = rx * cos_r - ry * sin_r
                y_rot = rx * sin_r + ry * cos_r
                inner_data[i, 0] = x_rot + x_off
                inner_data[i, 1] = y_rot + y_off
                inner_data[i, 2] = rot_deg # All aligned with global rotation
                
            # Check separation
            if not _check_inner_separation_fast(inner_data, tol=1e-6):
                # Penalty for overlap
                # Calculate minimum overlap depth for penalty
                # For simplicity, return large value + side length
                return 1000.0 + _compute_required_side(inner_data)
            
            req_side = _compute_required_side(inner_data)
            return req_side
            
        try:
            result = minimize(
                rigid_obj,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if result.fun < 500: # Valid
                x_off, y_off, rot_deg = result.x
                rot_rad = np.radians(rot_deg)
                cos_r = np.cos(rot_rad)
                sin_r = np.sin(rot_rad)
                
                inner_data = np.zeros((len(template), 3))
                for i, (rx, ry) in enumerate(template):
                    x_rot = rx * cos_r - ry * sin_r
                    y_rot = rx * sin_r + ry * cos_r
                    inner_data[i, 0] = x_rot + x_off
                    inner_data[i, 1] = y_rot + y_off
                    inner_data[i, 2] = rot_deg
                
                req_side = _compute_required_side(inner_data)
                
                if _validate(inner_data, outer_data, req_side):
                    if req_side < best_side:
                        best_side = req_side
                        best_inner = inner_data.copy()
                        
        except Exception:
            pass
            
    # If we have a good solution, try a local refinement on the full 33 DOF
    # This can squeeze out more performance if the rigid assumption was slightly off
    if best_inner is not None and time.time() < deadline - 60:
        def full_obj(params):
            inner = params.reshape(11, 3)
            if not _check_inner_separation_fast(inner, tol=1e-6):
                return 1000.0 + _compute_required_side(inner)
            return _compute_required_side(inner)
            
        try:
            result = minimize(
                full_obj,
                best_inner.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11}
            )
            if result.fun < 500:
                refined = result.x.reshape(11, 3)
                req_side = _compute_required_side(refined)
                if _validate(refined, outer_data, req_side):
                    if req_side < best_side:
                        best_side = req_side
                        best_inner = refined
        except Exception:
            pass
            
    if best_inner is None:
        # Fallback
        s3 = np.sqrt(3)
        inner_data = np.array([
            [0, 0, 0],
            [s3, 0, 0],
            [s3 * np.cos(np.radians(60)), s3 * np.sin(np.radians(60)), 0],
            [s3 * np.cos(np.radians(120)), s3 * np.sin(np.radians(120)), 0],
            [-s3, 0, 0],
            [s3 * np.cos(np.radians(240)), s3 * np.sin(np.radians(240)), 0],
            [s3 * np.cos(np.radians(300)), s3 * np.sin(np.radians(300)), 0],
            [2 * s3 * np.cos(np.radians(30)), 2 * s3 * np.sin(np.radians(30)), 0],
            [2 * s3 * np.cos(np.radians(90)), 2 * s3 * np.sin(np.radians(90)), 0],
            [2 * s3 * np.cos(np.radians(150)), 2 * s3 * np.sin(np.radians(150)), 0],
            [2 * s3 * np.cos(np.radians(210)), 2 * s3 * np.sin(np.radians(210)), 0],
        ])
        req_side = _compute_required_side(inner_data)
        best_inner = inner_data
        best_side = req_side
        
    return best_inner, outer_data, best_side

# EVOLVE-BLOCK-END
