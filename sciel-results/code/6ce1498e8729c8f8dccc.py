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
    # Project v1 and v2 onto all normals of n1
    proj1_n1 = v1 @ n1.T # (6, 6)
    proj2_n1 = v2 @ n1.T # (6, 6)
    
    min1 = proj1_n1.min(axis=0)
    max1 = proj1_n1.max(axis=0)
    min2 = proj2_n1.min(axis=0)
    max2 = proj2_n1.max(axis=0)
    
    # Separated on normal k if max1[k] < min2[k] - tol OR max2[k] < min1[k] - tol
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

def _compute_required_side(inner_data):
    """Compute minimum side length of outer hexagon (centered at origin, angle 0) to contain all inner hexagons."""
    required_side = 0.0
    # Outer hexagon normals (angle 0): 30, 90, 150, 210, 270, 330 degrees
    # Normals: (cos(30), sin(30)), (0, 1), ...
    # Actually, for a hexagon with vertex at (S, 0), the edge from (S,0) to (S/2, S*sqrt(3)/2) has normal (sqrt(3)/2, 1/2).
    # The projection of point p onto normal n is p . n.
    # The max projection of any vertex of any inner hex onto the outer normals must be <= S * (projection of outer vertex onto normal).
    # For outer hex side S, angle 0:
    # Vertex 0: (S, 0). Normal 0 (outward): (sqrt(3)/2, 1/2).
    # Wait, the normal to the edge adjacent to vertex 0?
    # Let's just use the point-in-poly logic or the projection logic directly.
    # The condition for p to be inside outer hex (center 0, side S, angle 0) is:
    # For each of the 6 outward normals n_k of the outer hex:
    # p . n_k <= S * (v_k . n_k) where v_k is the corresponding outer vertex? No.
    # The support function of the outer hex in direction n_k is h(n_k).
    # For a regular hexagon with side S, the support function in direction n (normal to a face) is S * sqrt(3)/2.
    # The normals to the faces are at 30, 90, 150, 210, 270, 330 degrees.
    # So for each inner vertex p, we check if p . n_k <= S * sqrt(3)/2 for all k.
    # Thus S >= max( p . n_k ) / (sqrt(3)/2).
    
    normals = []
    for k in range(6):
        angle = np.radians(30 + 60 * k)
        normals.append([np.cos(angle), np.sin(angle)])
    normals = np.array(normals) # (6, 2)
    
    max_proj = 0.0
    for i in range(len(inner_data)):
        ix, iy, ia = inner_data[i]
        v = _hex_vertices(ix, iy, 1.0, ia)
        # Project all vertices onto all 6 outer normals
        projs = v @ normals.T # (6, 6)
        max_proj = max(max_proj, np.max(projs))
        
    return max_proj / (np.sqrt(3) / 2)

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580.0 # Safety margin
    
    outer_data = np.array([0.0, 0.0, 0.0])
    
    # Initial configuration: 11 hexagons in a compact grid-like structure
    # Using axial coordinates for hex grid: (q, r) -> x = 1.5*q, y = sqrt(3)/2*q + sqrt(3)*r
    # We want 11 hexagons.
    # A 3x4 grid is 12. Remove one corner?
    # Or the "flower" (7) + 4 more.
    # Let's use a 3x4 block and optimize.
    # Coordinates for 3x4 block (q from -1 to 1, r from -1 to 2? No, let's center it).
    # Let's just use a simple valid initial guess:
    # Center at 0.
    # 6 neighbors.
    # 4 more at distance ~ 2*sqrt(3).
    
    s3 = np.sqrt(3)
    # Initial positions (x, y)
    # 1: (0,0)
    # 2-7: Neighbors
    # 8-11: Second ring
    init_positions = [
        (0, 0),
        (s3, 0),
        (s3/2, s3*np.sqrt(3)/2), # (s3*cos60, s3*sin60) -> s3/2, 3/2? No. s3 * (1/2, sqrt(3)/2) = (s3/2, 3/2).
        # Wait, s3 = sqrt(3).
        # Neighbor at 60 deg: (sqrt(3)*0.5, sqrt(3)*sqrt(3)/2) = (sqrt(3)/2, 1.5).
        (s3/2, 1.5),
        (-s3/2, 1.5),
        (-s3, 0),
        (-s3/2, -1.5),
        (s3/2, -1.5),
        # 4 more:
        # Let's place them at (2*s3, 0), (0, 2*s3), (-2*s3, 0), (0, -2*s3) - these might overlap with neighbors.
        # Better: (2*s3*cos(30), 2*s3*sin(30)) etc.
        # Let's just spread them out a bit more to ensure initial validity.
        (2.5*s3, 0),
        (2.5*s3/2, 2.5*1.5), # Approx
        (-2.5*s3/2, 2.5*1.5),
        (-2.5*s3, 0)
    ]
    
    # Convert to inner_data (x, y, angle=0)
    inner_data = np.array([[p[0], p[1], 0.0] for p in init_positions])
    
    # Ensure initial validity by pushing apart if needed?
    # For now, assume optimizer will fix it. But we need a valid start for minimization to work well.
    # Let's do a quick separation check and if invalid, increase scale.
    scale = 1.0
    while not _check_inner_separation_fast(inner_data, tol=1e-6):
        scale *= 1.1
        inner_data[:, 0] = init_positions[:, 0] * scale
        inner_data[:, 1] = init_positions[:, 1] * scale
        if scale > 10: break
        
    def objective(params):
        # params are 22 positions (x, y) for 11 hexagons
        # Angles are fixed to 0
        inner = np.zeros((11, 3))
        inner[:, 0] = params[0::2]
        inner[:, 1] = params[1::2]
        inner[:, 2] = 0.0
        
        # Check separation
        if not _check_inner_separation_fast(inner, tol=1e-6):
            # Calculate penalty: sum of overlaps?
            # Simple penalty: large value
            return 1000.0 + _compute_required_side(inner)
            
        req_side = _compute_required_side(inner)
        return req_side

    best_side = _compute_required_side(inner_data)
    best_inner = inner_data.copy()
    
    # Optimization
    x0 = np.concatenate([inner_data[:, 0], inner_data[:, 1]])
    
    try:
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        
        if result.fun < 500: # Valid
            refined_params = result.x
            refined_inner = np.zeros((11, 3))
            refined_inner[:, 0] = refined_params[0::2]
            refined_inner[:, 1] = refined_params[1::2]
            refined_inner[:, 2] = 0.0
            
            req_side = _compute_required_side(refined_inner)
            
            if _validate(refined_inner, outer_data, req_side):
                if req_side < best_side:
                    best_side = req_side
                    best_inner = refined_inner.copy()
                    
    except Exception as e:
        pass

    # If optimization failed or didn't improve, try a few random restarts with different initial spreads
    if time.time() < deadline - 60:
        for restart in range(5):
            if time.time() > deadline - 30: break
            # Perturb initial positions
            perturb = np.random.normal(0, 0.5, (11, 2))
            inner_data[:, :2] = init_positions * scale + perturb
            inner_data[:, 2] = 0.0
            
            # Ensure separation
            while not _check_inner_separation_fast(inner_data, tol=1e-6):
                inner_data[:, :2] += np.random.normal(0, 0.1, (11, 2))
                # Simple expansion if stuck
                if np.random.rand() < 0.1:
                    inner_data[:, :2] *= 1.05
            
            x0_r = np.concatenate([inner_data[:, 0], inner_data[:, 1]])
            try:
                result_r = minimize(
                    objective,
                    x0_r,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if result_r.fun < 500:
                    refined_inner = np.zeros((11, 3))
                    refined_inner[:, 0] = result_r.x[0::2]
                    refined_inner[:, 1] = result_r.x[1::2]
                    refined_inner[:, 2] = 0.0
                    req_side = _compute_required_side(refined_inner)
                    if _validate(refined_inner, outer_data, req_side):
                        if req_side < best_side:
                            best_side = req_side
                            best_inner = refined_inner.copy()
            except Exception:
                pass

    return best_inner, outer_data, best_side

# EVOLVE-BLOCK-END
