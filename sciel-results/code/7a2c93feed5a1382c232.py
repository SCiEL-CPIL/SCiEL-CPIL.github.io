"""Direct construction of 12 unit hexagons inside a regular hexagon, minimizing outer side."""
N = 12

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math

def _hex_verts(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _hex_face_normals(angle_deg):
    """Unit outward normals of the 6 faces of a hexagon with given rotation."""
    a = np.radians(angle_deg)
    normals = []
    for i in range(6):
        # Face i is between vertex i and vertex i+1; normal points outward
        theta = a + np.radians(60 * i + 30)
        normals.append([np.cos(theta), np.sin(theta)])
    return np.array(normals)

def _hex_apothem(side):
    """Apothem (distance from center to face midpoint) of a regular hexagon."""
    return side * math.sqrt(3) / 2.0

def _containment_penalty_fast(inner_verts, outer_verts, outer_normals, outer_apothem, outer_cx, outer_cy):
    """Penalty for inner hexagon not being fully inside outer hexagon.
    Uses face-normal formulation: for each outer face, check all inner vertices."""
    penalty = 0.0
    n_out = len(outer_verts)
    for i in range(n_out):
        # Face i: point on face is outer_verts[i] to outer_verts[(i+1)%6]
        # Normal direction: from center toward face midpoint
        v1 = outer_verts[i]
        v2 = outer_verts[(i + 1) % n_out]
        edge = v2 - v1
        edge_len = np.linalg.norm(edge)
        if edge_len < 1e-12:
            continue
        normal = np.array([-edge[1], edge[0]]) / edge_len
        # Check all inner vertices against this face
        for iv in inner_verts:
            rel = iv - v1
            cross = edge[0] * rel[1] - edge[1] * rel[0]
            if cross < 0:
                penalty += (-cross) / edge_len
    return penalty

def _sat_overlap_depth(h1, h2):
    """Minimum penetration depth between two convex polygons using SAT. Returns 0 if disjoint."""
    min_depth = float('inf')
    axes = []
    for poly in [h1, h2]:
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            edge = p2 - p1
            norm = np.linalg.norm(edge)
            if norm < 1e-10:
                continue
            normal = np.array([-edge[1], edge[0]]) / norm
            axes.append(normal)
    
    for axis in axes:
        proj1 = np.dot(h1, axis)
        proj2 = np.dot(h2, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 <= min2 or max2 <= min1:
            return 0.0
        
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_depth:
            min_depth = overlap
    
    return min_depth

def _total_penalty(params, outer_side, outer_center, outer_angle):
    """Total penalty: containment + pairwise disjointness."""
    inner_verts_list = []
    for i in range(N):
        cx = params[3*i]
        cy = params[3*i+1]
        ang = params[3*i+2]
        inner_verts_list.append(_hex_verts(cx, cy, 1.0, ang))
    
    outer_verts = _hex_verts(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    # Containment
    cont_pen = 0.0
    for iv in inner_verts_list:
        cont_pen += _containment_penalty_fast(iv, outer_verts, None, None, outer_center[0], outer_center[1])
    
    # Disjointness
    disj_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            depth = _sat_overlap_depth(inner_verts_list[i], inner_verts_list[j])
            if depth > 0:
                disj_pen += depth
    
    return cont_pen + disj_pen

def _build_honeycomb_config():
    """Build the natural honeycomb: 1 center + 6 ring + 5 outer ring positions."""
    positions = [(0.0, 0.0)]
    # Ring 1: 6 hexagons at distance 2 (touching center)
    for i in range(6):
        ang = np.radians(60 * i)
        positions.append((2.0 * np.cos(ang), 2.0 * np.sin(ang)))
    # Ring 2: 5 more (skip one to get 12 total)
    # The 6 outer positions of ring 2 are at angles 30, 90, 150, 210, 270, 330
    # Distance = 2 * sqrt(3) ≈ 3.464
    r2 = 2.0 * math.sqrt(3)
    for i in range(5):
        ang = np.radians(60 * i + 30)
        positions.append((r2 * np.cos(ang), r2 * np.sin(ang)))
    
    arr = np.array(positions)
    centroid = arr.mean(axis=0)
    return (arr - centroid).tolist()

def _build_lattice_config(nrows, ncols):
    """Build a hexagonal lattice of positions."""
    positions = []
    for r in range(nrows):
        for c in range(ncols):
            x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
            y = r * math.sqrt(3)
            positions.append((x, y))
    arr = np.array(positions)
    centroid = arr.mean(axis=0)
    return (arr - centroid).tolist()

def _build_staggered_config():
    """Two rows of 6, staggered."""
    positions = []
    for c in range(6):
        positions.append((c * 2.0 - 5.0, 0.0))
        positions.append((c * 2.0 - 4.0, math.sqrt(3)))
    arr = np.array(positions)
    centroid = arr.mean(axis=0)
    return (arr - centroid).tolist()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    # Generate candidate initial configurations
    configs = []
    configs.append(_build_honeycomb_config())
    configs.append(_build_lattice_config(3, 4))
    configs.append(_build_lattice_config(4, 3))
    configs.append(_build_lattice_config(2, 6))
    configs.append(_build_staggered_config())
    
    best_score = float('inf')
    best_result = None
    
    # Phase 1: Try multiple outer angles and sides with L-BFGS-B
    outer_angles = [0.0, 30.0]
    outer_sides = [4.0, 3.8, 3.6, 3.4, 3.2, 3.0, 2.8]
    
    for outer_angle in outer_angles:
        for outer_side in outer_sides:
            if time.time() > deadline - 30.0:
                break
            
            outer_center = (0.0, 0.0)
            
            for config in configs:
                if time.time() > deadline - 25.0:
                    break
                
                init_params = []
                for (x, y) in config:
                    init_params.extend([float(x), float(y), 0.0])
                init_params = np.array(init_params)
                
                # Quick check
                pen = _total_penalty(init_params, outer_side, outer_center, outer_angle)
                
                if pen < 1e-6:
                    if outer_side < best_score:
                        best_score = outer_side
                        best_result = (init_params.copy(), outer_center, outer_side, outer_angle)
                    continue
                
                if time.time() > deadline - 25.0:
                    continue
                
                def objective(p):
                    return _total_penalty(p, outer_side, outer_center, outer_angle)
                
                try:
                    result = minimize(
                        objective,
                        init_params,
                        method='L-BFGS-B',
                        options={'maxiter': 60, 'maxfun': 2000, 'ftol': 1e-13, 'gtol': 1e-9}
                    )
                    opt_params = result.x
                    opt_val = result.fun
                except Exception:
                    continue
                
                if opt_val < 1e-6:
                    if outer_side < best_score:
                        best_score = outer_side
                        best_result = (opt_params, outer_center, outer_side, outer_angle)
    
    # Phase 2: Binary search for minimal side length from best found
    if best_result is not None and time.time() < deadline - 10.0:
        opt_params, outer_center, outer_side, outer_angle = best_result
        
        # Binary search for the smallest side length that still admits a valid packing
        low = max(2.5, outer_side - 0.4)
        high = outer_side
        
        for _ in range(12):
            if time.time() > deadline - 5.0:
                break
            mid = (low + high) / 2.0
            if high - low < 1e-5:
                break
            
            try:
                result = minimize(
                    lambda p, s=mid: _total_penalty(p, s, outer_center, outer_angle),
                    opt_params,
                    method='L-BFGS-B',
                    options={'maxiter': 80, 'maxfun': 2500, 'ftol': 1e-14, 'gtol': 1e-11}
                )
                if result.fun < 1e-7:
                    high = mid
                    opt_params = result.x
                    outer_side = mid
                else:
                    low = mid
            except Exception:
                low = mid
        
        # Final high-precision polish at the found side length
        if time.time() < deadline - 3.0:
            try:
                result = minimize(
                    lambda p: _total_penalty(p, outer_side, outer_center, outer_angle),
                    opt_params,
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'maxfun': 4000, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if result.fun < 1e-8:
                    opt_params = result.x
            except Exception:
                pass
        
        best_result = (opt_params, outer_center, outer_side, outer_angle)
    
    # Fallback: guaranteed valid configuration
    if best_result is None:
        inner = []
        pos = _build_lattice_config(3, 4)
        for (x, y) in pos:
            inner.append([float(x), float(y), 0.0])
        return inner, (0.0, 0.0), 5.5, 0.0
    
    opt_params, outer_center, outer_side, outer_angle = best_result
    
    inner = []
    for i in range(N):
        x = float(opt_params[3*i])
        y = float(opt_params[3*i+1])
        ang = float(opt_params[3*i+2])
        inner.append([x, y, ang])
    
    # Final validation
    inner_verts_list = []
    for i in range(N):
        cx, cy, ang = opt_params[3*i], opt_params[3*i+1], opt_params[3*i+2]
        inner_verts_list.append(_hex_verts(cx, cy, 1.0, ang))
    outer_verts = _hex_verts(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    cont_pen = 0.0
    for iv in inner_verts_list:
        cont_pen += _containment_penalty_fast(iv, outer_verts, None, None, outer_center[0], outer_center[1])
    disj_pen = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = _sat_overlap_depth(inner_verts_list[i], inner_verts_list[j])
            if d > 0:
                disj_pen += d
    
    if cont_pen > 1e-4 or disj_pen > 1e-4:
        inner = []
        pos = _build_lattice_config(3, 4)
        for (x, y) in pos:
            inner.append([float(x), float(y), 0.0])
        return inner, (0.0, 0.0), 5.5, 0.0
    
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)

# EVOLVE-BLOCK-END
