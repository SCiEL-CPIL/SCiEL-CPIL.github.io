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
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _hex_verts_batch(params, side=1.0):
    """Vectorized vertex computation for all inner hexagons from params array [x,y,ang]*N."""
    n = len(params) // 3
    cx = params[0::3]
    cy = params[1::3]
    ang = np.radians(params[2::3])
    angles = ang[:, None] + np.radians(60 * np.arange(6))
    vx = cx[:, None] + side * np.cos(angles)
    vy = cy[:, None] + side * np.sin(angles)
    return vx.reshape(-1, 6), vy.reshape(-1, 6)

def _containment_penalty(inner_verts, outer_verts):
    """Penalty for inner hexagon not being fully inside outer hexagon."""
    penalty = 0.0
    n_out = len(outer_verts)
    for iv in inner_verts:
        for i in range(n_out):
            v1 = outer_verts[i]
            v2 = outer_verts[(i + 1) % n_out]
            edge = v2 - v1
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-12:
                continue
            rel = iv - v1
            cross = edge[0] * rel[1] - edge[1] * rel[0]
            if cross < 0:
                penalty += (-cross) / edge_len
    return penalty

def _containment_penalty_batch(inner_verts_list, outer_verts):
    """Vectorized containment penalty for all inner hexagons."""
    n_out = len(outer_verts)
    edges = []
    edge_lens = []
    for i in range(n_out):
        v1 = outer_verts[i]
        v2 = outer_verts[(i + 1) % n_out]
        edge = v2 - v1
        elen = np.linalg.norm(edge)
        if elen < 1e-12:
            continue
        edges.append(edge)
        edge_lens.append(elen)
    
    penalty = 0.0
    for iv in inner_verts_list:
        for edge, elen in zip(edges, edge_lens):
            for pt in iv:
                rel = pt - outer_verts[0] if edge is edges[0] else pt - outer_verts[edges.index(edge)]
                # Recompute properly
                pass
        # Fallback to per-vertex
        for i in range(n_out):
            v1 = outer_verts[i]
            v2 = outer_verts[(i + 1) % n_out]
            edge = v2 - v1
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-12:
                continue
            for pt in iv:
                rel = pt - v1
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

def _compute_penalty(params, outer_side, outer_center, outer_angle):
    """Compute all inner hex vertices and return containment + disjointness penalties."""
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
        cont_pen += _containment_penalty(iv, outer_verts)
    
    # Disjointness
    disj_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            depth = _sat_overlap_depth(inner_verts_list[i], inner_verts_list[j])
            if depth > 0:
                disj_pen += depth
    
    return cont_pen + disj_pen

def _compute_penalty_fast(params, outer_side, outer_center, outer_angle):
    """Optimized penalty computation with precomputed outer edges."""
    # Precompute outer hex edges
    outer_verts = _hex_verts(outer_center[0], outer_center[1], outer_side, outer_angle)
    outer_edges = []
    for i in range(6):
        v1 = outer_verts[i]
        v2 = outer_verts[(i + 1) % 6]
        edge = v2 - v1
        elen = np.linalg.norm(edge)
        outer_edges.append((v1, edge, elen))
    
    # Compute inner vertices
    inner_verts_list = []
    for i in range(N):
        cx = params[3*i]
        cy = params[3*i+1]
        ang = params[3*i+2]
        inner_verts_list.append(_hex_verts(cx, cy, 1.0, ang))
    
    # Containment penalty
    cont_pen = 0.0
    for iv in inner_verts_list:
        for v1, edge, elen in outer_edges:
            for pt in iv:
                rel = pt - v1
                cross = edge[0] * rel[1] - edge[1] * rel[0]
                if cross < 0:
                    cont_pen += (-cross) / elen
    
    # Disjointness penalty
    disj_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            depth = _sat_overlap_depth(inner_verts_list[i], inner_verts_list[j])
            if depth > 0:
                disj_pen += depth
    
    return cont_pen + disj_pen

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    def build_hexagonal_lattice(nrows, ncols):
        """Build a hexagonal lattice of positions (offset rows)."""
        positions = []
        for r in range(nrows):
            for c in range(ncols):
                x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
                y = r * np.sqrt(3)
                positions.append((x, y))
        return positions
    
    def build_ring_config():
        """6 around center + 5 more in second ring."""
        positions = [(0.0, 0.0)]
        for i in range(6):
            ang = np.radians(60 * i)
            positions.append((2.0 * np.cos(ang), 2.0 * np.sin(ang)))
        for i in range(6):
            ang = np.radians(60 * i + 30)
            positions.append((3.464 * np.cos(ang), 3.464 * np.sin(ang)))
        return positions[:12]
    
    # Generate candidate initial configurations
    configs = []
    
    pos_4x3 = build_hexagonal_lattice(4, 3)
    if len(pos_4x3) == N:
        arr = np.array(pos_4x3)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())
    
    pos_3x4 = build_hexagonal_lattice(3, 4)
    if len(pos_3x4) == N:
        arr = np.array(pos_3x4)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())
    
    pos_2x6 = build_hexagonal_lattice(2, 6)
    if len(pos_2x6) == N:
        arr = np.array(pos_2x6)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())
    
    ring_pos = build_ring_config()
    if len(ring_pos) == N:
        arr = np.array(ring_pos)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())
    
    pos_6x2 = build_hexagonal_lattice(6, 2)
    if len(pos_6x2) == N:
        arr = np.array(pos_6x2)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())
    
    line_pos = [(float(i - 5.5), 0.0) for i in range(12)]
    configs.append(line_pos)
    
    # Additional configurations: 2-3-3-2-2 arrangement and compact cluster
    pos_23322 = [
        (0.0, 0.0), (2.0, 0.0), (4.0, 0.0),
        (1.0, np.sqrt(3)), (3.0, np.sqrt(3)), (5.0, np.sqrt(3)),
        (2.0, 2*np.sqrt(3)), (4.0, 2*np.sqrt(3)), (6.0, 2*np.sqrt(3)),
        (1.0, 3*np.sqrt(3)), (3.0, 3*np.sqrt(3)), (5.0, 3*np.sqrt(3))
    ]
    if len(pos_23322) == N:
        arr = np.array(pos_23322)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())
    
    # Compact 3x4 with slight compression
    pos_3x4_compact = []
    for r in range(3):
        for c in range(4):
            x = c * 1.95 + (0.975 if r % 2 == 1 else 0.0)
            y = r * np.sqrt(3) * 0.98
            pos_3x4_compact.append((x, y))
    if len(pos_3x4_compact) == N:
        arr = np.array(pos_3x4_compact)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())
    
    # Rotated 4x3
    pos_4x3_rot = build_hexagonal_lattice(4, 3)
    if len(pos_4x3_rot) == N:
        arr = np.array(pos_4x3_rot)
        centroid = arr.mean(axis=0)
        arr = arr - centroid
        # Rotate 30 degrees
        theta = np.radians(30)
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        arr = arr @ rot.T
        configs.append(arr.tolist())
    
    # Compact ring: 6 inner + 6 outer with tighter spacing
    pos_compact_ring = [(0.0, 0.0)]
    for i in range(6):
        ang = np.radians(60 * i)
        pos_compact_ring.append((1.9 * np.cos(ang), 1.9 * np.sin(ang)))
    for i in range(6):
        ang = np.radians(60 * i + 30)
        pos_compact_ring.append((3.3 * np.cos(ang), 3.3 * np.sin(ang)))
    if len(pos_compact_ring) == N:
        arr = np.array(pos_compact_ring)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())
    
    best_score = float('inf')
    best_result = None
    
    # Phase 1: Quick L-BFGS-B search with multiple outer angles and sides
    outer_angles = [0.0, 30.0, 15.0, 45.0]
    outer_sides = [5.0, 4.5, 4.2, 4.0, 3.8, 3.6, 3.5, 3.4, 3.3, 3.2, 3.1, 3.0]
    
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
                
                pen = _compute_penalty_fast(init_params, outer_side, outer_center, outer_angle)
                
                if pen < 1e-6:
                    if outer_side < best_score:
                        best_score = outer_side
                        best_result = (init_params.copy(), outer_center, outer_side, outer_angle)
                    continue
                
                if time.time() > deadline - 25.0:
                    continue
                
                def objective(p, os=outer_side, oa=outer_angle):
                    return _compute_penalty_fast(p, os, outer_center, oa)
                
                try:
                    result = minimize(
                        objective,
                        init_params,
                        method='L-BFGS-B',
                        options={'maxiter': 100, 'maxfun': 3000, 'ftol': 1e-14, 'gtol': 1e-10}
                    )
                    opt_params = result.x
                    opt_val = result.fun
                except Exception:
                    continue
                
                if opt_val < 1e-6:
                    if outer_side < best_score:
                        best_score = outer_side
                        best_result = (opt_params, outer_center, outer_side, outer_angle)
    
    # Phase 2: Binary search for minimal side length with more iterations
    if best_result is not None and time.time() < deadline - 5.0:
        opt_params, outer_center, outer_side, outer_angle = best_result
        
        # Binary search for the smallest side length that still admits a valid packing
        low = max(2.8, outer_side - 0.5)
        high = outer_side
        
        for _ in range(15):
            if time.time() > deadline - 5.0:
                break
            mid = (low + high) / 2.0
            if high - low < 1e-6:
                break
            
            try:
                result = minimize(
                    lambda p, s=mid: _compute_penalty_fast(p, s, outer_center, outer_angle),
                    opt_params,
                    method='L-BFGS-B',
                    options={'maxiter': 120, 'maxfun': 3500, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if result.fun < 1e-8:
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
                    lambda p: _compute_penalty_fast(p, outer_side, outer_center, outer_angle),
                    opt_params,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'maxfun': 8000, 'ftol': 1e-16, 'gtol': 1e-14}
                )
                if result.fun < 1e-8:
                    opt_params = result.x
            except Exception:
                pass
        
        best_result = (opt_params, outer_center, outer_side, outer_angle)
    
    # Fallback
    if best_result is None:
        inner = []
        pos = build_hexagonal_lattice(3, 4)
        arr = np.array(pos)
        centroid = arr.mean(axis=0)
        for (x, y) in (arr - centroid):
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
    
    cont_pen = sum(_containment_penalty(iv, outer_verts) for iv in inner_verts_list)
    disj_pen = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = _sat_overlap_depth(inner_verts_list[i], inner_verts_list[j])
            if d > 0:
                disj_pen += d
    
    if cont_pen > 1e-3 or disj_pen > 1e-3:
        inner = []
        pos = build_hexagonal_lattice(3, 4)
        arr = np.array(pos)
        centroid = arr.mean(axis=0)
        for (x, y) in (arr - centroid):
            inner.append([float(x), float(y), 0.0])
        return inner, (0.0, 0.0), 5.5, 0.0
    
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)

# EVOLVE-BLOCK-END
