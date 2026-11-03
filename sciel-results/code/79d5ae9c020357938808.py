"""Packing 11 unit hexagons in a regular hexagon; minimize outer side length.

Structural change: SDF-based differentiable repulsion to settle hexagons into
dense equilibrium, then LP for exact enclosing hexagon.
"""
import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # safety margin within 150s limit

def _hex_vertex_offsets(angle_deg):
    """Return 6 vertex offsets for a unit hexagon rotated by angle_deg."""
    angle_rad = math.radians(angle_deg)
    offsets = np.zeros((6, 2))
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        offsets[k, 0] = math.cos(a)
        offsets[k, 1] = math.sin(a)
    return offsets

def _signed_distance_to_hexagon(point, center, angle_deg, offsets_cache=None):
    """
    Compute signed distance from a point to a unit regular hexagon centered at center
    with rotation angle_deg. Positive if outside, negative if inside.
    Uses the fact that a regular hexagon is the intersection of 6 half-planes.
    """
    if offsets_cache is None:
        offsets_cache = _hex_vertex_offsets(angle_deg)
    
    # The hexagon is defined by 6 half-planes. For a regular hexagon with "radius" 1
    # (distance from center to vertex = 1), the inradius (distance from center to edge) is sqrt(3)/2.
    # The 6 normals point from center toward each edge midpoint.
    # Edge normals are at angles: angle_deg + 30 + 60*k degrees (for k=0..5)
    # Wait, let me think more carefully.
    # A regular hexagon with vertices at angles 0, 60, 120, 180, 240, 300 (before rotation)
    # has edges between consecutive vertices. The normal to the edge between vertices at 
    # angles (60k-30) and (60k+30) points at angle 60k.
    # Actually, for a unit hexagon (circumradius=1), the inradius is sqrt(3)/2.
    # The half-plane constraint for each edge: n_k · (p - center) <= inradius
    # where n_k is the unit normal pointing outward from center toward edge k.
    
    # For a hexagon with vertex at angle θ, the edge opposite to that vertex 
    # has its outward normal at angle θ + 180. But actually, each edge is between
    # two adjacent vertices. Let me use a cleaner formulation.
    
    # For a regular hexagon with circumradius R=1, the 6 edge normals (outward) 
    # are at angles: 30 + 60*k + angle_deg for k=0..5 (if vertices are at 0+60k+angle_deg)
    # No wait. If vertices are at angles 0, 60, 120, ... (before rotation), then:
    # - Vertex 0 is at (1, 0), vertex 1 is at (cos60, sin60) = (0.5, sqrt(3)/2)
    # - Edge between v0 and v1: midpoint at angle 30, distance from center to edge = sqrt(3)/2
    # - Outward normal to this edge points at angle 30.
    # So edge normals are at angles 30 + 60*k for k=0..5.
    
    angles = np.array([math.radians(angle_deg + 30 + 60 * k) for k in range(6)])
    normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6, 2)
    
    # Signed distance: for each edge, the constraint is n_k · (p - c) <= inradius
    # The "violation" for edge k is n_k · (p - c) - inradius
    # The signed distance to the hexagon is max over all edges of (n_k · (p - c) - inradius)
    # But this is only correct when the point is outside. When inside, all values are negative,
    # and the signed distance should be negative (max of negative values = least negative = closest to 0).
    
    inradius = math.sqrt(3) / 2
    rel = point - center
    projections = normals @ rel  # (6,)
    violations = projections - inradius  # (6,)
    
    # Signed distance = max(violations) when outside, but when inside it's the max (closest to 0)
    # Actually, the signed distance to a convex polytope defined by half-planes n_k·x <= b_k
    # is max_k (n_k·x - b_k). This is correct both inside and outside.
    return float(np.max(violations))

def _sdf_between_hexagons(center1, angle1, center2, angle2, offsets_cache1=None, offsets_cache2=None):
    """
    Compute minimum signed distance between two unit hexagons.
    Positive if separated, negative if overlapping.
    We check vertices of hex1 against hex2 and vice versa.
    """
    if offsets_cache1 is None:
        offsets_cache1 = _hex_vertex_offsets(angle1)
    if offsets_cache2 is None:
        offsets_cache2 = _hex_vertex_offsets(angle2)
    
    # Vertices of hex1
    verts1 = center1 + offsets_cache1  # (6, 2)
    # Vertices of hex2
    verts2 = center2 + offsets_cache2  # (6, 2)
    
    # For each vertex of hex1, compute signed distance to hex2
    min_dist = float('inf')
    for v in verts1:
        d = _signed_distance_to_hexagon(v, center2, angle2, offsets_cache2)
        if d < min_dist:
            min_dist = d
    
    # For each vertex of hex2, compute signed distance to hex1
    for v in verts2:
        d = _signed_distance_to_hexagon(v, center1, angle1, offsets_cache1)
        if d < min_dist:
            min_dist = d
    
    return min_dist

def _smooth_overlap_penalty(params, angles, strength=100.0, smooth=0.01):
    """
    Compute a smooth penalty for overlaps between hexagons.
    params: array of 2*N values [x0, y0, x1, y1, ...]
    angles: list of N angles in degrees
    Returns: scalar penalty value and gradient (2*N,)
    """
    n = len(angles)
    centers = params.reshape(n, 2)
    
    # Precompute vertex offsets
    offsets_list = [_hex_vertex_offsets(a) for a in angles]
    
    penalty = 0.0
    grad = np.zeros(2 * n)
    
    for i in range(n):
        for j in range(i + 1, n):
            # Compute SDF between hex i and hex j
            # We need the gradient of the SDF with respect to positions
            # For now, use a simple approach: sample the SDF and use finite differences
            # Actually, let's compute it analytically for the vertex-based SDF
            
            c_i = centers[i]
            c_j = centers[j]
            
            # Check all vertices of hex i against hex j
            min_d = float('inf')
            min_grad_i = np.zeros(2)
            min_grad_j = np.zeros(2)
            
            for k in range(6):
                v = c_i + offsets_list[i][k]
                # SDF of v to hex j
                angles_j = np.array([math.radians(angles[j] + 30 + 60 * m) for m in range(6)])
                normals_j = np.column_stack([np.cos(angles_j), np.sin(angles_j)])
                inradius = math.sqrt(3) / 2
                
                rel = v - c_j
                projections = normals_j @ rel
                violations = projections - inradius
                idx_max = np.argmax(violations)
                d = violations[idx_max]
                
                if d < min_d:
                    min_d = d
                    # Gradient with respect to v (and thus c_i)
                    min_grad_i = normals_j[idx_max].copy()
                    # Gradient with respect to c_j
                    min_grad_j = -normals_j[idx_max].copy()
            
            # Check all vertices of hex j against hex i
            for k in range(6):
                v = c_j + offsets_list[j][k]
                angles_i = np.array([math.radians(angles[i] + 30 + 60 * m) for m in range(6)])
                normals_i = np.column_stack([np.cos(angles_i), np.sin(angles_i)])
                inradius = math.sqrt(3) / 2
                
                rel = v - c_i
                projections = normals_i @ rel
                violations = projections - inradius
                idx_max = np.argmax(violations)
                d = violations[idx_max]
                
                if d < min_d:
                    min_d = d
                    min_grad_i = -normals_i[idx_max].copy()
                    min_grad_j = normals_i[idx_max].copy()
            
            # Smooth penalty: if min_d < 0, we have overlap
            # Use a smooth approximation: penalty = strength * max(0, -min_d)^2 / (smooth + max(0,-min_d))
            # Or simpler: penalty = strength * (max(0, -min_d)) ** 2
            if min_d < 0:
                overlap = -min_d
                penalty += strength * overlap * overlap
                # Gradient: d(penalty)/d(overlap) = 2 * strength * overlap
                # d(overlap)/d(c_i) = -min_grad_i (since overlap = -min_d, and min_d increases with c_i in direction min_grad_i)
                # Actually: min_d = SDF, so d(min_d)/d(c_i) = min_grad_i
                # overlap = -min_d, so d(overlap)/d(c_i) = -min_grad_i
                # d(penalty)/d(c_i) = 2 * strength * overlap * (-min_grad_i)
                # Wait, let me redo this.
                # penalty = strength * overlap^2 where overlap = max(0, -min_d)
                # d(penalty)/d(c_i) = 2 * strength * overlap * d(overlap)/d(c_i)
                # d(overlap)/d(c_i) = -d(min_d)/d(c_i) = -min_grad_i
                grad[i*2] += 2 * strength * overlap * (-min_grad_i[0])
                grad[i*2+1] += 2 * strength * overlap * (-min_grad_i[1])
                grad[j*2] += 2 * strength * overlap * (-min_grad_j[0])
                grad[j*2+1] += 2 * strength * overlap * (-min_grad_j[1])
    
    return penalty, grad

def _sdf_settle(positions, angles, max_iter=200, step_size=0.01, strength=50.0):
    """
    Use gradient descent with SDF-based repulsion to settle hexagons into a dense configuration.
    positions: list of (x, y) initial positions
    angles: list of N angles in degrees
    Returns: list of (x, y) settled positions
    """
    n = len(angles)
    params = np.array([p[0] for p in positions] + [p[1] for p in positions])
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 30.0:
            break
        
        penalty, grad = _smooth_overlap_penalty(params, angles, strength=strength)
        
        if penalty < 1e-10:
            break
        
        # Gradient descent step
        params -= step_size * grad
        step_size *= 0.995  # Slow decay
    
    return [(params[i], params[i + n]) for i in range(n)]

def _min_enclosing_hex_side_lp(inner_hexes, angle_deg, all_vertices=None):
    """For a fixed outer hexagon angle, find optimal center and side length using LP."""
    if all_vertices is None:
        n_inner = len(inner_hexes)
        all_vertices = np.zeros((n_inner * 6, 2))
        for idx, (ix, iy, iangle) in enumerate(inner_hexes):
            angle_rad = math.radians(iangle)
            for k in range(6):
                a = angle_rad + math.pi * k / 3
                all_vertices[idx * 6 + k, 0] = ix + math.cos(a)
                all_vertices[idx * 6 + k, 1] = iy + math.sin(a)
    
    n_pts = all_vertices.shape[0]
    if n_pts == 0:
        return 0.0, 0.0, 0.0, angle_deg
    
    angles = np.array([math.radians(angle_deg + 30 + 60 * k) for k in range(6)])
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    rhs = -(all_vertices @ normals.T)
    b_ub = rhs.T.flatten()
    
    A_ub_parts = np.column_stack([
        np.repeat(-normals[:, 0], n_pts),
        np.repeat(-normals[:, 1], n_pts),
        np.full(6 * n_pts, -1.0)
    ])
    
    A_ub = np.vstack([A_ub_parts, [[0.0, 0.0, -1.0]]])
    b_ub = np.append(b_ub, 0.0)
    
    c = np.array([0.0, 0.0, 1.0])
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
    
    if result.success:
        cx_opt = result.x[0]
        cy_opt = result.x[1]
        r_opt = result.x[2]
    else:
        cx_mean = all_vertices[:, 0].mean()
        cy_mean = all_vertices[:, 1].mean()
        centered = all_vertices - np.array([cx_mean, cy_mean])
        projs = centered @ normals.T
        r_opt = projs.max()
        if r_opt < 0:
            r_opt = 0.0
        cx_opt = cx_mean
        cy_opt = cy_mean
    
    side = r_opt / (math.sqrt(3) / 2)
    return cx_opt, cy_opt, side, angle_deg

def _compute_all_vertices(inner_hexes):
    """Precompute all vertices of inner hexagons."""
    n_inner = len(inner_hexes)
    verts = np.zeros((n_inner * 6, 2))
    for idx, (ix, iy, iangle) in enumerate(inner_hexes):
        angle_rad = math.radians(iangle)
        for k in range(6):
            a = angle_rad + math.pi * k / 3
            verts[idx * 6 + k, 0] = ix + math.cos(a)
            verts[idx * 6 + k, 1] = iy + math.sin(a)
    return verts

def _min_enclosing_hex_side(inner_hexes):
    """Find minimum side length of a regular hexagon that contains all inner hexagons."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    all_vertices = _compute_all_vertices(inner_hexes)
    
    # Coarse sweep
    for angle_deg in np.arange(0, 60, 2.0):
        cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg), all_vertices)
        if side < best_side:
            best_side = side
            best_center = (cx, cy)
            best_angle = float(angle_deg)
    
    # Fine sweep
    if best_side < float('inf'):
        for angle_deg in np.arange(max(0.0, best_angle - 5), min(60.0, best_angle + 5), 0.25):
            cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg), all_vertices)
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)
    
    # Very fine sweep
    if best_side < float('inf') and time.time() < _DEADLINE - 20.0:
        for angle_deg in np.arange(max(0.0, best_angle - 1.0), min(60.0, best_angle + 1.0), 0.05):
            cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg), all_vertices)
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)
    
    # Bounded scalar minimization polish
    if best_side < float('inf') and time.time() < _DEADLINE - 10.0:
        def objective_angle(ang_p):
            ang_p = ang_p % 60.0
            _, _, side_p, _ = _min_enclosing_hex_side_lp(inner_hexes, float(ang_p), all_vertices)
            return side_p
        
        try:
            bounds = (max(0.0, best_angle - 1.0), min(60.0, best_angle + 1.0))
            res = minimize(objective_angle, x0=best_angle, bounds=[bounds], method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-12})
            if res.success:
                ang_opt = float(res.x[0] % 60.0)
                cx_v, cy_v, side_v, _ = _min_enclosing_hex_side_lp(inner_hexes, ang_opt, all_vertices)
                if side_v < best_side:
                    best_side = side_v
                    best_center = (cx_v, cy_v)
                    best_angle = ang_opt
        except Exception:
            pass
    
    return best_center[0], best_center[1], best_side, best_angle

def _build_hex_lattice_positions(n_hex, scale=1.001):
    """Build positions on a hexagonal lattice, selecting n_hex closest to origin."""
    sqrt3 = math.sqrt(3)
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
    
    candidates = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            dist_sq = x * x + y * y
            candidates.append((dist_sq, x, y))
    
    candidates.sort(key=lambda t: t[0])
    
    positions = []
    for _, x, y in candidates:
        positions.append((x, y))
        if len(positions) == n_hex:
            break
    
    return positions

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    sqrt3 = math.sqrt(3)
    best_result = None
    best_side = float('inf')
    
    # Try multiple initial configurations with different orientations
    config_angles = [0.0, 30.0]
    
    for inner_angle in config_angles:
        if time.time() > _DEADLINE - 20.0:
            break
        
        # Generate initial lattice positions
        positions = _build_hex_lattice_positions(N, scale=1.001)
        angles = [inner_angle] * N
        
        # SDF-based settling to remove any slight overlaps and optimize spacing
        if time.time() < _DEADLINE - 40.0:
            settled_positions = _sdf_settle(positions, angles, max_iter=150, step_size=0.005, strength=30.0)
        else:
            settled_positions = positions
        
        # Create inner hexagons from settled positions
        inner = [[p[0], p[1], inner_angle] for p in settled_positions]
        
        # Find optimal enclosing hexagon
        if time.time() < _DEADLINE - 15.0:
            cx, cy, side, angle = _min_enclosing_hex_side(inner)
        else:
            # Quick estimate
            all_v = _compute_all_vertices(inner)
            cx = all_v[:, 0].mean()
            cy = all_v[:, 1].mean()
            max_proj = 0
            for ang_deg in [0.0, 30.0]:
                ang_rad = math.radians(ang_deg + 30)
                n = np.array([math.cos(ang_rad), math.sin(ang_rad)])
                projs = (all_v - np.array([cx, cy])) @ n
                if projs.max() > max_proj:
                    max_proj = projs.max()
            side = max_proj / (math.sqrt(3) / 2) + 0.05
            angle = 0.0
        
        if side < best_side:
            best_side = side
            best_result = (inner, (cx, cy), side, angle)
    
    # Try rotated lattice configurations
    for rot_deg in [5.0, 10.0, 15.0, 20.0]:
        if time.time() > _DEADLINE - 25.0:
            break
        
        rot_rad = math.radians(rot_deg)
        positions = _build_hex_lattice_positions(N, scale=1.001)
        
        # Rotate positions
        rotated_positions = []
        for (px, py) in positions:
            rx = px * math.cos(rot_rad) - py * math.sin(rot_rad)
            ry = px * math.sin(rot_rad) + py * math.cos(rot_rad)
            rotated_positions.append((rx, ry))
        
        angles = [30.0 + rot_deg] * N
        
        # SDF settling
        if time.time() < _DEADLINE - 40.0:
            settled_positions = _sdf_settle(rotated_positions, angles, max_iter=100, step_size=0.005, strength=30.0)
        else:
            settled_positions = rotated_positions
        
        inner = [[p[0], p[1], angles[i]] for i, p in enumerate(settled_positions)]
        
        if time.time() < _DEADLINE - 15.0:
            cx, cy, side, angle = _min_enclosing_hex_side(inner)
        else:
            all_v = _compute_all_vertices(inner)
            cx = all_v[:, 0].mean()
            cy = all_v[:, 1].mean()
            max_proj = 0
            for ang_deg in [0.0, 30.0]:
                ang_rad = math.radians(ang_deg + 30)
                n = np.array([math.cos(ang_rad), math.sin(ang_rad)])
                projs = (all_v - np.array([cx, cy])) @ n
                if projs.max() > max_proj:
                    max_proj = projs.max()
            side = max_proj / (math.sqrt(3) / 2) + 0.05
            angle = 0.0
        
        if side < best_side:
            best_side = side
            best_result = (inner, (cx, cy), side, angle)
    
    # Local optimization of positions using Nelder-Mead with smooth SDF penalty
    if best_result is not None and time.time() < _DEADLINE - 35.0:
        best_inner = best_result[0]
        best_pos = [(h[0], h[1]) for h in best_inner]
        best_angles = [h[2] for h in best_inner]
        
        def position_objective(params):
            pos_x = params[:N]
            pos_y = params[N:2*N]
            
            # Check for overlaps using SDF
            centers = np.array([pos_x[i] for i in range(N)] + [pos_y[i] for i in range(N)]).reshape(N, 2)
            
            for i in range(N):
                for j in range(i + 1, N):
                    d = _sdf_between_hexagons(centers[i], best_angles[i], centers[j], best_angles[j])
                    if d < -0.01:  # Significant overlap
                        return best_side + 10.0 * abs(d)
            
            new_inner = [[pos_x[i], pos_y[i], best_angles[i]] for i in range(N)]
            
            if time.time() < _DEADLINE - 15.0:
                cx, cy, side, angle = _min_enclosing_hex_side(new_inner)
                return side
            else:
                all_v = _compute_all_vertices(new_inner)
                cx = all_v[:, 0].mean()
                cy = all_v[:, 1].mean()
                max_proj = 0
                for ang_deg in [0.0, 30.0]:
                    ang_rad = math.radians(ang_deg + 30)
                    n = np.array([math.cos(ang_rad), math.sin(ang_rad)])
                    projs = (all_v - np.array([cx, cy])) @ n
                    if projs.max() > max_proj:
                        max_proj = projs.max()
                return max_proj / (math.sqrt(3) / 2)
        
        x0 = []
        for (px, py) in best_pos:
            x0.extend([px, py])
        
        try:
            res = minimize(position_objective, x0, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.success:
                pos_x = res.x[:N]
                pos_y = res.x[N:2*N]
                new_inner = [[pos_x[i], pos_y[i], best_angles[i]] for i in range(N)]
                
                if time.time() < _DEADLINE - 10.0:
                    cx, cy, side, angle = _min_enclosing_hex_side(new_inner)
                else:
                    all_v = _compute_all_vertices(new_inner)
                    cx = all_v[:, 0].mean()
                    cy = all_v[:, 1].mean()
                    max_proj = 0
                    for ang_deg in [0.0, 30.0]:
                        ang_rad = math.radians(ang_deg + 30)
                        n = np.array([math.cos(ang_rad), math.sin(ang_rad)])
                        projs = (all_v - np.array([cx, cy])) @ n
                        if projs.max() > max_proj:
                            max_proj = projs.max()
                    side = max_proj / (math.sqrt(3) / 2)
                    angle = 0.0
                
                if side < best_side - 1e-8:
                    best_side = side
                    best_result = (new_inner, (cx, cy), side, angle)
        except Exception:
            pass
    
    if best_result is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result
