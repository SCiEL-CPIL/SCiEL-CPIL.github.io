# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, d, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _edge_normals(verts):
    n = len(verts)
    normals = np.zeros((n, 2))
    for i in range(n):
        p1 = verts[i]
        p2 = verts[(i+1) % n]
        edge = p2 - p1
        normal = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normals[i] = normal / norm
    return normals

def _sat_overlap(verts1, normals1, verts2, normals2):
    """
    Returns (overlap, mtv_axis, mtv_sign) for two convex polygons.
    overlap > 0 means they overlap. mtv_axis is the unit normal of the minimum overlap axis.
    mtv_sign: +1 means push verts1 in +mtv_axis direction, -1 means push in -mtv_axis.
    If separated, returns (0, None, 0).
    """
    all_normals = np.vstack([normals1, normals2])
    min_overlap = float('inf')
    best_axis = None
    best_sign = 0
    
    for normal in all_normals:
        proj1 = np.dot(verts1, normal)
        proj2 = np.dot(verts2, normal)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 - 1e-6:
            return 0.0, None, 0
        elif max2 < min1 - 1e-6:
            return 0.0, None, 0
        
        # Overlap on this axis
        # If max1 < min2: overlap = min2 - max1, push 1 in +normal
        # If max2 < min1: overlap = min1 - max2, push 1 in -normal
        # Otherwise they overlap on this axis
        if max1 < min2:
            overlap = min2 - max1
            sign = 1  # push verts1 in +normal
        elif max2 < min1:
            overlap = min1 - max2
            sign = -1  # push verts1 in -normal
        else:
            # Intervals overlap; compute the penetration depth
            # The amount by which they interpenetrate
            # Use the minimum of the two "gap" measures
            # Actually for SAT we need the axis with minimum overlap
            # Overlap = min(max1, max2) - max(min1, min2)
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < 0:
                return 0.0, None, 0
            # Determine sign: push in direction that reduces overlap
            # If center of proj1 < center of proj2, push 1 in -normal
            c1 = (min1 + max1) / 2
            c2 = (min2 + max2) / 2
            sign = 1 if c1 > c2 else -1
        
        if overlap < min_overlap:
            min_overlap = overlap
            best_axis = normal
            best_sign = sign
    
    return min_overlap, best_axis, best_sign

def _compute_all_verts_normals(inner):
    """Precompute vertices and normals for all 12 hexagons."""
    centers = inner[:, :2]
    angles = inner[:, 2]
    all_verts = np.zeros((12, 6, 2))
    all_normals = np.zeros((12, 6, 2))
    for i in range(12):
        verts = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
        all_verts[i] = verts
        all_normals[i] = _edge_normals(verts)
    return all_verts, all_normals

def _compute_s_and_overlap_fast(inner, outer_angle):
    centers = inner[:, :2]
    all_verts, all_normals = _compute_all_verts_normals(inner)
    
    a_rad = np.radians(outer_angle)
    outer_unit_verts = np.column_stack([
        np.cos(a_rad + np.radians(60) * np.arange(6)),
        np.sin(a_rad + np.radians(60) * np.arange(6))
    ])
    outer_normals = _edge_normals(outer_unit_verts)
    
    all_inner_verts_flat = all_verts.reshape(72, 2)
    
    max_proj = 0.0
    for normal in outer_normals:
        projs = np.dot(all_inner_verts_flat, normal)
        max_k = np.max(projs)
        if max_k > max_proj:
            max_proj = max_k
    
    s = max_proj * 2.0 / np.sqrt(3)
    s = max(s, np.sqrt(12))
    
    is_valid = True
    for i in range(12):
        for j in range(i+1, 12):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist > 2.0 * np.sqrt(3) + 0.1:
                continue
            overlap, _, _ = _sat_overlap(all_verts[i], all_normals[i], all_verts[j], all_normals[j])
            if overlap > 0:
                is_valid = False
                break
        if not is_valid:
            break
    
    return s, is_valid

def _resolve_collisions(inner, outer_angle, max_iters=50, push_factor=0.6):
    """
    Iteratively resolve collisions by pushing overlapping hexagons apart.
    Returns modified inner array.
    """
    inner = inner.copy()
    
    for _ in range(max_iters):
        all_verts, all_normals = _compute_all_verts_normals(inner)
        centers = inner[:, :2]
        
        any_overlap = False
        for i in range(12):
            for j in range(i+1, 12):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist > 2.0 * np.sqrt(3) + 0.1:
                    continue
                overlap, axis, sign = _sat_overlap(all_verts[i], all_normals[i], all_verts[j], all_normals[j])
                if overlap > 1e-8 and axis is not None:
                    any_overlap = True
                    # Push i in sign*axis direction, j in -sign*axis direction
                    push = sign * axis * overlap * push_factor
                    inner[i, 0] += push[0]
                    inner[i, 1] += push[1]
                    inner[j, 0] -= push[0]
                    inner[j, 1] -= push[1]
        
        if not any_overlap:
            break
    
    return inner

# Contact graph: 4-3-3-2 honeycomb slice
# Node indices 0-11. Edges list (i, j, base_angle_deg) where base_angle is
# the direction from node i to node j in the ideal lattice.
# Row 0 (4): nodes 0,1,2,3 at y=0
# Row 1 (3): nodes 4,5,6 at y=sqrt(3)/2*L, x offset by L/2
# Row 2 (3): nodes 7,8,9 at y=sqrt(3)*L, x offset by L
# Row 3 (2): nodes 10,11 at y=sqrt(3)*3/2*L, x offset by L
_CONTACT_EDGES = [
    # Row 0 horizontal
    (0, 1, 0.0),
    (1, 2, 0.0),
    (2, 3, 0.0),
    # Row 0 to Row 1 (diagonal down-right)
    (0, 4, -60.0),
    (1, 4, 60.0),
    (1, 5, -60.0),
    (2, 5, 60.0),
    (2, 6, -60.0),
    (3, 6, 60.0),
    # Row 1 horizontal
    (4, 5, 0.0),
    (5, 6, 0.0),
    # Row 1 to Row 2
    (4, 7, -60.0),
    (5, 7, 60.0),
    (5, 8, -60.0),
    (6, 8, 60.0),
    (6, 9, -60.0),
    # Row 2 horizontal
    (7, 8, 0.0),
    (8, 9, 0.0),
    # Row 2 to Row 3
    (7, 10, -60.0),
    (8, 10, 60.0),
    (8, 11, -60.0),
    (9, 11, 60.0),
    # Row 3 horizontal
    (10, 11, 0.0),
]
# 23 edges: 23 distances + 23 angles = 46 vars, but we use only a subset.
# For the contact-graph optimization we use 6 "row spacing" distances and 6 "column" distances.
# Simplified: use all 23 edge distances as variables (angles derived from geometry).

def _build_positions_from_distances(distances):
    """
    Build hexagon positions from edge distances using a fixed contact graph.
    distances: array of length 23 matching _CONTACT_EDGES order.
    Returns (12, 2) array of positions.
    """
    L = 2 * np.sqrt(3)
    positions = np.zeros((12, 2))
    # Node 0 at origin
    positions[0] = [0.0, 0.0]
    # Row 0: nodes 1, 2, 3
    positions[1] = [distances[0], 0.0]
    positions[2] = [positions[1, 0] + distances[1], 0.0]
    positions[3] = [positions[2, 0] + distances[2], 0.0]
    # Row 1: nodes 4, 5, 6
    # Node 4: below-right of node 0 and below-left of node 1
    # Use node 0 -> node 4: distance d3 at angle -60
    d3 = distances[3]
    positions[4] = [d3 * np.cos(np.radians(-60)), d3 * np.sin(np.radians(-60))]
    # Node 5: use node 1 -> node 5: distance d5 at angle -60
    d5 = distances[5]
    positions[5] = [positions[1, 0] + d5 * np.cos(np.radians(-60)), positions[1, 1] + d5 * np.sin(np.radians(-60))]
    # Node 6: use node 2 -> node 6: distance d7 at angle -60
    d7 = distances[7]
    positions[6] = [positions[2, 0] + d7 * np.cos(np.radians(-60)), positions[2, 1] + d7 * np.sin(np.radians(-60))]
    # Row 2: nodes 7, 8, 9
    d9 = distances[9]
    positions[7] = [positions[4, 0] + d9 * np.cos(np.radians(-60)), positions[4, 1] + d9 * np.sin(np.radians(-60))]
    d11 = distances[11]
    positions[8] = [positions[5, 0] + d11 * np.cos(np.radians(-60)), positions[5, 1] + d11 * np.sin(np.radians(-60))]
    d13 = distances[13]
    positions[9] = [positions[6, 0] + d13 * np.cos(np.radians(-60)), positions[6, 1] + d13 * np.sin(np.radians(-60))]
    # Row 3: nodes 10, 11
    d15 = distances[15]
    positions[10] = [positions[7, 0] + d15 * np.cos(np.radians(-60)), positions[7, 1] + d15 * np.sin(np.radians(-60))]
    d17 = distances[17]
    positions[11] = [positions[8, 0] + d17 * np.cos(np.radians(-60)), positions[8, 1] + d17 * np.sin(np.radians(-60))]
    return positions

def _generate_seeds():
    seeds = []
    L = 2 * np.sqrt(3) + 0.005  # Slightly tighter
    
    # Seed 1: Hexagonal cluster (center + 6 + 5)
    points = [[0.0, 0.0, 0.0]]
    for k in range(6):
        angle = np.radians(60 * k)
        points.append([L * np.cos(angle), L * np.sin(angle), 0.0])
    second_ring = [(2, 0), (1, 1), (0, 2), (-1, 1), (-2, 0)]
    for (q, r) in second_ring:
        x = L * (q + r/2.0)
        y = L * r * np.sqrt(3)/2
        points.append([x, y, 0.0])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts)
    
    # Seed 2: Triangular lattice 3x4
    points = []
    for row in range(3):
        for col in range(4):
            x = (col + row/2.0) * L
            y = row * L * np.sqrt(3)/2
            points.append([x, y, 0.0])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts)
    
    # Seed 3: Two rows of 6
    points = []
    for row in range(2):
        for col in range(6):
            x = col * L
            y = row * L * np.sqrt(3)
            points.append([x, y, 0.0])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts)
    
    # Seed 4: Circular arrangement with center
    points = [[0.0, 0.0, 0.0]]
    for k in range(11):
        angle = np.radians(360.0 * k / 11.0)
        r = L * 1.5
        points.append([r * np.cos(angle), r * np.sin(angle), 0.0])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts)
    
    # Seed 5: 4-4-4 grid
    points = []
    for row in range(3):
        for col in range(4):
            x = (col + row/2.0) * L
            y = row * L * np.sqrt(3)/2
            points.append([x, y, 0.0])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts)
    
    # Seed 6: Compact hexagonal arrangement (center + 5 around, then 6 more)
    points = [[0.0, 0.0, 0.0]]
    for k in range(5):
        angle = np.radians(72 * k + 18)
        points.append([L * np.cos(angle), L * np.sin(angle), 0.0])
    for k in range(6):
        angle = np.radians(60 * k)
        r = L * np.sqrt(3)
        points.append([r * np.cos(angle), r * np.sin(angle), 0.0])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts)
    
    # Seed 7: Ring of 6 + inner cluster
    points = []
    for k in range(6):
        angle = np.radians(60 * k)
        points.append([L * 1.5 * np.cos(angle), L * 1.5 * np.sin(angle), 0.0])
    for k in range(6):
        angle = np.radians(60 * k + 30)
        points.append([L * 0.5 * np.cos(angle), L * 0.5 * np.sin(angle), 0.0])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts)
    
    # Seed 8: 2-3-3-2-2 arrangement (more compact)
    points = []
    # Row 0: 2 hexagons
    for col in range(2):
        x = col * L
        y = 0.0
        points.append([x, y, 0.0])
    # Row 1: 3 hexagons
    for col in range(3):
        x = (col + 0.5) * L
        y = L * np.sqrt(3)/2
        points.append([x, y, 0.0])
    # Row 2: 3 hexagons
    for col in range(3):
        x = (col + 0.5) * L
        y = L * np.sqrt(3)
        points.append([x, y, 0.0])
    # Row 3: 2 hexagons
    for col in range(2):
        x = (col + 1.0) * L
        y = L * np.sqrt(3) * 1.5
        points.append([x, y, 0.0])
    # Row 4: 2 hexagons
    for col in range(2):
        x = (col + 1.0) * L
        y = L * np.sqrt(3) * 2.0
        points.append([x, y, 0.0])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts)
    
    # Seed 9: Rotated triangular lattice 3x4
    angle_rot = np.radians(15)
    points = []
    for row in range(3):
        for col in range(4):
            x = (col + row/2.0) * L
            y = row * L * np.sqrt(3)/2
            # Rotate
            xr = x * np.cos(angle_rot) - y * np.sin(angle_rot)
            yr = x * np.sin(angle_rot) + y * np.cos(angle_rot)
            points.append([xr, yr, 15.0])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts)
    
    # Seed 10: Diamond arrangement
    points = []
    # Center
    points.append([0.0, 0.0, 0.0])
    # 4 neighbors
    for k in range(4):
        angle = np.radians(90 * k + 45)
        points.append([L * np.cos(angle), L * np.sin(angle), 0.0])
    # 7 more in outer ring
    for k in range(7):
        angle = np.radians(360.0 * k / 7.0 + 20)
        r = L * 1.8
        points.append([r * np.cos(angle), r * np.sin(angle), 0.0])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts)
    
    # Seed 11: 4-3-3-2 honeycomb via contact graph
    L_ideal = 2 * np.sqrt(3)
    dists = np.full(len(_CONTACT_EDGES), L_ideal)
    pos = _build_positions_from_distances(dists)
    pts = np.column_stack([pos, np.zeros(12)])
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts)
    
    # Seed 12: Slightly compressed 4-3-3-2
    L_comp = 2 * np.sqrt(3) * 0.98
    dists = np.full(len(_CONTACT_EDGES), L_comp)
    pos = _build_positions_from_distances(dists)
    pts = np.column_stack([pos, np.zeros(12)])
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts)
    
    return seeds

def _contact_graph_objective(dists_and_angle):
    """
    Objective for contact-graph optimization.
    dists_and_angle: array of length 24 (23 edge distances + 1 outer angle).
    Returns s (minimizing s maximizes 1/s).
    """
    L_min = 2 * np.sqrt(3) * 0.9
    L_max = 2 * np.sqrt(3) * 1.1
    dists = np.clip(dists_and_angle[:23], L_min, L_max)
    outer_angle = dists_and_angle[23]
    
    pos = _build_positions_from_distances(dists)
    # Center
    center = np.mean(pos, axis=0)
    pos -= center
    inner = np.column_stack([pos, np.zeros(12)])
    
    s, is_valid = _compute_s_and_overlap_fast(inner, outer_angle)
    if is_valid:
        return s
    else:
        # Penalty for overlap
        all_verts, all_normals = _compute_all_verts_normals(inner)
        penalty = 0.0
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(inner[i, :2] - inner[j, :2])
                if d > 2.0 * np.sqrt(3) + 0.1:
                    continue
                overlap, _, _ = _sat_overlap(all_verts[i], all_normals[i], all_verts[j], all_normals[j])
                penalty += overlap
        return s + 5.0 * penalty + 2.0

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 615.0
    
    seeds = _generate_seeds()
    
    best_inner = None
    best_s = float('inf')
    best_outer_angle = 0.0
    
    for seed in seeds:
        for angle_deg in [0, 30, 15, 45, 60, 10, 20, 50, 25, 55]:
            s, is_valid = _compute_s_and_overlap_fast(seed, angle_deg)
            if is_valid and s < best_s:
                best_s = s
                best_inner = seed.copy()
                best_outer_angle = angle_deg
    
    if best_inner is None:
        best_inner = seeds[0].copy()
        best_s, _ = _compute_s_and_overlap_fast(best_inner, 0)
        best_outer_angle = 0
    
    # Phase 1: Contact-graph optimization (24 variables)
    from scipy.optimize import minimize
    
    L_ideal = 2 * np.sqrt(3)
    x0_contact = np.concatenate([np.full(23, L_ideal), [best_outer_angle]])
    
    for restart in range(6):
        if time.time() > deadline - 200.0:
            break
        if restart == 0:
            x_start = x0_contact.copy()
        else:
            x_start = x0_contact + np.random.randn(24) * 0.02
        
        try:
            result = minimize(_contact_graph_objective, x_start, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            dists_opt = np.clip(result.x[:23], 2*np.sqrt(3)*0.9, 2*np.sqrt(3)*1.1)
            outer_angle_opt = result.x[23]
            pos = _build_positions_from_distances(dists_opt)
            center = np.mean(pos, axis=0)
            pos -= center
            candidate_inner = np.column_stack([pos, np.zeros(12)])
            candidate_s, candidate_valid = _compute_s_and_overlap_fast(candidate_inner, outer_angle_opt)
            if candidate_valid and candidate_s < best_s:
                best_inner = candidate_inner
                best_s = candidate_s
                best_outer_angle = outer_angle_opt
                x0_contact = np.concatenate([dists_opt, [outer_angle_opt]])
        except Exception:
            pass
    
    # Phase 2: Free-position Nelder-Mead polish (37 variables)
    def objective(params):
        if time.time() > deadline - 50.0:
            return 1e10
        inner = params[:36].reshape(12, 3).copy()
        outer_angle = params[36]
        s, is_valid = _compute_s_and_overlap_fast(inner, outer_angle)
        if is_valid:
            return s
        else:
            all_verts, all_normals = _compute_all_verts_normals(inner)
            penalty = 0.0
            for i in range(12):
                for j in range(i+1, 12):
                    dist = np.linalg.norm(inner[i, :2] - inner[j, :2])
                    if dist > 2.0 * np.sqrt(3) + 0.1:
                        continue
                    overlap, _, _ = _sat_overlap(all_verts[i], all_normals[i], all_verts[j], all_normals[j])
                    penalty += overlap
            return s + 5.0 * penalty + 2.0
    
    x0 = np.concatenate([best_inner.flatten(), [best_outer_angle]])
    
    for restart in range(8):
        if time.time() > deadline - 60.0:
            break
        if restart == 0:
            x_start = x0.copy()
        else:
            x_start = x0 + np.random.randn(37) * 0.006
        
        try:
            result = minimize(objective, x_start, method='Nelder-Mead',
                             options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-12})
            candidate_inner = result.x[:36].reshape(12, 3).copy()
            candidate_angle = result.x[36]
            candidate_s, candidate_valid = _compute_s_and_overlap_fast(candidate_inner, candidate_angle)
            if candidate_valid and candidate_s < best_s:
                best_inner = candidate_inner
                best_s = candidate_s
                best_outer_angle = candidate_angle
                x0 = np.concatenate([best_inner.flatten(), [best_outer_angle]])
        except Exception:
            pass
    
    # Phase 3: Fine-tuning with collision resolution
    if time.time() < deadline - 80.0:
        np.random.seed(123)
        current_inner = best_inner.copy()
        current_s = best_s
        current_outer_angle = best_outer_angle
        
        for it in range(12000):
            if time.time() > deadline - 80.0:
                break
            new_inner = current_inner.copy()
            new_outer_angle = current_outer_angle
            idx = np.random.randint(0, 12)
            new_inner[idx, 0] += np.random.randn() * 0.003
            new_inner[idx, 1] += np.random.randn() * 0.003
            new_inner[idx, 2] += np.random.randn() * 0.0015
            if np.random.rand() < 0.3:
                new_outer_angle += np.random.randn() * 0.002
            
            new_s, new_valid = _compute_s_and_overlap_fast(new_inner, new_outer_angle)
            if not new_valid:
                new_inner = _resolve_collisions(new_inner, new_outer_angle, max_iters=25, push_factor=0.4)
                new_s, new_valid = _compute_s_and_overlap_fast(new_inner, new_outer_angle)
            
            if new_valid:
                if new_s < current_s:
                    current_inner = new_inner
                    current_s = new_s
                    current_outer_angle = new_outer_angle
                    if current_s < best_s:
                        best_inner = current_inner.copy()
                        best_s = current_s
                        best_outer_angle = current_outer_angle
                else:
                    delta = new_s - current_s
                    if np.random.rand() < np.exp(-delta / max(0.002, 1e-10)):
                        current_inner = new_inner
                        current_s = new_s
                        current_outer_angle = new_outer_angle
    
    # Phase 4: Ultra-fine polish
    if time.time() < deadline - 30.0:
        np.random.seed(456)
        current_inner = best_inner.copy()
        current_s = best_s
        current_outer_angle = best_outer_angle
        
        for it in range(5000):
            if time.time() > deadline - 30.0:
                break
            new_inner = current_inner.copy()
            new_outer_angle = current_outer_angle
            idx = np.random.randint(0, 12)
            new_inner[idx, 0] += np.random.randn() * 0.001
            new_inner[idx, 1] += np.random.randn() * 0.001
            new_inner[idx, 2] += np.random.randn() * 0.0005
            if np.random.rand() < 0.4:
                new_outer_angle += np.random.randn() * 0.0008
            
            new_s, new_valid = _compute_s_and_overlap_fast(new_inner, new_outer_angle)
            if not new_valid:
                new_inner = _resolve_collisions(new_inner, new_outer_angle, max_iters=15, push_factor=0.3)
                new_s, new_valid = _compute_s_and_overlap_fast(new_inner, new_outer_angle)
            
            if new_valid:
                if new_s < current_s:
                    current_inner = new_inner
                    current_s = new_s
                    current_outer_angle = new_outer_angle
                    if current_s < best_s:
                        best_inner = current_inner.copy()
                        best_s = current_s
                        best_outer_angle = current_outer_angle
    
    # Final validity check
    s_final, valid_final = _compute_s_and_overlap_fast(best_inner, best_outer_angle)
    if not valid_final:
        best_inner = seeds[0].copy()
        best_s, _ = _compute_s_and_overlap_fast(best_inner, 0)
        best_outer_angle = 0
    
    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    return best_inner, outer_hex_data, float(best_s)

# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-END
