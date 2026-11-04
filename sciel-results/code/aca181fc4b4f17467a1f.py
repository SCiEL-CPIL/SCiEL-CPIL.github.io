import numpy as np
import time
from scipy.optimize import minimize
import math

N = 12
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin for 150s limit

def _hex_vertices(center, side, angle_deg):
    """Return vertices of a regular hexagon."""
    x, y = center
    theta = np.deg2rad(angle_deg)
    angles = theta + np.arange(6) * (np.pi / 3)
    return np.array([
        [x + side * np.cos(a), y + side * np.sin(a)] 
        for a in angles
    ])

def _get_hex_vertices_vec(centers, angles, side=1.0):
    """
    Vectorized computation of vertices for N hexagons.
    centers: (N, 2)
    angles: (N,) in radians
    Returns: (N, 6, 2)
    """
    base_angles = np.arange(6) * (np.pi / 3)
    base_verts = np.stack([np.cos(base_angles), np.sin(base_angles)], axis=1) # (6, 2)
    
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    
    R = np.zeros((len(angles), 2, 2))
    R[:, 0, 0] = cos_a
    R[:, 0, 1] = -sin_a
    R[:, 1, 0] = sin_a
    R[:, 1, 1] = cos_a
    
    # Apply rotation: v_rot[i] = R[i] @ v_base
    rotated = np.einsum('nij,jk->nki', R, base_verts) # (N, 2, 6)
    rotated = rotated.transpose(0, 2, 1) # (N, 6, 2)
    
    scaled = rotated * side
    translated = scaled + centers[:, np.newaxis, :]
    
    return translated

def _minkowski_collision_penalty(centers, angles):
    """
    Calculate a smooth penalty for overlapping hexagons using Minkowski sum properties.
    
    For two convex polygons A and B, they overlap if the origin is in A - B (Minkowski difference).
    The distance from the origin to the boundary of A-B is a valid measure of separation.
    If distance > 0, they are separated. If distance < 0, they overlap.
    
    For regular hexagons, the Minkowski difference of two unit hexagons (with relative rotation)
    is a convex polygon. We can compute the support function of this polygon.
    
    A simpler, robust differentiable approximation for N=12:
    For each pair (i, j), we compute the minimum distance between any vertex of i and any edge of j,
    and vice versa, plus vertex-vertex distances. 
    However, a fully vectorized, smooth "signed distance" is complex.
    
    Alternative robust approach for differentiability:
    Use the fact that for two convex polygons, they are disjoint iff there exists a separating axis.
    We can sum the squared "violation" of separation along a set of candidate axes.
    Candidate axes: normals of edges of hex i, normals of edges of hex j.
    
    Let's implement the Separating Axis Theorem (SAT) penalty.
    For pair (i, j):
    Axes = normals of i (6) + normals of j (6).
    For each axis n, project both polygons onto n.
    Let [min_i, max_i] and [min_j, max_j] be the projections.
    If max_i < min_j or max_j < min_i, they are separated on this axis.
    If they overlap on ALL axes, they intersect.
    
    Penalty for a single axis where they overlap:
    overlap = min(max_i, max_j) - max(min_i, min_j)
    If overlap > 0, they overlap on this axis.
    To make it a global penalty that is 0 only if separated, we can take the MAX overlap over all axes?
    No, SAT says they are separated if ANY axis has separation.
    So they overlap if MIN separation over all axes is < 0?
    Actually, let sep = min_axis (separation on that axis).
    If sep > 0, separated. If sep <= 0, overlapping (or touching).
    We want a penalty for sep <= 0.
    Penalty = max(0, -sep)^2 ?
    
    Wait, sep on an axis is (min_j - max_i) if i is left of j, or (min_i - max_j) if j is left of i.
    Let's define separation on axis n as:
    dist_i = dot(centers_i, n)
    dist_j = dot(centers_j, n)
    radius_i = max(dot(verts_i, n)) - min(dot(verts_i, n)) ... no, radius is support function.
    h_i(n) = max(dot(v, n) for v in verts_i)
    h_i(-n) = min(dot(v, n) for v in verts_i)
    
    Separation along n:
    gap = min( h_i(n) - h_j(-n), h_j(n) - h_i(-n) ) ... this is getting complicated with signs.
    
    Let's stick to the vertex-vertex and vertex-edge distance approach but smoothed.
    Actually, for N=12, a simple but effective penalty is the sum of squared distances of vertices of one hexagon that fall INSIDE the other.
    Point-in-polygon is not differentiable.
    
    Let's go back to the Minkowski Sum idea but simplified.
    The "collision" condition for two convex sets A, B is 0 in A-B.
    The distance from 0 to A-B is the minimum distance between A and B.
    We can approximate the distance between two convex polygons by the minimum distance between all pairs of (vertex, edge) and (vertex, vertex).
    
    Let's use a "soft" minimum distance penalty.
    For pair (i, j):
    Compute all pairwise distances between vertices of i and vertices of j.
    Compute all distances from vertices of i to edges of j.
    Compute all distances from vertices of j to edges of i.
    Let D be the set of all these distances.
    min_dist = min(D).
    Penalty = max(0, -min_dist)^2 ? No, distance is always >= 0.
    If they overlap, the "distance" is 0 (if we define distance between sets as 0 when intersecting).
    But calculating "distance between sets" is 0 if intersecting, which is not smooth.
    
    Let's use the **Lagrange Multiplier / Energy** approach from physics:
    If we treat the hexagons as rigid bodies, we can use a penalty potential that is zero when separated and grows quadratically when overlapping.
    A common smooth approximation for the distance between two convex polygons is:
    d = min_{v in V_i, e in E_j} dist(v, e) ... but if v is inside, dist is 0.
    
    Let's try a different, very effective heuristic for hexagons:
    Since hexagons are centrally symmetric, we can use the center distance and relative angle.
    However, the most robust differentiable penalty for polygon packing is often the **sum of squared violations of the Separating Axis Theorem**.
    
    Let's implement SAT Penalty:
    For pair (i, j):
    Axes = normals of i + normals of j.
    For each axis n:
      proj_i_min, proj_i_max = min/max dot(verts_i, n)
      proj_j_min, proj_j_max = min/max dot(verts_j, n)
      
      # Check if intervals [proj_i_min, proj_i_max] and [proj_j_min, proj_j_max] overlap
      # Overlap length = min(proj_i_max, proj_j_max) - max(proj_i_min, proj_j_min)
      # If overlap > 0, they overlap on this axis.
      # If overlap <= 0, they are separated on this axis.
      
      # We want a penalty if they are NOT separated on ANY axis.
      # This is a logical AND of "overlap on axis k".
      # Penalty = product of (overlap_k + epsilon)? No.
      
      # Let's use the MAXIMUM separation.
      # sep_k = min(proj_i_max, proj_j_max) - max(proj_i_min, proj_j_min)
      # If sep_k < 0, they are separated on axis k.
      # If all sep_k >= 0, they overlap.
      # We want to penalize when min_k(sep_k) >= 0? No, that's wrong.
      # They overlap if min_k(sep_k) > 0? No.
      # They are separated if EXISTS k such that sep_k < 0.
      # They overlap if FOR ALL k, sep_k >= 0.
      
      # Let S = min_k(sep_k).
      # If S < 0, they are separated (at least on one axis).
      # If S >= 0, they overlap (on all axes).
      # So we can penalize max(0, S)^2.
      
      # Let's verify:
      # If separated, there is some axis where intervals don't touch.
      # On that axis, say i is left of j.
      # proj_i_max < proj_j_min.
      # sep = proj_i_max - proj_j_min < 0.
      # So min_k(sep_k) will be negative.
      # max(0, S) = 0. Penalty 0. Correct.
      
      # If overlapping, on all axes, intervals overlap.
      # sep_k >= 0 for all k.
      # S = min_k(sep_k) >= 0.
      # max(0, S) = S. Penalty S^2. Correct.
      
    This is a smooth, differentiable penalty!
    """
    verts = _get_hex_vertices_vec(centers, angles) # (N, 6, 2)
    
    # Get normals for each hexagon
    # Normal to edge k is perpendicular to (v_{k+1} - v_k)
    # Or simply, normals are at angles angle_deg + k*pi/3 + pi/6?
    # For a hexagon with vertices at angle theta + k*pi/3, the edges are between k and k+1.
    # The normal to the edge is at theta + (k+0.5)*pi/3.
    # Let's just compute normals from vertices to be safe.
    
    # v: (N, 6, 2)
    # edges: v[:, 1:] - v[:, :-1] ... wrap around
    # e_k = v_{k+1} - v_k
    # normal n_k = (e_y, -e_x) or (-e_y, e_x). We need outward normal.
    # For a convex polygon in CCW order, (dy, -dx) is inward?
    # Let's check: (1,0) to (0,1). Edge (-1, 1). Normal (1, 1) is outward?
    # Center (0,0). Midpoint (-0.5, 0.5). Dot((-0.5, 0.5), (1, 1)) = 0.
    # Let's just use the analytical normals for regular hexagons.
    # Vertex angles: theta + k*pi/3.
    # Edge k is between vertex k and k+1.
    # Normal angle: theta + (k + 0.5)*pi/3.
    
    base_normal_angles = np.arange(6) * (np.pi / 3) + np.pi / 6
    base_normals = np.stack([np.cos(base_normal_angles), np.sin(base_normal_angles)], axis=1) # (6, 2)
    
    # Rotate normals for each hexagon
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    R = np.zeros((len(angles), 2, 2))
    R[:, 0, 0] = cos_a
    R[:, 0, 1] = -sin_a
    R[:, 1, 0] = sin_a
    R[:, 1, 1] = cos_a
    
    normals = np.einsum('nij,jk->nki', R, base_normals).transpose(0, 2, 1) # (N, 6, 2)
    
    # Project vertices onto normals
    # verts: (N, 6, 2)
    # normals: (N, 6, 2)
    # We need projection of verts_i onto normal_j for all i, j, k, l?
    # No, for SAT, we check axes from i and axes from j.
    # Axis set for pair (i, j) = Normals_i (6) + Normals_j (6).
    
    # To vectorize:
    # We need proj(verts_i, n) for all n in Normals_i and Normals_j.
    
    # Let's compute all pairwise projections.
    # P[i, j, k, l] = dot(verts[i, k], normals[j, l])
    # This is (N, N, 6, 6).
    
    # verts: (N, 6, 2) -> (N, 1, 6, 1, 2)
    # normals: (N, 6, 2) -> (1, N, 1, 6, 2)
    
    v_exp = verts[:, np.newaxis, :, np.newaxis, :] # (N, 1, 6, 1, 2)
    n_exp = normals[np.newaxis, :, np.newaxis, :, :] # (1, N, 1, 6, 2)
    
    # proj[i, j, k, l] = dot(verts[i, k], normals[j, l])
    projs = np.einsum('nikl,mjpl->nmip', v_exp, n_exp) # (N, N, 6, 6)
    # Wait, indices:
    # v_exp: i, dummy, k, dummy, dim
    # n_exp: dummy, j, dummy, l, dim
    # Result: i, j, k, l
    # projs[i, j, k, l] = dot(verts[i, k], normals[j, l])
    
    # We need min and max projection of hex i onto normal j_l.
    # For fixed i, j, l:
    # min_proj = min_k(projs[i, j, k, l])
    # max_proj = max_k(projs[i, j, k, l])
    
    min_projs = np.min(projs, axis=2) # (N, N, 6) -> min over k (vertex of i)
    max_projs = np.max(projs, axis=2) # (N, N, 6) -> max over k (vertex of i)
    
    # Now, for each pair (i, j), we have 6 axes from i (where j is the "normal source" index? No.)
    # Let's re-map.
    # We want axes from i. These are normals[i, :].
    # So we look at projs[:, i, :, l]? No.
    # projs[i, j, k, l] is dot(vert_i_k, normal_j_l).
    # If we want axis = normal_i_l, we need dot(vert_i_k, normal_i_l) and dot(vert_j_k, normal_i_l).
    # So for axis normal_i_l:
    #   proj_i = projs[i, i, :, l] -> min/max over k
    #   proj_j = projs[j, i, :, l] -> min/max over k
    
    # Let's extract these.
    # min_proj_i_on_i_normals: (N, 6)
    # min_proj_j_on_i_normals: (N, N, 6) -> for each i, j, l
    
    # min_projs[i, i, l] is min proj of i on normal_i_l.
    # min_projs[j, i, l] is min proj of j on normal_i_l.
    
    # We need to gather these for all i, j.
    # Let's create arrays for "axes from i" and "axes from j".
    
    # Axes from i:
    # For each i, j, l (l=0..5):
    #   min_i = min_projs[i, i, l]
    #   max_i = max_projs[i, i, l]
    #   min_j = min_projs[j, i, l]
    #   max_j = max_projs[j, i, l]
    
    # We can do this with advanced indexing.
    idx_i = np.arange(N)[:, np.newaxis] # (N, 1)
    idx_j = np.arange(N)[np.newaxis, :] # (1, N)
    idx_l = np.arange(6)[np.newaxis, np.newaxis, :] # (1, 1, 6)
    
    # min_i: (N, N, 6)
    min_i = min_projs[idx_i, idx_i, idx_l]
    max_i = max_projs[idx_i, idx_i, idx_l]
    min_j = min_projs[idx_j, idx_i, idx_l]
    max_j = max_projs[idx_j, idx_i, idx_l]
    
    # Separation on axis l for pair (i, j):
    # sep = min(max_i, max_j) - max(min_i, min_j)
    sep_i_axes = np.minimum(max_i, max_j) - np.maximum(min_i, min_j) # (N, N, 6)
    
    # Similarly for axes from j:
    min_i2 = min_projs[idx_i, idx_j, idx_l]
    max_i2 = max_projs[idx_i, idx_j, idx_l]
    min_j2 = min_projs[idx_j, idx_j, idx_l]
    max_j2 = max_projs[idx_j, idx_j, idx_l]
    
    sep_j_axes = np.minimum(max_i2, max_j2) - np.maximum(min_i2, min_j2) # (N, N, 6)
    
    # Total separation for pair (i, j) is the max separation over all 12 axes?
    # No, they are separated if ANY axis has sep < 0.
    # So we want to know if min(sep_all) < 0.
    # Let S_ij = min( sep_i_axes[i,j,:], sep_j_axes[i,j,:] )
    # If S_ij < 0, separated.
    # If S_ij >= 0, overlapping.
    
    # Combine all 12 axes
    all_seps = np.concatenate([sep_i_axes, sep_j_axes], axis=2) # (N, N, 12)
    
    S = np.min(all_seps, axis=2) # (N, N)
    
    # Penalty for pair (i, j): max(0, S_ij)^2
    # We only want i < j
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    penalty_terms = np.maximum(0, S)
    penalty = np.sum(penalty_terms[mask]**2)
    
    return penalty

def _objective(params):
    centers = params[:N*2].reshape(N, 2)
    angles = params[N*2:N*2+N]
    outer_side = params[N*2+N]
    
    # Containment
    verts = _get_hex_vertices_vec(centers, angles)
    normal_angles = np.arange(6) * (np.pi / 3)
    normals = np.stack([np.cos(normal_angles), np.sin(normal_angles)], axis=1)
    projs = np.einsum('nij,jk->nik', verts, normals.T)
    max_projs = np.max(projs, axis=1)
    global_max_proj = np.max(max_projs)
    required_side = global_max_proj * 2.0 / np.sqrt(3.0)
    
    containment_penalty = np.maximum(0, required_side - outer_side)**2
    
    # Collision
    collision_pen = _minkowski_collision_penalty(centers, angles)
    
    obj = outer_side + 1000.0 * (containment_penalty + collision_pen)
    
    return obj

def construct():
    start_time = time.time()
    
    best_score = float('inf')
    best_params = None
    
    candidates = []
    
    # Start 1: Triangular lattice
    pts = []
    for i in range(6):
        for j in range(2):
            x = i * 1.0 + 0.5 * j
            y = j * np.sqrt(3) / 2
            pts.append([x, y])
    centroid = np.mean(pts, axis=0)
    pts = np.array(pts) - centroid
    centers = pts
    angles = np.zeros(N)
    side = 5.0
    params = np.concatenate([centers.flatten(), angles, [side]])
    candidates.append(params)
    
    # Start 2: Hexagonal cluster
    pts = [[0,0]]
    for k in range(6):
        a = k * np.pi / 3
        pts.append([np.sqrt(3)*np.cos(a), np.sqrt(3)*np.sin(a)])
    for k in range(6):
        a = (k+0.5) * np.pi / 3
        pts.append([2.0*np.cos(a), 2.0*np.sin(a)])
    # We have 13 points, take 12
    pts = pts[:12]
    centroid = np.mean(pts, axis=0)
    pts = np.array(pts) - centroid
    centers = pts
    angles = np.zeros(N)
    side = 5.0
    params = np.concatenate([centers.flatten(), angles, [side]])
    candidates.append(params)
    
    # Random starts
    np.random.seed(42)
    for _ in range(30):
        centers = np.random.uniform(-3, 3, (N, 2))
        angles = np.random.uniform(0, 2*np.pi, N)
        side = 6.0
        params = np.concatenate([centers.flatten(), angles, [side]])
        candidates.append(params)
    
    for i, init_params in enumerate(candidates):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        
        try:
            res = minimize(
                _objective,
                init_params,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if res.fun < best_score:
                # Verify validity
                params = res.x
                centers = params[:N*2].reshape(N, 2)
                angles = params[N*2:N*2+N]
                side = params[N*2+N]
                
                # Check containment
                verts = _get_hex_vertices_vec(centers, angles)
                normal_angles = np.arange(6) * (np.pi / 3)
                normals = np.stack([np.cos(normal_angles), np.sin(normal_angles)], axis=1)
                projs = np.einsum('nij,jk->nik', verts, normals.T)
                max_projs = np.max(projs, axis=1)
                global_max_proj = np.max(max_projs)
                required_side = global_max_proj * 2.0 / np.sqrt(3.0)
                
                if side < required_side - 1e-5:
                    continue
                
                # Check collision (should be 0 if valid)
                # We trust the optimizer if penalty is small
                if res.fun < best_score:
                    best_score = res.fun
                    best_params = params
                    
        except Exception as e:
            pass
            
    if best_params is not None:
        centers = best_params[:N*2].reshape(N, 2)
        angles = best_params[N*2:N*2+N]
        side = best_params[N*2+N]
        
        angles_deg = np.rad2deg(angles)
        
        inner = []
        for i in range(N):
            inner.append([float(centers[i][0]), float(centers[i][1]), float(angles_deg[i])])
            
        return (inner, (0.0, 0.0), float(side), 0.0)
    
    # Fallback
    return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

# Run construct to ensure it works
# construct()
