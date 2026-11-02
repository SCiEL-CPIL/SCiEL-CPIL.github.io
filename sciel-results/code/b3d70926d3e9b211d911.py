import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # Safety margin before 120s hard limit

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    """Get vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _hex_edge_normals(angle_deg=0.0):
    """Get the 6 outward normal directions for a unit hexagon with given rotation."""
    angle_rad = math.radians(angle_deg)
    normals = []
    for k in range(6):
        a = angle_rad + math.pi / 6 + math.pi * k / 3
        normals.append((math.cos(a), math.sin(a)))
    return normals

def _pairwise_min_gap(pos_i, ang_i, pos_j, ang_j):
    """
    Compute the minimum gap (distance) between two unit hexagons.
    Positive gap means separated, negative means overlapping.
    Uses support function / separating axis approach.
    """
    # For two convex polygons, the minimum distance can be found by checking
    # all edges of both polygons. We use a simpler approach: check if any
    # vertex of one is inside the other, and if not, find minimum edge-to-edge distance.
    
    # First check if they overlap (vertex in polygon or edge intersection)
    if _hexagons_overlap_fast(pos_i, ang_i, pos_j, ang_j):
        # They overlap; estimate how much by checking vertex penetration
        # Find minimum penetration depth
        verts_i = _get_hex_vertices(pos_i[0], pos_i[1], ang_i)
        verts_j = _get_hex_vertices(pos_j[0], pos_j[1], ang_j)
        
        min_pen = float('inf')
        # Check how deep each vertex of i penetrates into j
        for v in verts_i:
            d = _point_to_convex_polygon_distance(v, pos_j, ang_j)
            if d < 0:  # Inside
                pen = -d
                if pen < min_pen:
                    min_pen = pen
        for v in verts_j:
            d = _point_to_convex_polygon_distance(v, pos_i, ang_i)
            if d < 0:  # Inside
                pen = -d
                if pen < min_pen:
                    min_pen = pen
        
        # Return negative gap (overlap)
        return -min_pen
    
    # No overlap: find minimum distance between the two hexagons
    verts_i = _get_hex_vertices(pos_i[0], pos_i[1], ang_i)
    verts_j = _get_hex_vertices(pos_j[0], pos_j[1], ang_j)
    
    min_dist = float('inf')
    for vi in verts_i:
        for vj in verts_j:
            d = math.sqrt((vi[0]-vj[0])**2 + (vi[1]-vj[1])**2)
            if d < min_dist:
                min_dist = d
    # Also check edge-to-edge distances
    for k in range(6):
        a1 = verts_i[k]
        b1 = verts_i[(k+1)%6]
        for l in range(6):
            a2 = verts_j[l]
            b2 = verts_j[(l+1)%6]
            d = _segment_segment_distance(a1, b1, a2, b2)
            if d < min_dist:
                min_dist = d
    
    return min_dist

def _point_to_convex_polygon_distance(p, center, angle_deg):
    """
    Signed distance from point p to convex hexagon centered at 'center' with rotation.
    Positive = outside, negative = inside (penetration depth).
    """
    px, py = p
    cx, cy = center
    verts = _get_hex_vertices(cx, cy, angle_deg)
    
    # Check if point is inside
    if _point_in_convex_polygon_fast(p, verts):
        # Find minimum distance to any edge (this is the penetration depth)
        min_edge_dist = float('inf')
        for k in range(6):
            a = verts[k]
            b = verts[(k+1)%6]
            d = _point_to_segment_distance(p, a, b)
            if d < min_edge_dist:
                min_edge_dist = d
        return -min_edge_dist
    
    # Outside: find minimum distance to any edge
    min_dist = float('inf')
    for k in range(6):
        a = verts[k]
        b = verts[(k+1)%6]
        d = _point_to_segment_distance(p, a, b)
        if d < min_dist:
            min_dist = d
    return min_dist

def _point_to_segment_distance(p, a, b):
    """Distance from point p to segment ab."""
    px, py = p
    ax, ay = a
    bx, by = b
    
    dx = bx - ax
    dy = by - ay
    len_sq = dx*dx + dy*dy
    
    if len_sq < 1e-12:
        return math.sqrt((px-ax)**2 + (py-ay)**2)
    
    t = ((px-ax)*dx + (py-ay)*dy) / len_sq
    t = max(0, min(1, t))
    
    proj_x = ax + t*dx
    proj_y = ay + t*dy
    
    return math.sqrt((px-proj_x)**2 + (py-proj_y)**2)

def _segment_segment_distance(p1, p2, p3, p4):
    """Minimum distance between two segments."""
    # Check if segments intersect
    if _segments_intersect(p1, p2, p3, p4):
        return 0.0
    
    # Check endpoint-to-segment distances
    d1 = _point_to_segment_distance(p1, p3, p4)
    d2 = _point_to_segment_distance(p2, p3, p4)
    d3 = _point_to_segment_distance(p3, p1, p2)
    d4 = _point_to_segment_distance(p4, p1, p2)
    
    return min(d1, d2, d3, d4)

def _hexagons_overlap_fast(pos_i, ang_i, pos_j, ang_j):
    """Fast check if two hexagons overlap."""
    verts_i = _get_hex_vertices(pos_i[0], pos_i[1], ang_i)
    verts_j = _get_hex_vertices(pos_j[0], pos_j[1], ang_j)
    
    # Check if any vertex of i is inside j
    for v in verts_i:
        if _point_in_convex_polygon_fast(v, verts_j):
            return True
    
    # Check if any vertex of j is inside i
    for v in verts_j:
        if _point_in_convex_polygon_fast(v, verts_i):
            return True
    
    # Check edge intersections
    for k in range(6):
        a1 = verts_i[k]
        b1 = verts_i[(k+1)%6]
        for l in range(6):
            a2 = verts_j[l]
            b2 = verts_j[(l+1)%6]
            if _segments_intersect(a1, b1, a2, b2):
                return True
    
    return False

def _point_in_convex_polygon_fast(p, polygon):
    """Fast check if point is inside convex polygon."""
    px, py = p
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i+1)%n]
        cross = (x2-x1)*(py-y1) - (y2-y1)*(px-x1)
        if cross < -1e-12:
            return False
    return True

def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    
    c1 = cross(p3, p4, p1)
    c2 = cross(p3, p4, p2)
    c3 = cross(p1, p2, p3)
    c4 = cross(p1, p2, p4)
    
    if ((c1 > 0 and c2 < 0) or (c1 < 0 and c2 > 0)) and \
       ((c3 > 0 and c4 < 0) or (c3 < 0 and c4 > 0)):
        return True
    
    if abs(c1) < 1e-12 and _on_segment(p3, p4, p1):
        return True
    if abs(c2) < 1e-12 and _on_segment(p3, p4, p2):
        return True
    if abs(c3) < 1e-12 and _on_segment(p1, p2, p3):
        return True
    if abs(c4) < 1e-12 and _on_segment(p1, p2, p4):
        return True
    
    return False

def _on_segment(a, b, p):
    """Check if point p is on segment a-b."""
    return (min(a[0], b[0]) - 1e-12 <= p[0] <= max(a[0], b[0]) + 1e-12 and
            min(a[1], b[1]) - 1e-12 <= p[1] <= max(a[1], b[1]) + 1e-12)

def _check_disjoint(inner_positions, inner_angles, tol=1e-9):
    """Check that all inner hexagons are disjoint (no overlap)."""
    n = len(inner_positions)
    for i in range(n):
        for j in range(i+1, n):
            if _hexagons_overlap_fast(inner_positions[i], inner_angles[i], 
                                      inner_positions[j], inner_angles[j]):
                return False
    return True

def _find_min_enclosing_hex_side_lp(vertices, outer_angle_deg):
    """
    Find the minimum side length of a regular hexagon with given angle
    that contains all given vertices, using LP.
    Returns (center_x, center_y, side_length) or None.
    """
    normals = []
    for k in range(6):
        a = math.radians(outer_angle_deg + 30 + 60 * k)
        normals.append((math.cos(a), math.sin(a)))
    
    A = []
    b = []
    
    for vx, vy in vertices:
        for nx, ny in normals:
            v_dot_n = vx * nx + vy * ny
            
            # (c - v) . n <= R  =>  nx*cx + ny*cy - R <= v_dot_n
            A.append([nx, ny, -1.0])
            b.append(v_dot_n)
            
            # -(c - v) . n <= R  =>  -nx*cx - ny*cy - R <= -v_dot_n
            A.append([-nx, -ny, -1.0])
            b.append(-v_dot_n)
            
    # R >= 0
    A.append([0.0, 0.0, -1.0])
    b.append(0.0)
    
    c = [0.0, 0.0, 1.0]
    
    try:
        res = linprog(c, A_ub=A, b_ub=b, 
                     bounds=[(None, None), (None, None), (0, None)],
                     method='highs')
        if res.success:
            cx, cy, R = res.x
            side = R / (math.sqrt(3) / 2)
            return cx, cy, side
        else:
            return None
    except:
        return None

def _compute_all_vertices(positions, angles):
    """Compute all 6*N vertices for given positions and angles."""
    all_verts = []
    for (px, py), ang in zip(positions, angles):
        all_verts.extend(_get_hex_vertices(px, py, ang))
    return all_verts

def _smooth_penalty_objective(params, beta=50.0):
    """
    Differentiable objective: outer hexagon side length + penalty for overlaps.
    Uses soft-min for the overlap penalty to make it differentiable.
    """
    n = len(params) // 3
    positions = [(params[3*i], params[3*i+1]) for i in range(n)]
    angles = [params[3*i+2] for i in range(n)]
    
    # Compute all pairwise gaps
    gaps = []
    for i in range(n):
        for j in range(i+1, n):
            gap = _pairwise_min_gap(positions[i], angles[i], positions[j], angles[j])
            gaps.append(gap)
    
    # Soft penalty: use log-sum-exp to approximate min(gaps)
    # If any gap is negative (overlap), we want to penalize heavily
    # Use: -log(sum(exp(-beta * gap))) / beta approximates min(gap)
    # For large negative gaps, this gives large positive penalty
    if len(gaps) > 0:
        # Clip to avoid numerical issues
        gaps_clipped = [max(g, -10.0) for g in gaps]
        # Compute soft-min
        max_neg = -min(gaps_clipped)  # This is the most negative gap (most overlap)
        if max_neg > 1e-10:
            # There's overlap; compute penalty
            exps = [math.exp(-beta * g) for g in gaps_clipped]
            sum_exps = sum(exps)
            if sum_exps > 0:
                soft_min = -(1.0/beta) * math.log(sum_exps)
                # Penalty is large when soft_min is very negative
                penalty = max(0.0, -soft_min) ** 2 * 100.0
            else:
                penalty = 0.0
        else:
            penalty = 0.0
    else:
        penalty = 0.0
    
    # Compute outer hexagon side length
    all_verts = _compute_all_vertices(positions, angles)
    
    best_side = float('inf')
    # Try a few outer angles
    for outer_angle in [0.0, 15.0, 30.0]:
        res = _find_min_enclosing_hex_side_lp(all_verts, outer_angle)
        if res:
            side = res[2]
            if side < best_side:
                best_side = side
    
    if best_side == float('inf'):
        best_side = 100.0
    
    return best_side + penalty

def _nlp_optimize(positions, angles, max_iter=300, beta_start=20.0, beta_end=200.0):
    """
    Use NLP optimization with smooth penalty to find better packing.
    Returns improved positions and angles.
    """
    n = len(positions)
    
    # Flatten parameters
    x0 = []
    for i in range(n):
        x0.extend([positions[i][0], positions[i][1], angles[i]])
    
    # Adaptive beta schedule
    def objective_with_adaptive_beta(params, beta_holder):
        return _smooth_penalty_objective(params, beta_holder[0])
    
    # Use multiple restarts with different betas
    best_params = x0.copy()
    best_obj = float('inf')
    
    # Try increasing beta for sharper penalty
    for beta in [beta_start, 50.0, 100.0, 200.0]:
        if time.time() > _DEADLINE - 20.0:
            break
        
        def obj(params):
            return _smooth_penalty_objective(params, beta)
        
        try:
            res = minimize(obj, x0, method='L-BFGS-B', 
                          options={'maxiter': max_iter, 'gtol': 1e-6, 'ftol': 1e-8})
            if res.success:
                obj_val = res.fun
                if obj_val < best_obj:
                    best_obj = obj_val
                    best_params = res.x.copy()
        except:
            pass
        
        # Update x0 to best found so far for next beta
        x0 = best_params.copy()
    
    # Extract results
    n = len(best_params) // 3
    new_positions = [(best_params[3*i], best_params[3*i+1]) for i in range(n)]
    new_angles = [best_params[3*i+2] for i in range(n)]
    
    # Verify disjointness
    if _check_disjoint(new_positions, new_angles):
        return new_positions, new_angles
    
    # If NLP result has overlaps, try to repair by scaling up slightly
    # or fall back to original
    return positions, angles

def _nelder_mead_polish(positions, angles, max_iter=200, tol=1e-6):
    """
    Polish the inner positions using Nelder-Mead.
    Returns improved positions and angles.
    """
    # Flatten parameters: x0, y0, x1, y1, ..., angle0, angle1, ...
    x0 = []
    for px, py in positions:
        x0.extend([px, py])
    x0.extend(angles)
    
    def objective(params):
        n = len(positions)
        pos = [(params[2*i], params[2*i+1]) for i in range(n)]
        ang = list(params[2*n:])
        
        # Check disjointness
        if not _check_disjoint(pos, ang):
            return 1e6
        
        all_verts = _compute_all_vertices(pos, ang)
        
        # Find best outer angle
        best_side = float('inf')
        for outer_angle in np.arange(0, 30.1, 2.0):
            res = _find_min_enclosing_hex_side_lp(all_verts, outer_angle)
            if res:
                side = res[2]
                if side < best_side:
                    best_side = side
        
        return best_side
    
    # Use Nelder-Mead
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol})
        if res.success:
            n = len(positions)
            new_pos = [(res.x[2*i], res.x[2*i+1]) for i in range(n)]
            new_ang = list(res.x[2*n:])
            return new_pos, new_ang
    except:
        pass
    
    return positions, angles

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    sqrt3 = math.sqrt(3)
    
    # Start with hexagonal cluster arrangement
    scale = 1.000001  # Slightly larger than touching
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
    
    # Generate positions on triangular lattice
    lattice_positions = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            lattice_positions.append((x, y))
    
    # Sort by distance from origin and take N closest
    lattice_positions.sort(key=lambda p: p[0]*p[0] + p[1]*p[1])
    initial_positions = lattice_positions[:N]
    initial_angles = [0.0] * N
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Try different inner angle configurations
    angle_configs = [
        [0.0] * N,
        [30.0] * N,
    ]
    
    for inner_angles in angle_configs:
        if time.time() > _DEADLINE - 15.0:
            break
        
        positions = initial_positions
        angles = inner_angles
        
        # First pass: find best outer angle for initial configuration
        all_verts = _compute_all_vertices(positions, angles)
        
        best_coarse_side = float('inf')
        best_coarse_angle = 0.0
        best_coarse_center = (0.0, 0.0)
        
        # Coarse search for outer angle
        for outer_angle_deg in np.arange(0, 30.1, 1.0):
            if time.time() > _DEADLINE - 15.0:
                break
            res = _find_min_enclosing_hex_side_lp(all_verts, outer_angle_deg)
            if res:
                cx, cy, side = res
                if side < best_coarse_side:
                    best_coarse_side = side
                    best_coarse_angle = outer_angle_deg
                    best_coarse_center = (cx, cy)
        
        # Fine search around best coarse angle
        if best_coarse_angle > 0:
            for outer_angle_deg in np.arange(max(0, best_coarse_angle - 1.0), 
                                            min(30, best_coarse_angle + 1.0) + 0.01, 0.05):
                if time.time() > _DEADLINE - 15.0:
                    break
                res = _find_min_enclosing_hex_side_lp(all_verts, outer_angle_deg)
                if res:
                    cx, cy, side = res
                    if side < best_coarse_side:
                        best_coarse_side = side
                        best_coarse_angle = outer_angle_deg
                        best_coarse_center = (cx, cy)
        
        if best_coarse_side < best_side:
            best_side = best_coarse_side
            best_inner = [[pos[0], pos[1], ang] for pos, ang in zip(positions, angles)]
            best_center = best_coarse_center
            best_outer_angle = best_coarse_angle
        
        # NLP optimization with smooth penalty
        if time.time() < _DEADLINE - 40.0:
            nlp_pos, nlp_ang = _nlp_optimize(positions, angles, max_iter=200)
            
            if nlp_pos != positions or nlp_ang != angles:
                # Verify disjointness
                if _check_disjoint(nlp_pos, nlp_ang):
                    all_verts_nlp = _compute_all_vertices(nlp_pos, nlp_ang)
                    
                    # Find best outer angle for NLP configuration
                    best_nlp_side = float('inf')
                    best_nlp_angle = 0.0
                    best_nlp_center = (0.0, 0.0)
                    
                    for outer_angle_deg in np.arange(0, 30.1, 0.5):
                        if time.time() > _DEADLINE - 15.0:
                            break
                        res = _find_min_enclosing_hex_side_lp(all_verts_nlp, outer_angle_deg)
                        if res:
                            cx, cy, side = res
                            if side < best_nlp_side:
                                best_nlp_side = side
                                best_nlp_angle = outer_angle_deg
                                best_nlp_center = (cx, cy)
                    
                    # Fine polish on outer angle
                    if best_nlp_angle > 0:
                        for outer_angle_deg in np.arange(max(0, best_nlp_angle - 0.5), 
                                                        min(30, best_nlp_angle + 0.5) + 0.01, 0.01):
                            if time.time() > _DEADLINE - 15.0:
                                break
                            res = _find_min_enclosing_hex_side_lp(all_verts_nlp, outer_angle_deg)
                            if res:
                                cx, cy, side = res
                                if side < best_nlp_side:
                                    best_nlp_side = side
                                    best_nlp_angle = outer_angle_deg
                                    best_nlp_center = (cx, cy)
                    
                    if best_nlp_side < best_side:
                        best_side = best_nlp_side
                        best_inner = [[pos[0], pos[1], ang] for pos, ang in zip(nlp_pos, nlp_ang)]
                        best_center = best_nlp_center
                        best_outer_angle = best_nlp_angle
        
        # Nelder-Mead polish
        if time.time() < _DEADLINE - 25.0:
            polished_pos, polished_ang = _nelder_mead_polish(positions, angles, max_iter=100)
            
            if polished_pos != positions or polished_ang != angles:
                # Verify disjointness
                if _check_disjoint(polished_pos, polished_ang):
                    all_verts_polished = _compute_all_vertices(polished_pos, polished_ang)
                    
                    # Find best outer angle for polished configuration
                    best_polished_side = float('inf')
                    best_polished_angle = 0.0
                    best_polished_center = (0.0, 0.0)
                    
                    for outer_angle_deg in np.arange(0, 30.1, 0.5):
                        if time.time() > _DEADLINE - 15.0:
                            break
                        res = _find_min_enclosing_hex_side_lp(all_verts_polished, outer_angle_deg)
                        if res:
                            cx, cy, side = res
                            if side < best_polished_side:
                                best_polished_side = side
                                best_polished_angle = outer_angle_deg
                                best_polished_center = (cx, cy)
                    
                    # Fine polish on outer angle
                    if best_polished_angle > 0:
                        for outer_angle_deg in np.arange(max(0, best_polished_angle - 0.5), 
                                                        min(30, best_polished_angle + 0.5) + 0.01, 0.01):
                            if time.time() > _DEADLINE - 15.0:
                                break
                            res = _find_min_enclosing_hex_side_lp(all_verts_polished, outer_angle_deg)
                            if res:
                                cx, cy, side = res
                                if side < best_polished_side:
                                    best_polished_side = side
                                    best_polished_angle = outer_angle_deg
                                    best_polished_center = (cx, cy)
                    
                    if best_polished_side < best_side:
                        best_side = best_polished_side
                        best_inner = [[pos[0], pos[1], ang] for pos, ang in zip(polished_pos, polished_ang)]
                        best_center = best_polished_center
                        best_outer_angle = best_polished_angle
    
    if best_inner is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_outer_angle
