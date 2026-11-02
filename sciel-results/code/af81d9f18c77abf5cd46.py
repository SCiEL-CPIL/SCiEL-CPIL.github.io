import math
import numpy as np
import time
from scipy.optimize import minimize_scalar, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def _hex_vertices(cx, cy, s, rot_deg):
    a = math.radians(rot_deg)
    vs = []
    for k in range(6):
        ang = a + k * math.pi / 3.0
        vs.append((cx + s * math.cos(ang), cy + s * math.sin(ang)))
    return vs

def _point_in_hex(px, py, cx, cy, s, rot_deg):
    dx, dy = px - cx, py - cy
    a = math.radians(rot_deg)
    ca, sa = math.cos(a), math.sin(a)
    lx, ly = ca * dx + sa * dy, -sa * dx + ca * dy
    ap = math.sqrt(3) / 2.0 * s
    if abs(ly) > ap + 1e-9:
        return False
    if abs(lx) > s + 1e-9:
        return False
    for k in range(6):
        ang = k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        if lx * nx + ly * ny > s * math.cos(math.pi/6) + 1e-9:
            return False
    return True

def _hexes_disjoint(v1, v2, tol=1e-9):
    def seg_intersect(p1, p2, p3, p4):
        x1,y1 = p1; x2,y2 = p2; x3,y3 = p3; x4,y4 = p4
        d1x, d1y = x2-x1, y2-y1
        d2x, d2y = x4-x3, y4-y3
        denom = d1x*d2y - d1y*d2x
        if abs(denom) < 1e-15:
            return False
        t = ((x3-x1)*d2y - (y3-y1)*d2x) / denom
        u = ((x3-x1)*d1y - (y3-y1)*d1x) / denom
        return -tol <= t <= 1+tol and -tol <= u <= 1+tol

    for i in range(6):
        for j in range(6):
            if seg_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                return False
    cx1 = sum(p[0] for p in v1)/6; cy1 = sum(p[1] for p in v1)/6
    cx2 = sum(p[0] for p in v2)/6; cy2 = sum(p[1] for p in v2)/6
    if _point_in_hex(cx1, cy1, cx2, cy2, 1.0, 0.0) or _point_in_hex(cx2, cy2, cx1, cy1, 1.0, 0.0):
        return False
    return True

def construct():
    if time.time() > START_TIME + TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Generate diverse lattice-based candidates
    # We use axial coordinates (q, r) for the hexagonal grid.
    # Distance between centers is sqrt(3) for side length 1.
    # We generate sets of 11 non-overlapping hexagons.
    
    candidates = set()
    
    # Helper to add a tuple of axial coords to candidates
    def add_candidate(axial_list):
        # Canonical form: sort and normalize relative to center of mass
        # To avoid duplicates, we can just use a frozenset of the relative positions
        # But for simplicity, we'll just store the list and deduplicate later if needed
        # Or better, store the set of relative coordinates rounded to avoid float issues
        # Since these are integers, we can just use a frozenset of the tuples
        # However, translation invariance means (0,0) should be one of them.
        # Let's normalize by subtracting the first point
        if not axial_list: return
        base_q, base_r = axial_list[0]
        norm_list = tuple(sorted((q - base_q, r - base_r) for q, r in axial_list))
        candidates.add(norm_list)

    # 1. Start with the center hexagon
    # 2. Grow the cluster by adding adjacent hexagons
    # We'll do a breadth-first search or just random growth to generate many shapes
    
    def generate_clusters():
        clusters = []
        # Start with a single hexagon at (0,0)
        current = [(0, 0)]
        clusters.append(current[:])
        
        # Directions in axial coordinates
        dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        
        # We want to generate clusters of size 11
        # We'll use a recursive generation with a limit on depth/branching
        # To keep it fast, we'll limit the number of clusters generated
        
        max_clusters = 500
        generated = 0
        
        # Use a queue for BFS-like growth
        queue = [current]
        visited_states = {tuple(current)}
        
        while queue and generated < max_clusters:
            curr_cluster = queue.pop(0)
            if len(curr_cluster) >= N:
                if len(curr_cluster) == N:
                    add_candidate(curr_cluster)
                    generated += 1
                continue
            
            # Try adding a new hexagon adjacent to any existing one
            # To avoid symmetry, we can enforce an ordering or just generate all and dedup
            # Let's just generate all possible additions
            
            possible_additions = []
            for q, r in curr_cluster:
                for dq, dr in dirs:
                    nq, nr = q + dq, r + dr
                    # Check if (nq, nr) is already in the cluster
                    if (nq, nr) not in curr_cluster:
                        # Check if it's a "new" direction to reduce symmetry? 
                        # For now, just add it. Deduplication handles the rest.
                        possible_additions.append((nq, nr))
            
            # Remove duplicates in possible_additions
            unique_additions = list(set(possible_additions))
            
            for nq, nr in unique_additions:
                new_cluster = curr_cluster + [(nq, nr)]
                state = tuple(sorted(new_cluster))
                if state not in visited_states:
                    visited_states.add(state)
                    queue.append(new_cluster)
                    
        return clusters

    # Since full BFS might be slow or generate too many, let's use a smarter approach:
    # Predefined "good" shapes + Random Perturbations
    
    # Known good shapes (from previous iterations)
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    axial_4 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    axial_5 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    axial_6 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    axial_7 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    axial_8 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    axial_9 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ]
    axial_10 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    
    # Add these to candidates
    for ax in [axial_1, axial_2, axial_3, axial_4, axial_5, axial_6, axial_7, axial_8, axial_9, axial_10]:
        add_candidate(ax)

    # Generate more via simple random growth
    import random
    random.seed(42)
    
    def random_growth():
        cluster = [(0,0)]
        dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        while len(cluster) < N:
            # Pick a random existing hex
            q, r = random.choice(cluster)
            # Pick a random direction
            dq, dr = random.choice(dirs)
            nq, nr = q + dq, r + dr
            if (nq, nr) not in cluster:
                cluster.append((nq, nr))
        return cluster

    for _ in range(200):
        if time.time() > START_TIME + TIME_LIMIT - SAFETY_MARGIN:
            break
        add_candidate(random_growth())

    # Convert axial to cartesian
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    # Function to calculate minimum enclosing hexagon side length for a set of points
    def min_enclosing_hex_side(pts):
        """
        Finds the minimum side length S of a regular hexagon that encloses the points.
        Returns (S, center_x, center_y, rotation_angle_rad)
        """
        # The width of the hexagon in direction theta is max(proj) - min(proj).
        # For a regular hexagon with side S, the width in any direction is constant? No.
        # The width of a regular hexagon of side S is 2 * apothem = S * sqrt(3) in the normal directions to the sides.
        # In the directions of the vertices, the width is 2 * S.
        # Actually, the support function of a regular hexagon centered at origin with side S and rotation alpha is:
        # h(theta) = S * cos(pi/6) * |cos(theta - alpha - k*pi/3)| ... wait.
        
        # Let's use the property that a regular hexagon is the intersection of 6 half-planes.
        # Alternatively, we can optimize the rotation angle alpha.
        # For a fixed alpha, the optimal center is the center of the bounding box in the 3 principal axes of the hexagon.
        # The 3 axes are separated by 60 degrees.
        
        def width_for_alpha(alpha):
            # Normals to the sides of the hexagon
            # If rotation is alpha, the normals are at alpha + pi/6 + k*pi/3
            normals = []
            for k in range(3):
                ang = alpha + math.pi / 6.0 + k * math.pi / 3.0
                normals.append((math.cos(ang), math.sin(ang)))
            
            # Project points onto these 3 axes
            # For each axis, we need the range of projections.
            # The constraint is that the distance between the two parallel sides of the hexagon
            # must be at least the range of projections.
            # Distance between sides = 2 * apothem = S * sqrt(3).
            # So S * sqrt(3) >= range(proj) for all 3 axes.
            # Thus S >= max(range(proj)) / sqrt(3).
            
            max_range = 0
            for nx, ny in normals:
                projs = pts[:, 0] * nx + pts[:, 1] * ny
                rng = np.max(projs) - np.min(projs)
                if rng > max_range:
                    max_range = rng
            
            return max_range / math.sqrt(3)
        
        # Optimize alpha in [0, pi/3)
        # The function is periodic with period pi/3.
        # It is also convex-ish? Not necessarily, but we can use a fine grid + local opt.
        
        # Coarse grid
        n_grid = 100
        best_s = float('inf')
        best_alpha = 0
        
        for i in range(n_grid):
            alpha = i * (math.pi / 3.0) / n_grid
            s = width_for_alpha(alpha)
            if s < best_s:
                best_s = s
                best_alpha = alpha
        
        # Refine with minimize_scalar
        try:
            res = minimize_scalar(width_for_alpha, bounds=(0, math.pi/3), method='bounded', options={'xatol': 1e-12})
            if res.fun < best_s:
                best_s = res.fun
                best_alpha = res.x
        except Exception:
            pass
        
        # Now that we have the optimal rotation alpha, find the optimal center.
        # The center should be the midpoint of the bounding box for each axis.
        # For axis k, center_proj = (max_proj + min_proj) / 2.
        # We have 3 axes. We need to find (cx, cy) such that:
        # nx1*cx + ny1*cy = mid1
        # nx2*cx + ny2*cy = mid2
        # nx3*cx + ny3*cy = mid3
        # This is an overdetermined system. We can use least squares.
        
        normals = []
        mids = []
        for k in range(3):
            ang = best_alpha + math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            normals.append((nx, ny))
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            mid = (np.max(projs) + np.min(projs)) / 2.0
            mids.append(mid)
        
        A = np.array(normals) # 3x2
        b = np.array(mids)    # 3x1
        
        # Solve A @ [cx, cy]^T = b
        try:
            center, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            cx, cy = center
        except Exception:
            cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
            
        return best_s, cx, cy, best_alpha

    best_global_s = float('inf')
    best_global_pts = None
    best_global_center = None
    best_global_alpha = 0.0

    # Evaluate all candidates
    for i, cand in enumerate(candidates):
        if time.time() > START_TIME + TIME_LIMIT - SAFETY_MARGIN:
            break
        
        # Convert to cartesian
        pts = axial_to_cartesian(list(cand))
        
        # Quick rejection: if bounding box is too large, skip
        # Min side length for 11 hexagons is roughly sqrt(11 * area_hex / area_hex_container)
        # Area hex = 3*sqrt(3)/2 * S^2.
        # 11 * (3*sqrt(3)/2) <= S^2 * (3*sqrt(3)/2) * (S^2/1) -> S^2 >= 11 -> S >= sqrt(11) ~ 3.3
        # Let's say if S > 4.0, it's likely not optimal, but we still check.
        
        s, cx, cy, alpha = min_enclosing_hex_side(pts)
        
        if s < best_global_s:
            best_global_s = s
            best_global_pts = pts
            best_global_center = (cx, cy)
            best_global_alpha = alpha

    if best_global_pts is None:
        # Fallback
        pts = axial_to_cartesian(axial_1)
        best_global_s, cx, cy, alpha = min_enclosing_hex_side(pts)
        best_global_center = (cx, cy)
        best_global_alpha = alpha
        best_global_pts = pts

    # Final Polish: Optimize center and rotation simultaneously for the best cluster
    # We already did a scalar optimization for rotation assuming optimal center.
    # Now we can do a joint Nelder-Mead optimization for (cx, cy, alpha).
    
    def objective(params):
        cx, cy, alpha = params
        alpha = ((alpha % (math.pi/3)) + math.pi/3) % (math.pi/3)
        
        normals = []
        for k in range(3):
            ang = alpha + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
            
        max_range = 0
        for nx, ny in normals:
            projs = (best_global_pts[:, 0] - cx) * nx + (best_global_pts[:, 1] - cy) * ny
            rng = np.max(projs) - np.min(projs)
            if rng > max_range:
                max_range = rng
        
        return max_range / math.sqrt(3)

    init = [best_global_center[0], best_global_center[1], best_global_alpha]
    
    try:
        if time.time() < START_TIME + TIME_LIMIT - SAFETY_MARGIN:
            res = minimize(objective, init, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_global_s:
                best_global_s = res.fun
                best_global_center = (res.x[0], res.x[1])
                best_global_alpha = ((res.x[2] % (math.pi/3)) + math.pi/3) % (math.pi/3)
    except Exception:
        pass

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_alpha)
    
    # Return format
    inner = []
    for x, y in best_global_pts:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(best_global_s)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation

