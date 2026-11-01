"""
Combined program: discrete polyhex enumeration with continuous refinement.
Uses branch-and-bound over hexagonal dual lattice to find compact 11-cell polyhexes,
then applies LP-based outer optimization and joint continuous refinement.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar, linprog
from itertools import combinations

N = 11
SQRT3 = math.sqrt(3)
APOTHEM = SQRT3 / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 8.0

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as numpy array (6,2)."""
    thetas = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    normals = np.array([[math.cos(math.radians(30 + i * 60)),
                         math.sin(math.radians(30 + i * 60))] for i in range(6)])
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    max_proj = -np.inf
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        local = (verts - np.array([ox, oy])) @ R.T
        projs = local @ normals.T
        cur = projs.max()
        if cur > max_proj:
            max_proj = cur
    
    if max_proj < 0:
        max_proj = 0.0
    return 2.0 * max_proj / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=80.0):
    """
    Smooth surrogate for compute_required_side using log-sum-exp.
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    normals = np.array([[math.cos(math.radians(30 + i * 60)),
                         math.sin(math.radians(30 + i * 60))] for i in range(6)])
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    
    all_projs = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        local = (verts - np.array([ox, oy])) @ R.T
        projs = local @ normals.T
        all_projs.append(projs)
    
    P = np.concatenate([p.ravel() for p in all_projs])
    if P.size == 0:
        return 0.0
    
    m = P.max()
    if m < -1e6:
        return 0.0
    smooth_max = m + math.log(np.sum(np.exp(beta * (P - m)))) / beta
    if smooth_max < 0:
        smooth_max = 0.0
    return 2.0 * smooth_max / SQRT3

def sat_overlap(v1, v2, eps=1e-9):
    """
    Check if two convex polygons overlap using SAT.
    v1, v2: numpy arrays of shape (n, 2)
    Returns True if they overlap (including touching).
    """
    axes = []
    
    for poly in (v1, v2):
        n_pts = len(poly)
        for i in range(n_pts):
            p1 = poly[i]
            p2 = poly[(i + 1) % n_pts]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
    
    for axis in axes:
        proj1 = v1 @ axis
        proj2 = v2 @ axis
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    
    return True

def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    polys = []
    for (x, y, a) in inner:
        polys.append(get_hex_vertices(x, y, 1.0, math.radians(a)))
    
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap(polys[i], polys[j], eps):
                return False
    return True

def compute_overlap_penalty(inner, min_dist=SQRT3 - 0.005):
    """
    Compute a smooth penalty for overlaps based on center distances.
    """
    n = len(inner)
    if n < 2:
        return 0.0
    
    xs = np.array([inner[i][0] for i in range(n)])
    ys = np.array([inner[i][1] for i in range(n)])
    
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dists_sq = dx * dx + dy * dy
    
    iu = np.triu_indices(n, k=1)
    dists_sq_upper = dists_sq[iu]
    dists = np.sqrt(dists_sq_upper)
    
    diff = np.maximum(0.0, min_dist - dists)
    penalty = np.sum(diff * diff)
    
    return float(penalty)

def axial_to_cartesian(axial_list, scale=1.0):
    """Convert axial coordinates to cartesian with given scale."""
    pts = []
    for q, r in axial_list:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        pts.append((x, y))
    return pts

def calc_S_for_beta(pts, beta):
    """
    Use LP to find the minimum circumradius D for a set of points
    given outer hexagon rotation beta.
    Returns (S, center, D) or (inf, None, None).
    """
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return float('inf'), None, None
    
    normals = []
    for k in range(6):
        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(ang), math.sin(ang)))
    
    c_obj = np.array([0.0, 0.0, 1.0])
    A_ub = []
    b_ub = []
    
    for nx, ny in normals:
        for px, py in pts:
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-(px * nx + py * ny))
            A_ub.append([nx, ny, -1.0])
            b_ub.append(px * nx + py * ny)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(None, None), (None, None), (0, None)]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            D = res.x[2]
            cx = res.x[0]
            cy = res.x[1]
            S = (D + APOTHEM) / APOTHEM
            return S, (cx, cy), D
        else:
            return float('inf'), None, None
    except Exception:
        return float('inf'), None, None

def optimize_outer_for_inner(inner):
    """
    Optimize outer hexagon (center, angle) to minimize side length.
    inner: list of (x, y, angle_deg)
    Returns (side, center, angle_deg)
    """
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return compute_required_side(inner, (0, 0), 0), (0, 0), 0.0
    
    all_verts = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        all_verts.extend(verts)
    
    pts = np.array(all_verts)
    
    best_S = float('inf')
    best_beta = 0.0
    best_center = (0, 0)
    
    # Coarse search over angles
    for i in range(37):
        beta = i * (math.pi / 6) / 36
        S, center, D = calc_S_for_beta(pts, beta)
        if S < best_S:
            best_S = S
            best_beta = beta
            best_center = center
    
    # Fine search using minimize_scalar
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0], 
                                 bounds=(0, math.pi / 6), method='bounded',
                                 options={'maxiter': 100, 'xatol': 1e-10})
            if res.fun < best_S:
                best_S = res.fun
                best_beta = res.x
                _, best_center, _ = calc_S_for_beta(pts, best_beta)
    except Exception:
        pass
    
    best_angle_deg = math.degrees(best_beta)
    return best_S, best_center, best_angle_deg

def polyhex_diameter_width(axial_cells):
    """
    Compute a lower bound on the enclosing hexagon side length from diameter and width.
    For a set of unit hexagons, the minimum enclosing hexagon side is bounded below
    by the maximum of:
      - (diameter / 2 + apothem) / apothem  [for the direction of max extent]
      - (width / 2 + apothem) / apothem     [for perpendicular directions]
    We use a simpler proxy: the circumradius of the centers plus the apothem of inner hex.
    """
    centers = axial_to_cartesian(axial_cells, 1.0)
    if not centers:
        return 0.0
    cx = np.mean([c[0] for c in centers])
    cy = np.mean([c[1] for c in centers])
    max_r = max(math.hypot(c[0] - cx, c[1] - cy) for c in centers)
    # Lower bound: side >= (max_r + APOTHEM) / APOTHEM
    return (max_r + APOTHEM) / APOTHEM

def enumerate_polyhexes(n_cells, max_cells=11):
    """
    Enumerate all free polyhexes of size n_cells using a growth approach.
    Returns list of axial coordinate sets (normalized to have min q,r at origin).
    """
    if n_cells <= 0:
        return []
    if n_cells == 1:
        return [frozenset([(0, 0)])]
    
    # Directions for hexagonal grid
    dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    # Use a set to avoid duplicates
    polyhexes = set()
    
    # Start from all polyhexes of size n_cells-1
    prev = enumerate_polyhexes(n_cells - 1, max_cells)
    
    for ph in prev:
        # Get all cells
        cells = list(ph)
        # Find all empty neighbors
        candidates = set()
        for c in cells:
            q, r = c
            for dq, dr in dirs:
                nc = (q + dq, r + dr)
                if nc not in cells:
                    candidates.add(nc)
        
        for nc in candidates:
            new_ph = frozenset(ph | {nc})
            # Normalize: shift so that min q is 0, and for min q, min r is 0
            min_q = min(c[0] for c in new_ph)
            min_r_at_min_q = min(c[1] for c in new_ph if c[0] == min_q)
            normalized = frozenset((c[0] - min_q, c[1] - min_r_at_min_q) for c in new_ph)
            polyhexes.add(normalized)
    
    return [list(ph) for ph in polyhexes]

def find_compact_polyhexes(n_cells, max_candidates=20):
    """
    Find the most compact polyhexes of size n_cells by diameter/width lower bound.
    """
    if time.time() - START_TIME > TIME_LIMIT - 30:
        return []
    
    all_polyhexes = enumerate_polyhexes(n_cells)
    
    # Score each by lower bound on enclosing side
    scored = []
    for ph in all_polyhexes:
        lb = polyhex_diameter_width(ph)
        scored.append((lb, ph))
    
    # Sort by lower bound
    scored.sort(key=lambda x: x[0])
    
    # Return top candidates
    return [ph for lb, ph in scored[:max_candidates]]

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    # Phase 1: Discrete enumeration of compact polyhexes
    candidates = []
    
    # Find compact polyhexes via enumeration
    if time.time() < deadline - 60:
        compact = find_compact_polyhexes(N, max_candidates=15)
        for ph in compact:
            candidates.append(ph)
    
    # Add some known good layouts as fallbacks
    axial_fallback_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 1), (-1, 2), (-2, 1)
    ]
    candidates.append(axial_fallback_1)
    
    axial_fallback_2 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    candidates.append(axial_fallback_2)
    
    axial_fallback_3 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_fallback_3)
    
    # Remove duplicates and limit candidates
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = frozenset(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    candidates = unique_candidates[:25]
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Phase 2: Evaluate each candidate layout
    for layout_idx, axial_coords in enumerate(candidates):
        if time.time() > deadline - 40:
            break
        
        if len(axial_coords) > N:
            axial_coords = axial_coords[:N]
        if len(axial_coords) < N:
            continue
        
        # Convert to cartesian with slight scale to ensure disjointness
        epsilon = 1e-4
        scale = 1.0 + epsilon
        
        centers = axial_to_cartesian(axial_coords, scale)
        initial_inner = [[x, y, 0.0] for x, y in centers]
        
        # Optimize outer hexagon using LP
        side, center, angle_deg = optimize_outer_for_inner(initial_inner)
        
        if side < best_side:
            best_side = side
            best_inner = initial_inner
            best_outer_center = center
            best_outer_angle = angle_deg
    
    if best_inner is None:
        # Fallback
        axial_coords = candidates[0][:N]
        centers = axial_to_cartesian(axial_coords, 1.0 + 1e-4)
        best_inner = [[x, y, 0.0] for x, y in centers]
        side, center, angle_deg = optimize_outer_for_inner(best_inner)
        best_side = side
        best_outer_center = center
        best_outer_angle = angle_deg
    
    # Phase 3: Joint continuous refinement with simulated annealing
    def objective_joint_smooth(params, beta=50.0):
        inner_list = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            a = params[i*3+2]
            inner_list.append([x, y, a])
        
        ocx = params[N*3]
        ocy = params[N*3+1]
        oangle = params[N*3+2]
        
        side = compute_required_side_smooth(inner_list, (ocx, ocy), oangle, beta)
        penalty = compute_overlap_penalty(inner_list)
        
        return side + 1000.0 * penalty
    
    initial_params = []
    for x, y, a in best_inner:
        initial_params.extend([x, y, a])
    initial_params.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])
    
    current_params = np.array(initial_params)
    current_val = objective_joint_smooth(current_params)
    best_params = current_params.copy()
    best_val_joint = current_val
    
    # Simulated annealing
    np.random.seed(42)
    T = 0.5
    T_min = 0.001
    cool = 0.9998
    max_iter = 5000
    
    for it in range(max_iter):
        if time.time() > deadline - 25:
            break
        
        neighbor = current_params.copy()
        r = np.random.random()
        
        if r < 0.60:
            # Perturb one inner hexagon position
            idx = np.random.randint(0, N)
            dim = np.random.randint(0, 2)
            neighbor[idx * 3 + dim] += np.random.normal(0, 0.2)
        elif r < 0.75:
            # Perturb one inner hexagon angle
            idx = np.random.randint(0, N)
            neighbor[idx * 3 + 2] += np.random.normal(0, 8.0)
        elif r < 0.85:
            # Perturb outer center
            dim = np.random.randint(0, 2)
            neighbor[N * 3 + dim] += np.random.normal(0, 0.1)
        else:
            # Perturb outer angle
            neighbor[N * 3 + 2] += np.random.normal(0, 5.0)
        
        neighbor_val = objective_joint_smooth(neighbor)
        
        if neighbor_val < current_val:
            current_params = neighbor
            current_val = neighbor_val
            if current_val < best_val_joint:
                best_params = current_params.copy()
                best_val_joint = current_val
        else:
            # Accept with probability exp(-delta/T)
            if np.random.random() < math.exp(-(neighbor_val - current_val) / max(T, 1e-10)):
                current_params = neighbor
                current_val = neighbor_val
        
        T = max(T * cool, T_min)
    
    # Extract best joint result
    best_inner_joint = []
    for i in range(N):
        x = best_params[i*3]
        y = best_params[i*3+1]
        a = best_params[i*3+2]
        best_inner_joint.append([x, y, a])
    
    best_ocx = best_params[N*3]
    best_ocy = best_params[N*3+1]
    best_oangle = best_params[N*3+2]
    
    # Check disjointness and update if valid
    if check_all_disjoint(best_inner_joint, eps=1e-5):
        best_inner = best_inner_joint
        best_outer_center = (best_ocx, best_ocy)
        best_outer_angle = best_oangle
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)
    
    # Phase 4: Final polish using LP-based outer optimization
    if time.time() < deadline - 5:
        side, center, angle_deg = optimize_outer_for_inner(best_inner)
        if side < best_side:
            best_side = side
            best_outer_center = center
            best_outer_angle = angle_deg
    
    # Ensure valid return
    if best_inner is None:
        best_inner = [[0, 0, 0]] * N
        best_outer_center = (0, 0)
        best_outer_angle = 0.0
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)
    
    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
