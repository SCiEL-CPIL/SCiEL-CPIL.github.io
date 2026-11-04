import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar, linprog

N = 11
DEADLINE = 120.0
SAFETY_MARGIN = 5.0

def _hex_vertices(center, side, rotation):
    """Return vertices of a regular hexagon with given center, side length, and rotation (radians)."""
    cx, cy = center
    angles = rotation + np.arange(6) * (math.pi / 3)
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _convex_hull(points):
    """Compute convex hull of 2D points using Andrew's monotone chain."""
    points = np.array(points, dtype=float)
    points = np.unique(points, axis=0)
    n = len(points)
    if n <= 2:
        return points
    points = points[np.lexsort((points[:, 1], points[:, 0]))]
    lower = []
    for p in points:
        while len(lower) >= 2:
            cross = (lower[-1][0] - lower[-2][0]) * (p[1] - lower[-2][1]) - \
                    (lower[-1][1] - lower[-2][1]) * (p[0] - lower[-2][0])
            if cross <= 1e-12:
                lower.pop()
            else:
                break
        lower.append(p)
    upper = []
    for p in points[::-1]:
        while len(upper) >= 2:
            cross = (upper[-1][0] - upper[-2][0]) * (p[1] - upper[-2][1]) - \
                    (upper[-1][1] - upper[-2][1]) * (p[0] - upper[-2][0])
            if cross <= 1e-12:
                upper.pop()
            else:
                break
        upper.append(p)
    return np.vstack([lower[:-1], upper[:-1]])

def _compute_min_side_from_hull(hull, theta):
    """Compute minimal side length of outer hexagon (with rotation theta) enclosing the hull.
    The outer hexagon has 6 sides; we project hull vertices onto 6 normals and find the 
    minimal R such that all projections fit in [-R, R]. Then S = 2*R/sqrt(3).
    Actually, for a regular hexagon with side S, the inradius is S*sqrt(3)/2.
    So R = S*sqrt(3)/2, meaning S = 2*R/sqrt(3).
    But we need the minimal S such that there exists a center C where all hull points
    project into [-R, R] along each of the 6 normal directions.
    This is equivalent to: for each direction k, the range of projections must be <= 2R.
    And R must be at least half the maximum range.
    But the center C can be chosen to minimize the maximum absolute deviation.
    For a single direction, the optimal center projection is the midpoint of [min, max].
    The constraint is that R >= (max_k - min_k)/2 for all k.
    So R_min = max_k (max_k - min_k)/2.
    And S = 2*R_min/sqrt(3) = max_k (max_k - min_k)/sqrt(3).
    """
    angles = theta + np.arange(6) * (math.pi / 3)
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    projs = hull @ normals.T
    widths = np.max(projs, axis=0) - np.min(projs, axis=0)
    max_width = np.max(widths)
    return max_width / math.sqrt(3)

def _get_all_vertices(centers, rotations):
    """Get all 6*N vertices from N hexagons with given centers and rotations."""
    verts = []
    for i in range(len(centers)):
        cx, cy = centers[i]
        v = _hex_vertices((cx, cy), 1.0, rotations[i])
        verts.append(v)
    return np.vstack(verts)

def _objective(params, centers_base, rotations_base, hull_cache=None):
    """Objective: minimize the outer hexagon side length.
    params: [theta_outer] or extended parameters.
    For now, we fix the inner arrangement and only optimize the outer rotation theta.
    """
    theta = params[0]
    # Get hull of all inner vertices
    # Since centers and rotations are fixed, the hull is fixed
    # We just need to compute min side for this theta
    if hull_cache is not None:
        hull = hull_cache
    else:
        all_verts = _get_all_vertices(centers_base, rotations_base)
        hull = _convex_hull(all_verts)
    return _compute_min_side_from_hull(hull, theta)

def _seed_hexagonal_lattice():
    """Generate the standard hexagonal lattice arrangement of 11 hexagons."""
    d = math.sqrt(3)
    centers = [(0.0, 0.0)]
    for i in range(6):
        angle = i * math.pi / 3
        centers.append((d * math.cos(angle), d * math.sin(angle)))
    ring2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = d * (q + r / 2.0)
                y = d * (math.sqrt(3) / 2.0) * r
                ring2.append((x, y))
    # We need 11 total, so center + 6 ring1 + 4 from ring2
    # Pick 4 ring2 centers that give the best packing
    best_combo = None
    best_S = float('inf')
    import itertools
    for combo in itertools.combinations(range(12), 4):
        cur_centers = centers[:7] + [ring2[i] for i in combo]
        cur_rotations = [0.0] * 11
        all_verts = _get_all_vertices(cur_centers, cur_rotations)
        hull = _convex_hull(all_verts)
        # Try a few thetas
        for theta in np.linspace(0, math.pi / 3, 20):
            S = _compute_min_side_from_hull(hull, theta)
            if S < best_S:
                best_S = S
                best_combo = combo
    # Return the best arrangement
    final_centers = centers[:7] + [ring2[i] for i in best_combo]
    final_rotations = [0.0] * 11
    return final_centers, final_rotations

def _try_rotated_arrangement():
    """Try arrangements where some hexagons are rotated relative to neighbors."""
    d = math.sqrt(3)
    base_centers = [(0.0, 0.0)]
    for i in range(6):
        angle = i * math.pi / 3
        base_centers.append((d * math.cos(angle), d * math.sin(angle)))
    ring2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = d * (q + r / 2.0)
                y = d * (math.sqrt(3) / 2.0) * r
                ring2.append((x, y))
    
    best_result = None
    best_S = float('inf')
    
    # Try rotating the 4 ring-2 hexagons by various angles
    import itertools
    angle_options = [0.0, math.pi / 6, -math.pi / 6, math.pi / 3, -math.pi / 3]
    
    for combo in itertools.combinations(range(12), 4):
        for rot_combo in itertools.product(angle_options, repeat=4):
            centers = base_centers[:7] + [ring2[i] for i in combo]
            rotations = [0.0] * 7 + list(rot_combo)
            all_verts = _get_all_vertices(centers, rotations)
            hull = _convex_hull(all_verts)
            for theta in np.linspace(0, math.pi / 3, 10):
                S = _compute_min_side_from_hull(hull, theta)
                if S < best_S:
                    best_S = S
                    best_result = (centers, rotations, theta)
    
    return best_result

def construct():
    start_time = time.time()
    
    # Generate initial candidate arrangements
    candidates = []
    
    # Candidate 1: Standard hexagonal lattice
    centers1, rotations1 = _seed_hexagonal_lattice()
    candidates.append((centers1, rotations1))
    
    # Candidate 2: Try rotated arrangements
    if time.time() - start_time < DEADLINE - SAFETY_MARGIN:
        result2 = _try_rotated_arrangement()
        if result2 is not None:
            candidates.append((result2[0], result2[1]))
    
    # For each candidate, optimize the outer hexagon rotation theta
    best_S = float('inf')
    best_centers = None
    best_rotations = None
    best_theta = 0.0
    
    for centers, rotations in candidates:
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        all_verts = _get_all_vertices(centers, rotations)
        hull = _convex_hull(all_verts)
        
        # Optimize theta
        def obj(theta):
            return _compute_min_side_from_hull(hull, theta)
        
        # Multi-start optimization
        best_local_S = float('inf')
        best_local_theta = 0.0
        
        # Coarse sweep
        thetas = np.linspace(0, math.pi / 3, 100)
        for t in thetas:
            S = obj(t)
            if S < best_local_S:
                best_local_S = S
                best_local_theta = t
        
        # Fine refinement
        try:
            res = minimize_scalar(obj, bounds=(0, math.pi / 3), method='bounded', 
                                  options={'xatol': 1e-10})
            if res.success and res.fun < best_local_S:
                best_local_S = res.fun
                best_local_theta = res.x
        except:
            pass
        
        if best_local_S < best_S:
            best_S = best_local_S
            best_centers = centers
            best_rotations = rotations
            best_theta = best_local_theta
    
    # Fallback if nothing was found
    if best_centers is None:
        # Return a trivial valid packing (spread out)
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Compute the optimal center of the outer hexagon
    # The center should be the midpoint of the projections for the tightest direction
    all_verts = _get_all_vertices(best_centers, best_rotations)
    hull = _convex_hull(all_verts)
    angles = best_theta + np.arange(6) * (math.pi / 3)
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    projs = hull @ normals.T
    
    # For each direction, the optimal center projection is the midpoint
    # We want to find a center (cx, cy) that minimizes the maximum deviation
    # This is a linear programming problem
    # Minimize R
    # Subject to: |proj_k - c . n_k| <= R for all k, all hull points
    
    # Actually, let's just use the centroid of the hull as a good approximation
    cx = np.mean(hull[:, 0])
    cy = np.mean(hull[:, 1])
    
    # Verify and compute the actual required S
    # The required R is max over all hull points and all directions of |p . n - c . n|
    all_projs = all_verts @ normals.T
    center_projs = np.array([cx * nx + cy * ny for nx, ny in normals])
    deviations = np.abs(all_projs - center_projs[np.newaxis, :])
    R_required = np.max(deviations)
    S_final = 2.0 * R_required / math.sqrt(3)
    
    # If S_final is larger than best_S, adjust center
    # Try to find a better center using LP
    if time.time() - start_time < DEADLINE - SAFETY_MARGIN:
        try:
            c_vec = np.array([0.0, 0.0, 1.0])
            A_ub = []
            b_ub = []
            for k in range(6):
                nx, ny = normals[k]
                for p in all_verts:
                    px, py = p
                    proj = px * nx + py * ny
                    A_ub.append([nx, ny, 1.0])
                    b_ub.append(proj)
                    A_ub.append([-nx, -ny, 1.0])
                    b_ub.append(-proj)
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            bounds = [(None, None), (None, None), (0, None)]
            res = linprog(c_vec, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                cx, cy, R_opt = res.x
                S_final = 2.0 * R_opt / math.sqrt(3)
        except:
            pass
    
    # Format output
    inner_rows = []
    for i in range(len(best_centers)):
        cx_i, cy_i = best_centers[i]
        rot_deg = math.degrees(best_rotations[i])
        inner_rows.append([float(cx_i), float(cy_i), float(rot_deg)])
    
    center = (float(cx), float(cy))
    side = float(S_final)
    angle_deg = float(math.degrees(best_theta))
    
    return inner_rows, center, side, angle_deg
