import numpy as np
import scipy.spatial
import scipy.optimize
import time
from itertools import combinations

N = 14

def _get_triangle_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=np.int32)

_TRI_INDICES = _get_triangle_indices(N)
_I = _TRI_INDICES[:, 0]
_J = _TRI_INDICES[:, 1]
_K = _TRI_INDICES[:, 2]
_M = len(_TRI_INDICES)

def _min_triangle_area(points):
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross).min()

def _hull_area(points):
    try:
        hull = scipy.spatial.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _polygon_area(points):
    """Shoelace formula for polygon area (assumes CCW ordering)"""
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i, 0] * points[j, 1] - points[j, 0] * points[i, 1]
    return 0.5 * abs(area)

def _score(points):
    try:
        if len(points) < 3:
            return 0.0
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        mta = _min_triangle_area(points)
        return mta / ha
    except Exception:
        return 0.0

def _normalize(points):
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return points
        return points / np.sqrt(ha)
    except Exception:
        return points

def _regular_polygon(n, radius=1.0, rotation=0.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _angles_to_polygon(angles, scale=1.0):
    """
    Convert interior angles (summing to (n-2)*pi) to polygon vertices.
    Uses the turning angle approach: exterior angle at vertex i = pi - interior_angle[i]
    The polygon is built by walking around with these turning angles.
    Returns vertices in CCW order.
    """
    n = len(angles)
    # Exterior angles: pi - interior_angle
    ext_angles = np.pi - angles
    # The sum of exterior angles should be 2*pi for a closed convex polygon
    # Normalize to ensure closure
    ext_angles = ext_angles / np.sum(ext_angles) * 2 * np.pi
    
    # Build polygon by integrating direction changes
    # Start at origin, going in direction 0
    vertices = np.zeros((n, 2))
    direction = 0.0
    # We need to determine edge lengths. For a given set of exterior angles,
    # the shape is determined up to scale. We use unit edge lengths initially
    # and then scale.
    
    # Actually, for a convex polygon with given interior angles, the shape
    # (up to similarity) is determined. Let's use the approach of building
    # the polygon from the angles.
    
    # Alternative: use the fact that for a convex n-gon, the interior angles
    # determine the shape up to scale and rotation. We'll build it using
    # the turning angles.
    
    # Start with a point, and walk around
    vertices[0] = [0, 0]
    current_dir = 0.0
    # Edge lengths are determined by the angles for a closed polygon
    # For simplicity, use unit edges and then adjust
    
    # Actually, let's use a different approach: parameterize by the angles
    # and compute vertices using the constraint that the polygon closes.
    
    # For a convex polygon with exterior angles theta_i (sum = 2*pi),
    # the vertices can be computed. Let's use a numerical approach.
    
    # Simple approach: start at origin, first edge in direction 0
    # Then turn by each exterior angle
    # Edge lengths are all 1 initially, then we'll scale
    
    pos = np.array([0.0, 0.0])
    vertices[0] = pos
    for i in range(n - 1):
        # Move in current direction
        pos = pos + np.array([np.cos(current_dir), np.sin(current_dir)])
        vertices[i + 1] = pos
        # Turn
        current_dir += ext_angles[i]
    
    # The polygon may not close exactly with unit edges. 
    # For the purpose of optimization, we'll work with the shape
    # and normalize later.
    
    return vertices

def _polygon_from_angles(angles):
    """
    Given interior angles of a convex n-gon (sum = (n-2)*pi),
    compute the vertices of the polygon.
    Uses the method of building from exterior angles with computed edge lengths.
    """
    n = len(angles)
    # Exterior angles
    ext = np.pi - angles
    # Normalize so sum = 2*pi (should already be, but for safety)
    ext = ext / np.sum(ext) * 2 * np.pi
    
    # Build the polygon: we need to find edge lengths such that the polygon closes.
    # For a convex polygon with given exterior angles, the edge lengths are
    # determined up to a scale factor. We can compute them using the closure constraint.
    
    # Direction of each edge
    dirs = np.zeros(n)
    dirs[0] = 0.0
    for i in range(1, n):
        dirs[i] = dirs[i-1] + ext[i-1]
    
    # The closure condition: sum of (length_i * direction_i) = 0
    # This is a 2x n system. For a convex polygon, there's a 1-parameter family
    # of solutions. We can fix one edge length and solve.
    
    # Actually, for a given set of exterior angles, the shape is determined up to
    # similarity. We can compute it by choosing a convenient parameterization.
    
    # Use the fact that the vertices lie on a line if we project appropriately.
    # Simpler: use the "support function" or just compute numerically.
    
    # Let's use a direct numerical approach: 
    # Place vertex 0 at origin, edge 0 along x-axis.
    # Then compute subsequent vertices, and adjust the last edge to close.
    
    # For optimization purposes, we'll use a simpler parameterization:
    # We'll represent the polygon by its vertices directly, but constrain
    # them to be in convex position.
    
    # Actually, let's just return a regular-ish polygon based on the angles
    # for now, and do the optimization in vertex space with convexity constraints.
    
    # Better approach: use the angles to define the normal directions,
    # then compute vertices as intersections of consecutive edge lines.
    
    # Edge normals (pointing inward for CCW polygon)
    # If edge i has direction dirs[i], its inward normal is (sin(dirs[i]), -cos(dirs[i]))
    # for a CCW polygon... actually let me think about this more carefully.
    
    # For a CCW polygon, the interior is to the left of each edge.
    # Edge i goes from vertex i to vertex i+1.
    # The direction of edge i is dirs[i].
    # The inward normal (pointing into the polygon) is (-sin(dirs[i]), cos(dirs[i]))
    
    # Let's use a different, cleaner approach:
    # Given the exterior angles, the polygon's shape is determined.
    # We can compute it by starting with an arbitrary point and direction,
    # and using the fact that for a closed polygon, the edge lengths satisfy
    # a linear system.
    
    # For simplicity in this optimization, let's use a parameterization where
    # we fix the first vertex and first edge, and compute the rest.
    
    # Actually, the cleanest approach for optimization is to work directly
    # with vertex coordinates and use a smooth penalty for non-convexity,
    # or to use the angle representation with a proper closure solver.
    
    # Let me use a practical approach: compute the polygon from angles
    # using the method of "folding" - start with unit edges and project
    # to make it close.
    
    # For now, return a placeholder that we'll replace with a proper solver
    # in the optimization loop.
    
    # Actually, let's use a completely different and cleaner approach:
    # Represent the convex polygon by its support function or by the
    # angles of its edges, and compute vertices via intersection.
    
    # Edge directions (tangent directions)
    # For a convex CCW polygon, the edge directions increase by the exterior angles.
    # Let edge_i have direction phi_i.
    # phi_0 = 0, phi_{i+1} = phi_i + ext[i]
    
    # The polygon is the intersection of half-planes.
    # Edge i: the line through some point with direction phi_i, with the polygon on the left.
    
    # To compute vertices: vertex i is the intersection of edge i-1 and edge i.
    # We need to determine the offset of each edge from the origin.
    
    # For a convex polygon, the offsets are determined up to a translation.
    # We can fix the offset of edge 0 to be 0 (passing through origin).
    
    # Actually, this is getting complex. Let me use a simpler but effective approach:
    # Optimize directly in vertex space with a smooth convexity constraint.
    
    raise NotImplementedError("Use vertex-space optimization instead")

def _smooth_min_triangle_area(points, beta=100.0):
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    max_area = np.max(areas)
    if max_area < 1e-15:
        return 0.0
    
    shifted = areas - max_area
    exp_vals = np.exp(-beta * shifted)
    log_sum = -beta * max_area + np.log(np.sum(exp_vals))
    smooth_min = -log_sum / beta
    return smooth_min

def _neg_smooth_score_and_grad(points_flat, beta=100.0):
    """
    Returns (-smooth_score, gradient) for L-BFGS-B.
    Uses analytic gradient of the log-sum-exp soft-min.
    Assumes all points are in convex position (on the hull).
    """
    points = points_flat.reshape(-1, 2)
    
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    max_area = np.max(areas)
    if max_area < 1e-15:
        return 0.0, np.zeros_like(points_flat)

    shifted = areas - max_area
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    if sum_exp == 0:
        return 0.0, np.zeros_like(points_flat)
    weights = exp_vals / sum_exp
    
    log_sum = -beta * max_area + np.log(sum_exp)
    smooth_min = -log_sum / beta
    
    # Use polygon area (assuming convex position)
    ha = _polygon_area(points)
    if ha < 1e-12:
        return 0.0, np.zeros_like(points_flat)
    
    score = smooth_min / ha
    
    signs = np.sign(cross)
    
    d_cross_di_x = -(v1[:, 1] + v2[:, 1])
    d_cross_di_y = (v1[:, 0] + v2[:, 0])
    d_cross_dj_x = v2[:, 1]
    d_cross_dj_y = -v2[:, 0]
    d_cross_dk_x = -v1[:, 1]
    d_cross_dk_y = v1[:, 0]
    
    w = weights * signs * 0.5
    
    grad_points = np.zeros_like(points)
    
    dA_di = w[:, None] * np.column_stack([d_cross_di_x, d_cross_di_y])
    dA_dj = w[:, None] * np.column_stack([d_cross_dj_x, d_cross_dj_y])
    dA_dk = w[:, None] * np.column_stack([d_cross_dk_x, d_cross_dk_y])
    
    np.add.at(grad_points, _I, dA_di)
    np.add.at(grad_points, _J, dA_dj)
    np.add.at(grad_points, _K, dA_dk)
    
    # Gradient of score = smooth_min / ha
    # d(score)/d(points) = (grad_smooth_min * ha - smooth_min * grad_ha) / ha^2
    # For simplicity, approximate as grad_smooth_min / ha
    # This is valid when the hull is stable
    
    # Compute gradient of polygon area for better accuracy
    # Polygon area = 0.5 * sum(x_i * y_{i+1} - x_{i+1} * y_i)
    n = len(points)
    grad_ha = np.zeros_like(points)
    for i in range(n):
        j = (i + 1) % n
        # d(area)/d(x_i) = 0.5 * (y_{i+1} - y_{i-1})
        # d(area)/d(y_i) = 0.5 * (x_{i-1} - x_{i+1})
        ip = (i - 1) % n
        grad_ha[i, 0] = 0.5 * (points[j, 1] - points[ip, 1])
        grad_ha[i, 1] = 0.5 * (points[ip, 0] - points[j, 0])
    
    # Full gradient: (grad_sm * ha - sm * grad_ha) / ha^2
    grad_score = (grad_points * ha - smooth_min * grad_ha) / (ha * ha)
    
    return -score, -grad_score.flatten()

def _is_convex(points):
    """Check if points are in convex position (CCW, all interior angles < pi)"""
    n = len(points)
    for i in range(n):
        p0 = points[i]
        p1 = points[(i + 1) % n]
        p2 = points[(i + 2) % n]
        cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
        if cross <= 1e-10:
            return False
    return True

def _convexity_penalty(points, beta=50.0):
    """
    Smooth penalty for non-convexity.
    For a convex CCW polygon, all consecutive triple cross products should be positive.
    We penalize negative cross products.
    """
    n = len(points)
    penalty = 0.0
    for i in range(n):
        p0 = points[i]
        p1 = points[(i + 1) % n]
        p2 = points[(i + 2) % n]
        cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
        if cross < 0:
            penalty += -cross * beta
    return penalty

def _neg_smooth_score_with_convexity(points_flat, beta=100.0, convex_beta=50.0):
    """
    Negative smooth score plus convexity penalty.
    """
    points = points_flat.reshape(-1, 2)
    
    # Compute smooth score
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    max_area = np.max(areas)
    if max_area < 1e-15:
        return 1e6, np.zeros_like(points_flat)

    shifted = areas - max_area
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    if sum_exp == 0:
        return 1e6, np.zeros_like(points_flat)
    weights = exp_vals / sum_exp
    
    log_sum = -beta * max_area + np.log(sum_exp)
    smooth_min = -log_sum / beta
    
    ha = _polygon_area(points)
    if ha < 1e-12:
        return 1e6, np.zeros_like(points_flat)
    
    score = smooth_min / ha
    
    # Convexity penalty
    n = len(points)
    convex_penalty = 0.0
    grad_convex = np.zeros_like(points)
    for i in range(n):
        p0 = points[i]
        p1 = points[(i + 1) % n]
        p2 = points[(i + 2) % n]
        cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
        if cross < 0:
            convex_penalty += -cross * convex_beta
            # Gradient of -cross * convex_beta with respect to points
            # cross = (p1x-p0x)(p2y-p1y) - (p1y-p0y)(p2x-p1x)
            # d(cross)/d(p0) = (-1, 1) * (p2 - p1) ... let me compute properly
            # d(cross)/d(p0x) = -(p2y-p1y)
            # d(cross)/d(p0y) = (p2x-p1x)
            # d(cross)/d(p1x) = (p2y-p1y) - (p1y-p0y) = p2y - p0y
            # d(cross)/d(p1y) = -(p2x-p1x) + (p1x-p0x) = p0x - p2x
            # d(cross)/d(p2x) = -(p1y-p0y)
            # d(cross)/d(p2y) = (p1x-p0x)
            
            dpx0 = -(p2[1] - p1[1])
            dpy0 = (p2[0] - p1[0])
            dpx1 = (p2[1] - p0[1])
            dpy1 = (p0[0] - p2[0])
            dpx2 = -(p1[1] - p0[1])
            dpy2 = (p1[0] - p0[0])
            
            # We're penalizing -cross, so gradient of penalty is -convex_beta * d(cross)/d(point)
            grad_convex[i] += -convex_beta * np.array([dpx0, dpy0])
            grad_convex[(i + 1) % n] += -convex_beta * np.array([dpx1, dpy1])
            grad_convex[(i + 2) % n] += -convex_beta * np.array([dpx2, dpy2])
    
    total_penalty = convex_penalty
    
    # Score part gradient
    signs = np.sign(cross)
    d_cross_di_x = -(v1[:, 1] + v2[:, 1])
    d_cross_di_y = (v1[:, 0] + v2[:, 0])
    d_cross_dj_x = v2[:, 1]
    d_cross_dj_y = -v2[:, 0]
    d_cross_dk_x = -v1[:, 1]
    d_cross_dk_y = v1[:, 0]
    
    w = weights * signs * 0.5
    
    grad_points = np.zeros_like(points)
    dA_di = w[:, None] * np.column_stack([d_cross_di_x, d_cross_di_y])
    dA_dj = w[:, None] * np.column_stack([d_cross_dj_x, d_cross_dj_y])
    dA_dk = w[:, None] * np.column_stack([d_cross_dk_x, d_cross_dk_y])
    
    np.add.at(grad_points, _I, dA_di)
    np.add.at(grad_points, _J, dA_dj)
    np.add.at(grad_points, _K, dA_dk)
    
    grad_ha = np.zeros_like(points)
    for i in range(n):
        j = (i + 1) % n
        ip = (i - 1) % n
        grad_ha[i, 0] = 0.5 * (points[j, 1] - points[ip, 1])
        grad_ha[i, 1] = 0.5 * (points[ip, 0] - points[j, 0])
    
    grad_score = (grad_points * ha - smooth_min * grad_ha) / (ha * ha)
    
    total_obj = -score + total_penalty
    total_grad = -grad_score + grad_convex
    
    return total_obj, total_grad.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = 0.0
    best_points = None

    def _update_best(points):
        nonlocal best_score, best_points
        pts = _normalize(points)
        sc = _score(pts)
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()
        return sc

    def _try_optimize_smooth(init_points, beta_start=30.0, beta_end=300.0, n_stages=4, maxiter_per_stage=600, convex_beta=50.0):
        if time.time() > deadline - 10:
            return
        init_norm = _normalize(init_points)
        x0 = init_norm.flatten()
        
        betas = np.linspace(beta_start, beta_end, n_stages)
        
        x = x0.copy()
        for beta in betas:
            if time.time() > deadline - 10:
                break
            
            def _obj_and_grad(x_flat, beta_val=beta):
                return _neg_smooth_score_with_convexity(x_flat, beta_val, convex_beta)
            
            try:
                res = scipy.optimize.minimize(
                    _obj_and_grad,
                    x,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': maxiter_per_stage, 'gtol': 1e-10, 'ftol': 1e-14}
                )
                x = res.x
                _update_best(x.reshape(-1, 2))
            except Exception:
                pass
        
        if time.time() < deadline - 5:
            try:
                def _obj_and_grad_high_beta(x_flat):
                    return _neg_smooth_score_with_convexity(x_flat, 800.0, 100.0)
                
                res = scipy.optimize.minimize(
                    _obj_and_grad_high_beta,
                    x,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'gtol': 1e-11, 'ftol': 1e-15}
                )
                x = res.x
                _update_best(x.reshape(-1, 2))
            except Exception:
                pass

    # Strategy 1: Regular polygon as starting point
    reg = _regular_polygon(N, 1.0)
    _update_best(reg)
    _try_optimize_smooth(reg, beta_start=30.0, beta_end=300.0, n_stages=4, maxiter_per_stage=600)
    
    # Strategy 2: Perturbed regular polygons
    rng = np.random.RandomState(42)
    for i in range(15):
        if time.time() > deadline - 20:
            break
        noise = rng.uniform(-0.05, 0.05, (N, 2))
        init = _regular_polygon(N, 1.0) + noise
        # Ensure convexity by projecting if needed
        if not _is_convex(init):
            # Project to convex position by keeping the convex hull vertices
            try:
                hull = scipy.spatial.ConvexHull(init)
                hull_pts = init[hull.vertices]
                if len(hull_pts) == N:
                    init = hull_pts
                else:
                    init = reg  # Fall back to regular
            except Exception:
                init = reg
        _try_optimize_smooth(init, beta_start=25.0, beta_end=250.0, n_stages=3, maxiter_per_stage=500)
    
    # Strategy 3: Two-ring inspired configurations (but ensure convexity)
    splits = [(7, 7), (6, 8), (8, 6), (5, 9), (9, 5)]
    for n1, n2 in splits:
        if time.time() > deadline - 15:
            break
        for r_ratio in [0.5, 0.6, 0.7, 0.8]:
            for rot in [0.0, np.pi / N]:
                if time.time() > deadline - 12:
                    continue
                pts1 = _regular_polygon(n1, 1.0, 0.0)
                pts2 = _regular_polygon(n2, r_ratio, rot)
                init = np.vstack([pts1, pts2])
                # Sort by angle to ensure proper polygon ordering
                angles = np.arctan2(init[:, 1], init[:, 0])
                idx = np.argsort(angles)
                init = init[idx]
                _try_optimize_smooth(init, beta_start=25.0, beta_end=200.0, n_stages=3, maxiter_per_stage=400)
    
    # Strategy 4: Random convex polygons
    for i in range(8):
        if time.time() > deadline - 10:
            break
        # Generate random convex polygon
        angles = np.sort(rng.uniform(0, 2 * np.pi, N))
        radii = rng.uniform(0.7, 1.0, N)
        init = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        _try_optimize_smooth(init, beta_start=20.0, beta_end=200.0, n_stages=3, maxiter_per_stage=400)
    
    # Final check
    if best_points is None:
        best_points = _normalize(_regular_polygon(N, 1.0))
    else:
        best_points = _normalize(best_points)
        
    return best_points
