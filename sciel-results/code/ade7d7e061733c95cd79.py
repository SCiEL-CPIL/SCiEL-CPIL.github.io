import numpy as np
import time
from scipy.spatial import ConvexHull

N = 12
SQRT3 = np.sqrt(3.0)

def _hexagon_verts(cx, cy, side=1.0, angle_rad=0.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angles = angle_rad + np.arange(0, 2*np.pi, np.pi/3)
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack([xs, ys])

def _hex_face_constraints(S, outer_angle_rad):
    """
    Return 6 half-space coefficients (a, b, c) such that a*x + b*y <= c
    for a regular hexagon centered at origin with side S and rotation outer_angle_rad.
    """
    cons = []
    for k in range(6):
        theta = outer_angle_rad + k * np.pi / 3
        # Outward normal direction
        a = np.cos(theta)
        b = np.sin(theta)
        # Distance from center to each side = apothem = S * sqrt(3) / 2
        c = S * SQRT3 / 2.0
        cons.append((a, b, c))
    return cons

def _max_violation(centers, sides, rotations, S, outer_angle_rad):
    """
    Return the maximum constraint violation over all inner hexagon vertices.
    A positive value means infeasible.
    """
    cons = _hex_face_constraints(S, outer_angle_rad)
    max_viol = 0.0
    for i in range(N):
        cx, cy = centers[i]
        side = sides[i]
        rot = rotations[i]
        verts = _hexagon_verts(cx, cy, side, rot)
        for (a, b, c) in cons:
            vals = a * verts[:, 0] + b * verts[:, 1] - c
            viol = np.max(vals)
            if viol > max_viol:
                max_viol = viol
    return max_viol

def _min_enclosing_hexagon(centers, sides, rotations):
    """
    For a fixed configuration of inner hexagons, find the minimum side length S
    and optimal rotation angle of the enclosing hexagon centered at origin.
    Uses the duality: the minimal enclosing hexagon of a set of points has sides
    that are support lines of the convex hull. We can solve for S and angle by
    finding the minimum over angle of the maximum over hull edges of the required S.
    
    Actually, a simpler approach: the minimal enclosing regular hexagon centered at origin
    with free rotation can be found by checking 6 candidate rotations based on the
    convex hull edges, or by optimizing over angle.
    
    For efficiency, we'll optimize over angle using a 1D search.
    """
    # Get all vertices of all inner hexagons
    all_verts = []
    for i in range(N):
        cx, cy = centers[i]
        side = sides[i]
        rot = rotations[i]
        verts = _hexagon_verts(cx, cy, side, rot)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # Convex hull of all vertices
    try:
        hull = ConvexHull(all_verts)
        hull_verts = all_verts[hull.vertices]
    except Exception:
        # Fallback: use all points
        hull_verts = all_verts
    
    def required_S(angle_rad):
        """For a given outer hexagon rotation, compute the required S."""
        cons = _hex_face_constraints(1.0, angle_rad)  # S=1, scale later
        max_c = 0.0
        for (a, b, c) in cons:
            # For each face, the required c' such that a*x + b*y <= c' for all hull points
            vals = a * hull_verts[:, 0] + b * hull_verts[:, 1]
            max_val = np.max(vals)
            # Since c = S * sqrt(3)/2 for unit S, we need c' = max_val
            # So S = 2 * max_val / sqrt(3)
            required = 2.0 * max_val / SQRT3
            if required > max_c:
                max_c = required
        return max_c
    
    # Optimize over angle
    best_S = float('inf')
    best_angle = 0.0
    
    # Coarse scan
    angles = np.linspace(0, np.pi/3, 61)
    for ang in angles:
        S_val = required_S(ang)
        if S_val < best_S:
            best_S = S_val
            best_angle = ang
    
    # Fine scan around best
    if best_S < float('inf'):
        fine_angles = np.linspace(best_angle - np.pi/60, best_angle + np.pi/60, 61)
        for ang in fine_angles:
            S_val = required_S(ang)
            if S_val < best_S:
                best_S = S_val
                best_angle = ang
    
    return best_S, best_angle

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s limit
    
    # Initial configuration: honeycomb-like arrangement
    # Center + 6 neighbors + 5 more
    centers = np.zeros((N, 2))
    sides = np.ones(N)
    rotations = np.zeros(N)
    
    centers[0] = [0.0, 0.0]
    dist1 = SQRT3  # Minimum distance between centers of non-overlapping unit hexagons
    for i in range(1, 7):
        ang = np.pi / 3 * (i - 1)
        centers[i] = [dist1 * np.cos(ang), dist1 * np.sin(ang)]
    
    # 5 more at distance 3, angles 30, 90, 150, 210, 270 degrees
    dist2 = 3.0
    angles2 = np.deg2rad([30, 90, 150, 210, 270])
    for i in range(7, 12):
        centers[i] = [dist2 * np.cos(angles2[i-7]), dist2 * np.sin(angles2[i-7])]
    
    # Get initial enclosing hexagon
    S_init, angle_init = _min_enclosing_hexagon(centers, sides, rotations)
    
    # Optimization: minimize S over positions, sides, and rotations
    # We'll use a penalty-based approach with scipy.optimize
    
    def _objective_and_penalty(x):
        """
        x = [x1, y1, s1, r1, x2, y2, s2, r2, ..., x12, y12, s12, r12]
        Returns (S, penalty)
        """
        centers = x[0::4][:, :2] if False else None
        # Reshape
        n = len(x) // 4
        centers = x[0:n*4:4].reshape(n, 1) if False else None
        # Better:
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        rotations = np.zeros(n)
        for i in range(n):
            centers[i, 0] = x[4*i]
            centers[i, 1] = x[4*i+1]
            sides[i] = x[4*i+2]
            rotations[i] = x[4*i+3]
        
        S, angle = _min_enclosing_hexagon(centers, sides, rotations)
        
        # Overlap penalty: for each pair, check if hexagons overlap
        # Use distance between centers vs minimum required distance
        # For two unit hexagons, minimum center distance is sqrt(3)
        # For arbitrary sides, the minimum distance depends on orientations
        # Approximate: require distance >= (s_i + s_j) * sqrt(3) / 2
        penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                min_d = (sides[i] + sides[j]) * SQRT3 / 2.0
                if d < min_d:
                    penalty += (min_d - d) ** 2
        return S, penalty
    
    # Variable vector
    x0 = np.zeros(N * 4)
    for i in range(N):
        x0[4*i] = centers[i, 0]
        x0[4*i+1] = centers[i, 1]
        x0[4*i+2] = sides[i]
        x0[4*i+3] = rotations[i]
    
    best_x = x0.copy()
    best_S, best_pen = _objective_and_penalty(x0)
    best_total = best_S + 100 * best_pen
    
    # Simple Nelder-Mead style or use scipy
    from scipy.optimize import minimize
    
    def objective_fn(x):
        S, pen = _objective_and_penalty(x)
        return S + 100 * pen
    
    # Bounds: positions in [-10, 10], sides in [0.5, 1.5], rotations in [-pi, pi]
    bounds = []
    for i in range(N):
        bounds.append((-10, 10))
        bounds.append((-10, 10))
        bounds.append((0.5, 1.5))
        bounds.append((-np.pi, np.pi))
    
    # Try L-BFGS-B
    try:
        res = minimize(objective_fn, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 200, 'ftol': 1e-10})
        if res.fun < best_total:
            best_x = res.x
            best_S, best_pen = _objective_and_penalty(res.x)
            best_total = res.fun
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        # Extract best configuration
        n = N
        final_centers = np.zeros((n, 2))
        final_sides = np.zeros(n)
        final_rotations = np.zeros(n)
        for i in range(n):
            final_centers[i, 0] = best_x[4*i]
            final_centers[i, 1] = best_x[4*i+1]
            final_sides[i] = best_x[4*i+2]
            final_rotations[i] = best_x[4*i+3]
        
        # Compute enclosing hexagon
        S_final, angle_final = _min_enclosing_hexagon(final_centers, final_sides, final_rotations)
        
        # Verify validity
        max_viol = _max_violation(final_centers, final_sides, final_rotations, S_final, angle_final)
        
        inner = []
        for i in range(n):
            inner.append([float(final_centers[i, 0]), float(final_centers[i, 1]), float(np.rad2deg(final_rotations[i]))])
        
        return inner, (0.0, 0.0), float(S_final), float(np.rad2deg(angle_final))
    
    # If we have more time, do a few more optimization runs with different initial guesses
    for restart in range(3):
        if time.time() > deadline:
            break
        
        # Perturb initial guess
        x_perturbed = x0.copy()
        np.random.seed(restart)
        for i in range(N):
            x_perturbed[4*i] += np.random.normal(0, 0.5)
            x_perturbed[4*i+1] += np.random.normal(0, 0.5)
        
        try:
            res = minimize(objective_fn, x_perturbed, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 100, 'ftol': 1e-10})
            if res.fun < best_total:
                best_x = res.x
                best_S, best_pen = _objective_and_penalty(res.x)
                best_total = res.fun
        except Exception:
            pass
    
    # Final extraction
    n = N
    final_centers = np.zeros((n, 2))
    final_sides = np.zeros(n)
    final_rotations = np.zeros(n)
    for i in range(n):
        final_centers[i, 0] = best_x[4*i]
        final_centers[i, 1] = best_x[4*i+1]
        final_sides[i] = best_x[4*i+2]
        final_rotations[i] = best_x[4*i+3]
    
    S_final, angle_final = _min_enclosing_hexagon(final_centers, final_sides, final_rotations)
    
    inner = []
    for i in range(n):
        inner.append([float(final_centers[i, 0]), float(final_centers[i, 1]), float(np.rad2deg(final_rotations[i]))])
    
    return inner, (0.0, 0.0), float(S_final), float(np.rad2deg(angle_final))
