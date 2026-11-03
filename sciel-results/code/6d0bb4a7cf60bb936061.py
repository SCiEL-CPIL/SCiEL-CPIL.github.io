"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.optimize import minimize, LinearConstraint

N = 12
START_TIME = time.time()
TIME_LIMIT = 140.0  # Leave 10s margin before 150s kill

def hexagon_vertices(cx, cy, side, angle_deg):
    """
    Return vertices of a regular hexagon centered at (cx, cy) with given side length
    and rotation (in degrees).
    Vertices are ordered counter-clockwise starting from the rightmost point when angle=0.
    """
    angle_rad = np.deg2rad(angle_deg)
    angles = angle_rad + np.deg2rad(np.array([0, 60, 120, 180, 240, 300]))
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack((xs, ys))

def convex_hull(points):
    """
    Compute the convex hull of a set of 2D points using the monotone chain algorithm.
    Returns the hull vertices in counter-clockwise order.
    """
    points = np.array(points)
    points = np.unique(points, axis=0)
    if len(points) <= 2:
        return points
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    points = sorted(map(list, points), key=lambda p: (p[0], p[1]))
    
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    hull = lower[:-1] + upper[:-1]
    return np.array(hull)

def min_enclosing_hexagon(vertices):
    """
    Given a set of points (vertices of inner hexagons), find the smallest regular hexagon
    that contains all points. We minimize the side length of the enclosing hexagon.
    The hexagon can be rotated and translated.
    
    We use the fact that the minimum enclosing regular hexagon can be found by considering
    the support functions. For a given rotation angle, we can find the optimal translation
    and side length analytically.
    """
    best_side = float('inf')
    best_center = (0, 0)
    best_angle = 0
    
    # The optimal angle is likely to be related to the orientation of the points.
    # We'll sample a range of angles.
    # For a hexagon, the symmetry is 60 degrees, so we only need to search [0, 60).
    
    def side_for_angle(theta_deg):
        theta_rad = np.deg2rad(theta_deg)
        # The normal directions for the hexagon sides
        # A regular hexagon with side s and rotation theta has support function:
        # h(u) = s * max_{k=0..5} cos(u - theta - k*pi/3)
        # Actually, for a hexagon centered at origin with rotation theta, the support in direction u is:
        # s * max_{k} cos(u - theta - k*pi/3)
        
        # We need to find s and center (cx, cy) such that all points are inside.
        # Point (x, y) is inside hexagon centered at (cx, cy) with side s and rotation theta iff:
        # For all k=0..5: (x-cx, y-cy) dot n_k <= s * apothem
        # where n_k are the outward normals, and apothem = s * sqrt(3)/2
        
        # Let's use the constraint:
        # max_k ( (x-cx)*n_kx + (y-cy)*n_ky ) <= s * apothem
        
        # Let d_k = (x-cx)*n_kx + (y-cy)*n_ky
        # We need max_k d_k <= s * apothem for all points.
        # So s >= max_k d_k / apothem for all points.
        # s >= max_{points, k} d_k / apothem
        
        # But d_k depends on cx, cy.
        # d_k = x*n_kx + y*n_ky - cx*n_kx - cy*n_ky
        # Let v_k = (n_kx, n_ky)
        # d_k = dot(p, v_k) - dot(c, v_k)
        
        # We want to minimize s = max_{p, k} (dot(p, v_k) - dot(c, v_k)) / apothem
        # subject to also the "lower" constraints? No, the hexagon is convex, so if all points
        # satisfy the half-plane constraints for all 6 sides, they are inside.
        
        # Actually, the condition for a point to be inside a convex polygon is that it satisfies
        # all the half-plane inequalities. For a regular hexagon, these are the 6 inequalities.
        
        # So we need:
        # For all points p and all k: dot(p, v_k) - dot(c, v_k) <= s * apothem
        # And also: dot(p, -v_k) - dot(c, -v_k) <= s * apothem  (which is the opposite side)
        # Wait, the 6 normals are 120 degrees apart? No, 60 degrees apart.
        # The normals are at angles theta, theta+60, ..., theta+300.
        # The opposite normal is at theta+180, which is one of the 6.
        
        # So the constraints are:
        # For all p, k: dot(p, v_k) - dot(c, v_k) <= s * apothem
        # where v_k = (cos(theta + k*pi/3), sin(theta + k*pi/3)) for k=0..5
        
        # This is equivalent to:
        # max_k (dot(p, v_k)) - max_k (dot(c, v_k)) <= s * apothem  (for each p? No)
        
        # Actually, for a fixed c, the condition for point p is:
        # max_k (dot(p - c, v_k)) <= s * apothem
        
        # So s >= max_{p, k} dot(p - c, v_k) / apothem
        
        # We want to minimize s over c.
        # s(c) = max_{p, k} (dot(p, v_k) - dot(c, v_k)) / apothem
        
        # Let A_k = max_p dot(p, v_k)
        # Then s(c) = max_k (A_k - dot(c, v_k)) / apothem
        
        # We want to minimize max_k (A_k - dot(c, v_k)).
        # This is a convex optimization problem in c = (cx, cy).
        
        # Let f(c) = max_k (A_k - dot(c, v_k))
        # We want to minimize f(c).
        
        # This is the minimum of the maximum of linear functions, which is a convex problem.
        # It can be solved by linear programming or by finding the point where the
        # "tightest" constraints intersect.
        
        # Alternatively, we can use the fact that the optimal c is the center of the
        # minimum enclosing hexagon, which can be found by considering the vertices
        # of the convex hull of the points.
        
        # For now, let's use a simple approach: the optimal c is such that the
        # maximum over k is minimized. This is equivalent to finding the center of
        # the smallest hexagon (with fixed orientation) enclosing the points.
        
        # For a fixed orientation, the problem reduces to finding the smallest
        # axis-aligned (in the rotated frame) hexagon enclosing the points.
        
        # Let's transform the points to the rotated frame.
        # In the rotated frame, the hexagon is axis-aligned (normals along axes).
        # The constraints are:
        # For k=0..5: dot(p_rot, v_k_rot) <= s * apothem + dot(c_rot, v_k_rot)
        # where v_k_rot are the normals in the rotated frame.
        
        # Actually, it's easier to work in the original frame.
        
        # Let's use scipy.optimize to solve for c that minimizes f(c).
        
        v = np.array([[np.cos(theta_rad + k * np.pi / 3), np.sin(theta_rad + k * np.pi / 3)] for k in range(6)])
        # v is 6x2
        
        # A_k = max_p dot(p, v_k)
        # points is Nx2, v is 6x2
        # dot(p, v_k) for all p, k is points @ v.T, which is Nx6
        dots = vertices @ v.T  # Nx6
        A = np.max(dots, axis=0)  # 6
        
        # f(c) = max_k (A_k - c @ v_k)
        # We want to minimize f(c) over c = (cx, cy)
        
        def objective(c):
            cx, cy = c
            vals = A - (cx * v[:, 0] + cy * v[:, 1])
            return np.max(vals)
        
        # Use Nelder-Mead to minimize f(c)
        # Initial guess: centroid of the points
        c0 = np.mean(vertices, axis=0)
        
        res = minimize(objective, c0, method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000})
        
        c_opt = res.x
        f_opt = res.fun
        apothem = np.sqrt(3) / 2
        s = f_opt / apothem
        
        return s, c_opt
    
    # Sample angles
    best_s = float('inf')
    best_c = None
    best_theta = 0
    
    # First, do a coarse search
    for theta_deg in np.linspace(0, 60, 61):
        s, c = side_for_angle(theta_deg)
        if s < best_s:
            best_s = s
            best_c = c
            best_theta = theta_deg
    
    # Refine around the best angle
    if best_c is not None:
        for theta_deg in np.linspace(best_theta - 2, best_theta + 2, 41):
            s, c = side_for_angle(theta_deg)
            if s < best_s:
                best_s = s
                best_c = c
                best_theta = theta_deg
    
    return best_s, best_c, best_theta

def construct():
    global START_TIME
    if time.time() - START_TIME > TIME_LIMIT:
        # Return a trivial solution
        inner = [[0.0, 0.0, 0.0]] * N
        return inner, (0.0, 0.0), 100.0, 0.0
    
    random.seed(42)
    np.random.seed(42)
    
    # Initial placement: hexagonal lattice
    # Side length 1 hexagon has width 2, height sqrt(3)
    # In a hexagonal lattice, the distance between centers is 2 for horizontal,
    # and sqrt(3) for vertical (with offset).
    
    # Let's place them in a 4x3 grid or similar.
    # For 12 hexagons, a 4x3 grid works.
    
    inner = []
    # Hexagonal lattice positions
    dx = 2.0  # horizontal distance
    dy = np.sqrt(3)  # vertical distance
    
    # 4 columns, 3 rows
    for row in range(3):
        for col in range(4):
            x = col * dx + (0.5 * dx if row % 2 == 1 else 0)
            y = row * dy
            inner.append([x, y, 0.0])
    
    # Compute initial enclosing hexagon
    all_vertices = []
    for x, y, angle in inner:
        verts = hexagon_vertices(x, y, 1.0, angle)
        all_vertices.extend(verts.tolist())
    
    all_vertices = np.array(all_vertices)
    side, center, angle = min_enclosing_hexagon(all_vertices)
    
    best_side = side
    best_inner = inner
    best_center = center
    best_angle = angle
    
    # Local optimization: perturb positions and angles
    # We'll use a simple gradient-free method: perturb each parameter slightly
    # and keep the change if it improves the objective.
    
    def evaluate(inner):
        all_vertices = []
        for x, y, angle in inner:
            verts = hexagon_vertices(x, y, 1.0, angle)
            all_vertices.extend(verts.tolist())
        all_vertices = np.array(all_vertices)
        s, c, a = min_enclosing_hexagon(all_vertices)
        return s, c, a
    
    # Iterative improvement
    step_size = 0.1
    max_iter = 100
    
    for it in range(max_iter):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        
        improved = False
        for i in range(N):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            
            # Try perturbing x, y, angle
            for param_idx in range(3):
                orig_val = inner[i][param_idx]
                for delta in [-step_size, step_size]:
                    inner[i][param_idx] = orig_val + delta
                    s, c, a = evaluate(inner)
                    if s < best_side:
                        best_side = s
                        best_inner = [row[:] for row in inner]
                        best_center = c
                        best_angle = a
                        improved = True
                    else:
                        inner[i][param_idx] = orig_val
        
        if not improved:
            step_size *= 0.5
            if step_size < 0.01:
                break
    
    return best_inner, best_center, best_side, best_angle

# EVOLVE-BLOCK-START
def construct():
    global START_TIME
    if time.time() - START_TIME > TIME_LIMIT:
        # Return a trivial solution
        inner = [[0.0, 0.0, 0.0]] * N
        return inner, (0.0, 0.0), 100.0, 0.0
    
    random.seed(42)
    np.random.seed(42)
    
    # Initial placement: hexagonal lattice
    inner = []
    dx = 2.0
    dy = np.sqrt(3)
    
    for row in range(3):
        for col in range(4):
            x = col * dx + (0.5 * dx if row % 2 == 1 else 0)
            y = row * dy
            inner.append([x, y, 0.0])
    
    # Compute initial enclosing hexagon
    all_vertices = []
    for x, y, angle in inner:
        verts = hexagon_vertices(x, y, 1.0, angle)
        all_vertices.extend(verts.tolist())
    
    all_vertices = np.array(all_vertices)
    side, center, angle = min_enclosing_hexagon(all_vertices)
    
    best_side = side
    best_inner = inner
    best_center = center
    best_angle = angle
    
    # Local optimization
    step_size = 0.1
    max_iter = 50
    
    def evaluate(inner):
        all_vertices = []
        for x, y, angle in inner:
            verts = hexagon_vertices(x, y, 1.0, angle)
            all_vertices.extend(verts.tolist())
        all_vertices = np.array(all_vertices)
        s, c, a = min_enclosing_hexagon(all_vertices)
        return s, c, a
    
    for it in range(max_iter):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        
        improved = False
        for i in range(N):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            
            for param_idx in range(3):
                orig_val = inner[i][param_idx]
                for delta in [-step_size, step_size]:
                    inner[i][param_idx] = orig_val + delta
                    s, c, a = evaluate(inner)
                    if s < best_side:
                        best_side = s
                        best_inner = [row[:] for row in inner]
                        best_center = c
                        best_angle = a
                        improved = True
                    else:
                        inner[i][param_idx] = orig_val
        
        if not improved:
            step_size *= 0.5
            if step_size < 0.01:
                break
    
    return best_inner, best_center, best_side, best_angle
# EVOLVE-BLOCK-END
