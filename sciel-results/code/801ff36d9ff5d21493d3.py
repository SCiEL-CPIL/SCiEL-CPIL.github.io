"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial import Delaunay

N = 32
DEADLINE = 120.0
SAFETY_MARGIN = 10.0

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint, linprog
from scipy.spatial import Delaunay
import time
import itertools

N = 32
DEADLINE = 120.0
SAFETY_MARGIN = 10.0


def _phyllotaxis_init(n):
    """Generate n points using phyllotaxis distribution in unit square."""
    i = np.arange(n)
    # Use golden angle for phyllotaxis
    theta = i * np.pi * (3 - np.sqrt(5))
    r = np.sqrt(i / n)
    x = 0.5 + 0.5 * r * np.cos(theta)
    y = 0.5 + 0.5 * r * np.sin(theta)
    return np.column_stack([x, y])


def _grid_init(n):
    """Generate n points using grid distribution in unit square."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < n:
                cells.append([(i + 0.5) / side, (j + 0.5) / side])
    return np.array(cells[:n])


def _get_delaunay_edges(points):
    """Get Delaunay triangulation edges from points."""
    try:
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    edge = tuple(sorted((simplex[i], simplex[j])))
                    edges.add(edge)
        return list(edges)
    except Exception:
        # Fallback: complete graph
        return [(i, j) for i in range(len(points)) for j in range(i + 1, len(points))]


def _solve_sdp_relaxation(points, edges, max_time=5.0):
    """
    Solve SDP relaxation for circle packing.
    Variables: x_i, y_i (positions), r_i (radii) for i in 0..n-1
    Maximize: sum(r_i)
    Subject to:
      x_i >= r_i, x_i <= 1 - r_i, y_i >= r_i, y_i <= 1 - r_i (boundary)
      (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 for (i,j) in edges
      r_i >= 0
    
    We linearize the distance constraints using the contact graph.
    For each edge (i,j): (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    This is non-convex, but we can use a linear approximation around current positions.
    
    Actually, let's use a different approach: we'll use coordinate descent with
    soft-min objective, which is more practical.
    """
    pass  # Not used, we'll use coordinate descent instead


def _softmin_clearance(points, radii, edges, beta=10.0):
    """Compute soft-min of clearances for connected pairs."""
    min_clearance = np.inf
    for i, j in edges:
        dx = points[i, 0] - points[j, 0]
        dy = points[i, 1] - points[j, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        clearance = dist - (radii[i] + radii[j])
        if clearance < min_clearance:
            min_clearance = clearance
    return min_clearance


def _objective_and_grad(x, points, radii, edges, beta=10.0):
    """
    Objective: maximize sum(radii) - penalty * max(0, -min_clearance)^2
    Actually, let's use: maximize sum(radii) subject to clearance >= 0
    We'll use a penalty method.
    """
    n = len(points)
    # x contains: [x_0, y_0, r_0, x_1, y_1, r_1, ..., x_{n-1}, y_{n-1}, r_{n-1}]
    pts = np.array([[x[3*i], x[3*i+1]] for i in range(n)])
    rs = np.array([x[3*i+2] for i in range(n)])
    
    # Boundary constraints: r_i <= x_i <= 1-r_i, r_i <= y_i <= 1-r_i
    # Penalty for boundary violations
    penalty = 0.0
    for i in range(n):
        if rs[i] < 0:
            penalty += 100.0 * rs[i] ** 2
        if pts[i, 0] < rs[i]:
            penalty += 100.0 * (pts[i, 0] - rs[i]) ** 2
        if pts[i, 0] > 1 - rs[i]:
            penalty += 100.0 * (pts[i, 0] - (1 - rs[i])) ** 2
        if pts[i, 1] < rs[i]:
            penalty += 100.0 * (pts[i, 1] - rs[i]) ** 2
        if pts[i, 1] > 1 - rs[i]:
            penalty += 100.0 * (pts[i, 1] - (1 - rs[i])) ** 2
    
    # Distance constraints
    min_clearance = np.inf
    for i, j in edges:
        dx = pts[i, 0] - pts[j, 0]
        dy = pts[i, 1] - pts[j, 1]
        dist = np.sqrt(dx * dx + dy * dy + 1e-12)
        clearance = dist - (rs[i] + rs[j])
        if clearance < 0:
            penalty += 100.0 * clearance ** 2
        if clearance < min_clearance:
            min_clearance = clearance
    
    obj = np.sum(rs) - penalty
    return -obj  # minimize negative


def _coordinate_descent(points, radii, max_iter=200, tol=1e-8):
    """
    Coordinate descent to maximize sum(radii) with disjoint interiors.
    For each circle, adjust its radius and position to maximize radius
    while maintaining non-overlap with all other circles and boundary.
    """
    n = len(points)
    points = points.copy()
    radii = radii.copy()
    
    # Get edges from Delaunay
    edges = _get_delaunay_edges(points)
    
    best_sum = np.sum(radii)
    
    for iteration in range(max_iter):
        improved = False
        
        for i in range(n):
            # Find max possible radius for circle i given current positions of others
            # Constraint from boundary:
            r_max_boundary = min(
                points[i, 0],
                1 - points[i, 0],
                points[i, 1],
                1 - points[i, 1]
            )
            
            # Constraint from other circles:
            r_max_other = r_max_boundary
            for j in range(n):
                if j == i:
                    continue
                dx = points[i, 0] - points[j, 0]
                dy = points[i, 1] - points[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                # r_i + r_j <= dist => r_i <= dist - r_j
                r_candidate = dist - radii[j]
                if r_candidate < r_max_other:
                    r_max_other = r_candidate
            
            r_max_other = max(0, r_max_other)
            
            # Try to increase radius
            new_r = max(radii[i], r_max_other)
            if new_r > radii[i] + 1e-10:
                radii[i] = new_r
                improved = True
            
            # Now try to adjust position to allow larger radius
            # Move circle i in direction that maximizes min clearance
            best_pos = points[i].copy()
            best_r = radii[i]
            
            # Try small perturbations
            for dx in [-0.05, -0.02, 0, 0.02, 0.05]:
                for dy in [-0.05, -0.02, 0, 0.02, 0.05]:
                    new_x = points[i, 0] + dx
                    new_y = points[i, 1] + dy
                    
                    # Check boundary
                    if new_x < 0 or new_x > 1 or new_y < 0 or new_y > 1:
                        continue
                    
                    r_max_b = min(new_x, 1 - new_x, new_y, 1 - new_y)
                    r_max_o = r_max_b
                    
                    for j in range(n):
                        if j == i:
                            continue
                        ddx = new_x - points[j, 0]
                        ddy = new_y - points[j, 1]
                        dist = np.sqrt(ddx * ddx + ddy * ddy)
                        r_cand = dist - radii[j]
                        if r_cand < r_max_o:
                            r_max_o = r_cand
                    
                    r_max_o = max(0, r_max_o)
                    
                    if r_max_o > best_r + 1e-10:
                        best_r = r_max_o
                        best_pos = np.array([new_x, new_y])
            
            if best_r > radii[i] + 1e-10:
                points[i] = best_pos
                radii[i] = best_r
                improved = True
                # Update edges
                edges = _get_delaunay_edges(points)
        
        current_sum = np.sum(radii)
        if current_sum > best_sum:
            best_sum = current_sum
        
        if not improved:
            break
    
    return points, radii, best_sum


def _refine_with_optimization(points, radii, max_time=30.0):
    """
    Use scipy minimize to refine the solution.
    """
    n = len(points)
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = points[i, 0]
        x0[3*i+1] = points[i, 1]
        x0[3*i+2] = radii[i]
    
    edges = _get_delaunay_edges(points)
    
    def objective(x):
        pts = np.array([[x[3*i], x[3*i+1]] for i in range(n)])
        rs = np.array([x[3*i+2] for i in range(n)])
        
        penalty = 0.0
        for i in range(n):
            if rs[i] < 0:
                penalty += 1000.0 * rs[i] ** 2
            if pts[i, 0] < rs[i]:
                penalty += 1000.0 * (pts[i, 0] - rs[i]) ** 2
            if pts[i, 0] > 1 - rs[i]:
                penalty += 1000.0 * (pts[i, 0] - (1 - rs[i])) ** 2
            if pts[i, 1] < rs[i]:
                penalty += 1000.0 * (pts[i, 1] - rs[i]) ** 2
            if pts[i, 1] > 1 - rs[i]:
                penalty += 1000.0 * (pts[i, 1] - (1 - rs[i])) ** 2
        
        for i, j in edges:
            dx = pts[i, 0] - pts[j, 0]
            dy = pts[i, 1] - pts[j, 1]
            dist = np.sqrt(dx * dx + dy * dy + 1e-12)
            clearance = dist - (rs[i] + rs[j])
            if clearance < 0:
                penalty += 1000.0 * clearance ** 2
        
        return -np.sum(rs) + penalty
    
    start_time = time.time()
    try:
        result = minimize(
            objective, x0, method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.x is not None:
            pts = np.array([[result.x[3*i], result.x[3*i+1]] for i in range(n)])
            rs = np.array([result.x[3*i+2] for i in range(n)])
            
            # Clamp to valid range
            for i in range(n):
                rs[i] = max(0, min(rs[i], 0.5))
                pts[i, 0] = max(rs[i], min(pts[i, 0], 1 - rs[i]))
                pts[i, 1] = max(rs[i], min(pts[i, 1], 1 - rs[i]))
            
            # Verify and fix any remaining overlaps
            for _ in range(10):
                changed = False
                for i in range(n):
                    for j in range(i+1, n):
                        dx = pts[i, 0] - pts[j, 0]
                        dy = pts[i, 1] - pts[j, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < rs[i] + rs[j] + 1e-10:
                            # Reduce the smaller radius
                            if rs[i] < rs[j]:
                                rs[i] = max(0, (dist - rs[j]) * 0.99)
                            else:
                                rs[j] = max(0, (dist - rs[i]) * 0.99)
                            changed = True
                if not changed:
                    break
            
            return pts, rs
    
    except Exception:
        pass
    
    return points, radii


def construct():
    """
    Pack 32 circles with disjoint interiors inside [0,1]^2 and maximize sum of radii.
    Returns array of shape (32, 3): rows (x, y, radius).
    """
    start_time = time.time()
    
    # Try multiple initializations
    best_points = None
    best_radii = None
    best_sum = -np.inf
    
    inits = [
        _phyllotaxis_init(N),
        _grid_init(N),
    ]
    
    # Add some random inits
    np.random.seed(42)
    for _ in range(3):
        inits.append(np.random.rand(N, 2) * 0.8 + 0.1)
    
    for init_pts in inits:
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        # Initialize radii
        init_radii = np.full(N, 0.1)
        
        # Coordinate descent
        pts, radii, s = _coordinate_descent(init_pts, init_radii, max_iter=100)
        
        if s > best_sum:
            best_sum = s
            best_points = pts.copy()
            best_radii = radii.copy()
    
    if best_points is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    r = 0.5 / side - 1e-3
                    cells.append([(i + 0.5) / side, (j + 0.5) / side, r])
        return np.array(cells[:N])
    
    # Refine with optimization
    if time.time() - start_time < DEADLINE - SAFETY_MARGIN:
        best_points, best_radii = _refine_with_optimization(
            best_points, best_radii,
            max_time=DEADLINE - SAFETY_MARGIN - (time.time() - start_time)
        )
    
    # Final verification and clamping
    for i in range(N):
        r = max(0, best_radii[i])
        best_radii[i] = r
        best_points[i, 0] = max(r, min(best_points[i, 0], 1 - r))
        best_points[i, 1] = max(r, min(best_points[i, 1], 1 - r))
    
    # Fix any remaining overlaps
    for _ in range(50):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                dx = best_points[i, 0] - best_points[j, 0]
                dy = best_points[i, 1] - best_points[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < best_radii[i] + best_radii[j] + 1e-12:
                    # Reduce the smaller radius
                    if best_radii[i] < best_radii[j]:
                        new_r = max(0, (dist - best_radii[j]) * 0.99)
                        if new_r < best_radii[i]:
                            best_radii[i] = new_r
                            changed = True
                    else:
                        new_r = max(0, (dist - best_radii[i]) * 0.99)
                        if new_r < best_radii[j]:
                            best_radii[j] = new_r
                            changed = True
        if not changed:
            break
    
    # Build result
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = best_points[i, 0]
        result[i, 1] = best_points[i, 1]
        result[i, 2] = best_radii[i]
    
    return result
# EVOLVE-BLOCK-END
