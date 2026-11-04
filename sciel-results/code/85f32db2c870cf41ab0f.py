"""Rewritten program: force-directed circle packing in perimeter-4 rectangle."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
DEADLINE = 115.0  # safety margin before 120s limit
EPS = 1e-6

def _rect_perimeter(w, h):
    return 2.0 * (w + h)

def _compute_rect(circles):
    """Compute bounding rectangle of circles (x,y,r)."""
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y =np.max(ys + rs)
    return max_x - min_x, max_y - min_y

def _is_valid(circles, tol=1e-5):
    """Check if circles are valid: no overlaps, perimeter <= 4."""
    if len(circles) != N:
        return False
    # Check non-negative radii
    if np.any(circles[:, 2] <= 0):
        return False
    # Check perimeter
    w, h = _compute_rect(circles)
    if _rect_perimeter(w, h) > 4.0 + tol:
        return False
    # Check pairwise non-overlap
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i, 2] + circles[j, 2] - tol:
                return False
    return True

def _score(circles):
    """Sum of radii, 0 if invalid."""
    if not _is_valid(circles):
        return 0.0
    return np.sum(circles[:, 2])

def _grid_init(side, n_circles):
    """Create a grid of circles in a unit square, scaled later."""
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-3
            cells.append([x, y, r])
    return np.array(cells[:n_circles])

def _force_directed_step(circles, dt=0.001, k_repel=10.0, k_wall=5.0, k_contact=1.0):
    """
    One force-directed step:
    - Repulsive forces between overlapping circles
    - Attractive forces to maintain contact (spring-like for touching pairs)
    - Wall repulsion to keep circles inside bounding box
    Returns updated circles.
    """
    n = len(circles)
    forces = np.zeros((n, 2))
    
    # Bounding box
    w, h = _compute_rect(circles)
    min_x = np.min(circles[:, 0] - circles[:, 2])
    max_x = np.max(circles[:, 0] + circles[:, 2])
    min_y = np.min(circles[:, 1] - circles[:, 2])
    max_y = np.max(circles[:, 1] + circles[:, 2])
    
    # Perimeter constraint: 2*(w+h) <= 4 => w+h <= 2
    # We'll work in a normalized space and scale at the end.
    # For now, just apply forces in local coordinates.
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = circles[i, 2] + circles[j, 2]
            if dist < 1e-9:
                dist = 1e-9
            if dist < min_dist:
                # Repulsive force
                overlap = min_dist - dist
                fx = k_repel * overlap * dx / dist
                fy = k_repel * overlap * dy / dist
                forces[i, 0] += fx
                forces[i, 1] += fy
                forces[j, 0] -= fx
                forces[j, 1] -= fy
            elif dist < min_dist * 1.1:
                # Mild attractive force to encourage contact
                gap = dist - min_dist
                fx = -k_contact * 0.1 * gap * dx / dist
                fy = -k_contact * 0.1 * gap * dy / dist
                forces[i, 0] += fx
                forces[i, 1] += fy
                forces[j, 0] -= fx
                forces[j, 1] -= fy
    
    # Wall forces: push circles away from bounding box edges
    # We want to keep the bounding box compact
    for i in range(n):
        # Left wall
        if circles[i, 0] - circles[i, 2] < min_x + 1e-6:
            forces[i, 0] += k_wall * 0.1
        # Right wall
        if circles[i, 0] + circles[i, 2] > max_x - 1e-6:
            forces[i, 0] -= k_wall * 0.1
        # Bottom wall
        if circles[i, 1] - circles[i, 2] < min_y + 1e-6:
            forces[i, 1] += k_wall * 0.1
        # Top wall
        if circles[i, 1] + circles[i, 2] > max_y - 1e-6:
            forces[i, 1] -= k_wall * 0.1
    
    circles[:, 0] += dt * forces[:, 0]
    circles[:, 1] += dt * forces[:, 1]
    return circles

def _optimize_radii(positions, w, h, max_iter=500):
    """
    Given fixed positions and rectangle dimensions, optimize radii.
    Maximize sum of radii subject to:
    - r_i + r_j <= dist(i,j) for all pairs
    - r_i <= x_i - min_x, r_i <= max_x - x_i, r_i <= y_i - min_y, r_i <= max_y - y_i
    - r_i > 0
    """
    n = len(positions)
    xs = positions[:, 0]
    ys = positions[:, 1]
    
    # Compute all pairwise distances
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d[i, j] = np.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2)
    
    # Constraints on radii
    # r_i <= x_i, r_i <= w - x_i, r_i <= y_i, r_i <= h - y_i (assuming min_x=0, min_y=0)
    # Actually, we need to be careful. Let's place the rectangle so that min_x=0, min_y=0.
    # Then constraints are:
    # r_i <= x_i, r_i <= w - x_i, r_i <= y_i, r_i <= h - y_i
    # And r_i + r_j <= d[i,j]
    
    from scipy.optimize import linprog
    
    # Maximize sum(r_i) => minimize -sum(r_i)
    c = -np.ones(n)
    
    # Inequality constraints: A_ub @ r <= b_ub
    # Pairwise: r_i + r_j <= d[i,j]
    A_ub = []
    b_ub = []
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            A_ub.append(row)
            b_ub.append(d[i, j])
    
    # Wall constraints
    for i in range(n):
        for val in [xs[i], w - xs[i], ys[i], h - ys[i]]:
            row = np.zeros(n)
            row[i] = 1
            A_ub.append(row)
            b_ub.append(max(val, 0))
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = [(1e-8, None)] * n
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.status == 0:
        radii = res.x
        return radii
    else:
        # Fallback: greedy radius assignment
        radii = np.zeros(n)
        for i in range(n):
            r_max = min(xs[i], w - xs[i], ys[i], h - ys[i])
            for j in range(i):
                r_max = min(r_max, d[i, j] - radii[j])
            radii[i] = max(r_max, 1e-8)
        return radii

def _solve_with_given_layout(positions, w, h):
    """Solve for optimal radii given positions and rectangle."""
    # Shift positions so min_x=0, min_y=0
    min_x = np.min(positions[:, 0])
    min_y = np.min(positions[:, 1])
    positions = positions.copy()
    positions[:, 0] -= min_x
    positions[:, 1] -= min_y
    
    radii = _optimize_radii(positions, w, h)
    
    circles = np.column_stack([positions[:, 0], positions[:, 1], radii])
    return circles

def _random_positions(n, w, h, seed=None):
    """Generate random positions within [0,w]x[0,h]."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0, w, n)
    ys = rng.uniform(0, h, n)
    return np.column_stack([xs, ys])

def construct():
    start_time = time.time()
    best_circles = None
    best_score = 0.0
    
    # Initial grid solution
    side = int(np.ceil(np.sqrt(N)))
    grid = _grid_init(side, N)
    # Scale to perimeter 4: w+h=2, let's use w=h=1
    w, h = 1.0, 1.0
    grid_circles = _solve_with_given_layout(grid[:, :2], w, h)
    if _is_valid(grid_circles):
        best_circles = grid_circles
        best_score = _score(grid_circles)
    
    # Try different aspect ratios
    aspect_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]
    
    for ar in aspect_ratios:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # w + h = 2, w/h = ar => w = 2*ar/(1+ar), h = 2/(1+ar)
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 / (1.0 + ar)
        
        # Try grid initialization
        side = int(np.ceil(np.sqrt(N * ar)))
        if side < 2:
            side = 2
        grid = _grid_init(side, N)
        # Scale grid to [0,1]x[0,1] then to [0,w]x[0,h]
        grid_pos = grid[:, :2].copy()
        grid_pos[:, 0] *= w
        grid_pos[:, 1] *= h
        circles = _solve_with_given_layout(grid_pos, w, h)
        if _is_valid(circles):
            s = _score(circles)
            if s > best_score:
                best_score = s
                best_circles = circles
        
        # Try random initializations with force-directed relaxation
        for trial in range(3):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            seed = hash((ar, trial)) % (2**32)
            positions = _random_positions(N, w, h, seed=seed)
            
            # Force-directed relaxation
            circles_fd = np.column_stack([positions, np.full(N, 0.01)])
            for step in range(50):
                if time.time() - start_time > DEADLINE - 5:
                    break
                circles_fd = _force_directed_step(circles_fd, dt=0.002)
            
            # Optimize radii
            circles = _solve_with_given_layout(circles_fd[:, :2], w, h)
            if _is_valid(circles):
                s = _score(circles)
                if s > best_score:
                    best_score = s
                    best_circles = circles
    
    # Final: try to improve best solution with local optimization
    if best_circles is not None and time.time() - start_time < DEADLINE - 10:
        # Perturb and re-optimize
        for _ in range(5):
            if time.time() - start_time > DEADLINE - 5:
                break
            perturbed = best_circles.copy()
            perturbed[:, 0] += np.random.normal(0, 0.01, N)
            perturbed[:, 1] += np.random.normal(0, 0.01, N)
            w, h = _compute_rect(perturbed)
            circles = _solve_with_given_layout(perturbed[:, :2], w, h)
            if _is_valid(circles):
                s = _score(circles)
                if s > best_score:
                    best_score = s
                    best_circles = circles
    
    if best_circles is None:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        best_circles = _grid_init(side, N)
        w, h = _compute_rect(best_circles)
        # Scale to perimeter 4
        scale = 2.0 / (w + h)
        best_circles[:, 0] *= scale
        best_circles[:, 1] *= scale
        best_circles[:, 2] *= scale
    
    return best_circles
