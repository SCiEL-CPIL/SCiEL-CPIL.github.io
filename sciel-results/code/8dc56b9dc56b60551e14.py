"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 11
SQRT3 = np.sqrt(3)
AREA_EQUI = SQRT3 / 4.0  # area of equilateral triangle with side 1

# EVOLVE-BLOCK-START

def _area_tri(p1, p2, p3):
    """Compute area of triangle formed by three points."""
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    return 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))

def _min_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points."""
    n = len(points)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _area_tri(points[i], points[j], points[k])
                if a < min_a:
                    min_a = a
    return min_a

def _is_inside(pt, eps=1e-10):
    """Check if point is inside or on the boundary of the unit equilateral triangle."""
    x, y = pt
    if y < -eps:
        return False
    if y > SQRT3 * x + eps:
        return False
    if y > -SQRT3 * (x - 1) + eps:
        return False
    return True

def _project_to_triangle(pt):
    """Project a point to the nearest point inside the equilateral triangle."""
    x, y = pt
    if _is_inside(pt):
        return np.array([x, y])
    
    # Find the closest point on the boundary
    # The triangle has three edges:
    # Edge 1: from (0,0) to (1,0) -> y=0
    # Edge 2: from (0,0) to (0.5, sqrt(3)/2) -> y = sqrt(3)*x
    # Edge 3: from (1,0) to (0.5, sqrt(3)/2) -> y = -sqrt(3)*(x-1)
    
    best_pt = None
    best_dist = float('inf')
    
    # Project to edge 1: y = 0, 0 <= x <= 1
    px, py = x, 0.0
    px = np.clip(px, 0, 1)
    d = (px - x)**2 + (py - y)**2
    if d < best_dist:
        best_dist = d
        best_pt = np.array([px, py])
    
    # Project to edge 2: y = sqrt(3)*x, from (0,0) to (0.5, sqrt(3)/2)
    # Line: sqrt(3)*x - y = 0
    # Distance from (x,y) to line ax+by+c=0 is |ax+by+c|/sqrt(a^2+b^2)
    a, b, c = SQRT3, -1.0, 0.0
    num = abs(a*x + b*y + c)
    den = np.sqrt(a*a + b*b)
    # Foot of perpendicular
    t = (a*x + b*y + c) / (a*a + b*b)
    fx = x - a * t
    fy = y - b * t
    # Check if foot is on the segment
    # Parametrize: (0,0) to (0.5, sqrt(3)/2)
    # Point on line: s*(0.5, sqrt(3)/2), s in [0,1]
    if 0 <= fx <= 0.5 and 0 <= fy <= SQRT3/2:
        d = (fx - x)**2 + (fy - y)**2
        if d < best_dist:
            best_dist = d
            best_pt = np.array([fx, fy])
    else:
        # Check endpoints
        for ex, ey in [(0,0), (0.5, SQRT3/2)]:
            d = (ex - x)**2 + (ey - y)**2
            if d < best_dist:
                best_dist = d
                best_pt = np.array([ex, ey])
    
    # Project to edge 3: y = -sqrt(3)*(x-1) = -sqrt(3)*x + sqrt(3)
    # Line: sqrt(3)*x + y - sqrt(3) = 0
    a, b, c = SQRT3, 1.0, -SQRT3
    t = (a*x + b*y + c) / (a*a + b*b)
    fx = x - a * t
    fy = y - b * t
    if 0.5 <= fx <= 1.0 and 0 <= fy <= SQRT3/2:
        d = (fx - x)**2 + (fy - y)**2
        if d < best_dist:
            best_dist = d
            best_pt = np.array([fx, fy])
    else:
        for ex, ey in [(1,0), (0.5, SQRT3/2)]:
            d = (ex - x)**2 + (ey - y)**2
            if d < best_dist:
                best_dist = d
                best_pt = np.array([ex, ey])
    
    return best_pt

def _objective_flat(x_flat):
    """Objective function for optimization: negative of min area."""
    points = x_flat.reshape(N, 2)
    # Project any points outside back inside
    for i in range(N):
        if not _is_inside(points[i]):
            points[i] = _project_to_triangle(points[i])
    return -_min_area(points)

def _gradient_free_optimize(points, max_iter=500, step_size=0.05):
    """Simple Nelder-Mead style optimization using scipy."""
    x0 = points.flatten()
    result = minimize(_objective_flat, x0, method='Nelder-Mead',
                      options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10})
    best_points = result.x.reshape(N, 2)
    for i in range(N):
        if not _is_inside(best_points[i]):
            best_points[i] = _project_to_triangle(best_points[i])
    return best_points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    # Start with a reasonable initial configuration
    # Use a grid-like or symmetric starting point
    # For 11 points, try a configuration with some symmetry
    
    # Initial guess: distribute points somewhat evenly
    # Use a hexagonal-like packing or grid
    best_points = None
    best_score = -float('inf')
    
    # Try several initial configurations
    rng = np.random.RandomState(42)
    
    # Configuration 1: Symmetric arrangement
    # Place points in a pattern that exploits symmetry
    # For 11 points in equilateral triangle, try:
    # 3 on each edge (including corners) + 5 interior
    # Or use a known good starting point
    
    # Let's start with a semi-structured initial guess
    # Based on known results for small N heilbronn problems
    initial_configs = []
    
    # Config A: Grid-like
    # Row 0 (bottom): 4 points
    # Row 1: 3 points  
    # Row 2: 2 points
    # Row 3 (top): 2 points
    # Total: 11
    
    # Actually, let me use a more systematic approach
    # Start with points on a triangular grid
    
    # Triangular lattice points within the triangle
    # For a triangular grid with spacing s, points at (i*s, j*s*sqrt(3)/2)
    # where i,j >= 0 and i + j/sqrt(3) <= 1/sqrt(3) ... 
    # Actually in barycentric-like coordinates:
    # Point (i, j) maps to (i*s + j*s/2, j*s*sqrt(3)/2)
    # Constraint: i*s + j*s/2 <= 1 and j*s*sqrt(3)/2 <= sqrt(3)/2
    # => i + j/2 <= 1/s and j <= 1/s
    # So i + j/2 <= n where n = 1/s
    
    for n_grid in [3, 4, 5]:
        s = 1.0 / n_grid
        pts = []
        for j in range(n_grid + 1):
            for i in range(n_grid - j + 1):
                x = i * s + j * s / 2
                y = j * s * SQRT3 / 2
                if _is_inside([x, y]):
                    pts.append([x, y])
        
        if len(pts) >= N:
            # Take first N or sample N
            if len(pts) > N:
                # Select N points that are well-distributed
                indices = np.linspace(0, len(pts)-1, N, dtype=int)
                selected = [pts[i] for i in indices]
            else:
                selected = pts
            initial_configs.append(np.array(selected))
    
    # Config B: Random with some structure
    for seed in range(5):
        rng2 = np.random.RandomState(seed)
        pts = []
        # Add some structure: include midpoints of edges
        pts.append([0.0, 0.0])
        pts.append([1.0, 0.0])
        pts.append([0.5, SQRT3/2])
        pts.append([0.5, 0.0])
        pts.append([0.25, SQRT3/4])
        pts.append([0.75, SQRT3/4])
        # Fill remaining randomly
        while len(pts) < N:
            x = rng2.random()
            y = rng2.random() * SQRT3 / 2
            if _is_inside([x, y]):
                pts.append([x, y])
        initial_configs.append(np.array(pts[:N]))
    
    # Evaluate each initial config and pick the best
    for config in initial_configs:
        if len(config) != N:
            continue
        # Ensure all points are inside
        valid = True
        for i in range(N):
            if not _is_inside(config[i]):
                valid = False
                break
        if not valid:
            continue
        score = _min_area(config)
        if score > best_score:
            best_score = score
            best_points = config.copy()
    
    if best_points is None:
        # Fallback: simple random
        pts = []
        while len(pts) < N:
            x = rng.random()
            y = rng.random() * SQRT3 / 2
            if _is_inside([x, y]):
                pts.append([x, y])
        best_points = np.array(pts)
        best_score = _min_area(best_points)
    
    # Now optimize from the best starting point
    current_points = best_points.copy()
    current_score = best_score
    
    # Use scipy's Nelder-Mead for local optimization
    while time.time() < deadline - 10.0:
        try:
            optimized = _gradient_free_optimize(current_points, max_iter=200, step_size=0.03)
            opt_score = _min_area(optimized)
            if opt_score > current_score:
                current_points = optimized
                current_score = opt_score
            else:
                # Try a smaller perturbation
                break
        except Exception:
            break
    
    # Try multiple restarts with perturbations if time allows
    while time.time() < deadline - 5.0:
        # Perturb current best
        perturbed = current_points + rng.normal(0, 0.02, size=(N, 2))
        # Project back inside
        for i in range(N):
            if not _is_inside(perturbed[i]):
                perturbed[i] = _project_to_triangle(perturbed[i])
        
        try:
            optimized = _gradient_free_optimize(perturbed, max_iter=100, step_size=0.02)
            opt_score = _min_area(optimized)
            if opt_score > current_score:
                current_points = optimized
                current_score = opt_score
        except Exception:
            break
    
    # Final projection to ensure validity
    for i in range(N):
        if not _is_inside(current_points[i]):
            current_points[i] = _project_to_triangle(current_points[i])
    
    return current_points

# EVOLVE-BLOCK-END
