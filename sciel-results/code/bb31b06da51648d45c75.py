"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

N = 11
SQRT3 = np.sqrt(3)
TRI_AREA = SQRT3 / 4  # area of unit-side equilateral triangle

def _area(p1, p2, p3):
    """Area of triangle with vertices p1, p2, p3."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _all_areas(points):
    """Compute all C(n,3) triangle areas. Returns array of areas and the min."""
    n = len(points)
    areas = []
    min_a = float('inf')
    min_idx = None
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = _area(points[i], points[j], points[k])
                areas.append(a)
                if a < min_a:
                    min_a = a
                    min_idx = (i, j, k)
    return np.array(areas), min_a, min_idx

def _min_area_and_triple(points):
    """Return min area and the triple index."""
    n = len(points)
    min_a = float('inf')
    min_idx = None
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = _area(points[i], points[j], points[k])
                if a < min_a:
                    min_a = a
                    min_idx = (i, j, k)
    return min_a, min_idx

def _inside_triangle(p):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    x, y = p
    if y < -1e-9:
        return False
    if y > SQRT3/2 + 1e-9:
        return False
    # Left edge: from (0,0) to (0.5, SQRT3/2), line: y = sqrt(3)*x
    # Right edge: from (1,0) to (0.5, SQRT3/2), line: y = -sqrt(3)*(x-1)
    if y > SQRT3 * x + 1e-9:
        return False
    if y > -SQRT3 * (x - 1) + 1e-9:
        return False
    return True

def _clamp_to_triangle(p):
    """Clamp point to be inside the triangle."""
    x, y = p
    # Clamp to bounds
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(SQRT3/2, y))
    # Check left edge
    if y > SQRT3 * x:
        # Project onto left edge
        t = y / SQRT3
        x = t
        y = SQRT3 * t
    # Check right edge
    if y > -SQRT3 * (x - 1):
        # Project onto right edge
        t = (1 - x)
        x = 1 - t
        y = SQRT3 * t
    # Final clamp
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(SQRT3/2, y))
    return np.array([x, y])

def _structured_init(seed=0):
    """Create a structured initial configuration."""
    import random
    rng = random.Random(seed)
    points = []
    # Vertices
    points.append([0.0, 0.0])
    points.append([1.0, 0.0])
    points.append([0.5, SQRT3/2])
    # Midpoints
    points.append([0.5, 0.0])
    points.append([0.25, SQRT3/4])
    points.append([0.75, SQRT3/4])
    # Center-ish points
    points.append([0.5, SQRT3/6])
    points.append([0.333, SQRT3/6])
    points.append([0.667, SQRT3/6])
    # Two more
    points.append([0.5, SQRT3/3])
    points.append([0.4, SQRT3/5])
    
    # Add small random perturbation
    for i in range(len(points)):
        points[i][0] += rng.uniform(-0.02, 0.02)
        points[i][1] += rng.uniform(-0.02, 0.02)
        points[i] = _clamp_to_triangle(points[i])
    
    return np.array(points[:N])

def _random_init(seed=0):
    """Create a random initial configuration."""
    import random
    rng = random.Random(seed)
    points = []
    while len(points) < N:
        u, v = rng.random(), rng.random()
        if u + v < 0.95:
            p = [u + 0.5*v, SQRT3/2 * v]
            p = _clamp_to_triangle(p)
            points.append(p)
    return np.array(points)

def _local_search(points, max_iter=5000, step_scale=0.05, deadline=None):
    """Iterative local search: perturb random point, accept if min area improves."""
    import random
    rng = random.Random(hash(tuple(map(tuple, points))) % 10000)
    points = points.copy()
    min_a, min_idx = _min_area_and_triple(points)
    best_min_a = min_a
    best_points = points.copy()
    step = step_scale
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        # Pick a random point to perturb
        idx = rng.randrange(N)
        orig = points[idx].copy()
        # Random perturbation
        dx = rng.gauss(0, step)
        dy = rng.gauss(0, step)
        new_p = np.array([orig[0] + dx, orig[1] + dy])
        new_p = _clamp_to_triangle(new_p)
        points[idx] = new_p
        
        new_min_a, _ = _min_area_and_triple(points)
        
        if new_min_a > best_min_a:
            best_min_a = new_min_a
            best_points = points.copy()
            step *= 0.95  # shrink step slightly
        else:
            # Revert
            points[idx] = orig
            step *= 0.99  # shrink step more when not improving
    
    return best_points, best_min_a

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    best_points = None
    best_score = 0
    
    # Try multiple initial configurations
    inits = []
    # Structured inits with different seeds
    for s in range(20):
        inits.append(_structured_init(seed=s))
    # Random inits
    for s in range(10):
        inits.append(_random_init(seed=s))
    
    for i, init in enumerate(inits):
        if time.time() > deadline:
            break
        
        # Evaluate initial
        min_a, _ = _min_area_and_triple(init)
        if min_a > best_score:
            best_score = min_a
            best_points = init.copy()
        
        # Run local search
        pts, score = _local_search(init, max_iter=2000, step_scale=0.03, deadline=deadline)
        if score > best_score:
            best_score = score
            best_points = pts
    
    if best_points is None:
        best_points = _structured_init(seed=0)
    
    return best_points
