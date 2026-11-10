"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50
DEADLINE = 120.0  # seconds, safety margin below 150s kill
SAFE_MARGIN = 5.0  # seconds to leave before deadline


def _compute_score(pts):
    """Compute the packing radius: min of half-min-pair-distance and min-boundary-margin."""
    n = len(pts)
    # Pairwise distances
    min_pair = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < min_pair:
                min_pair = d
    # Boundary margins
    min_margin = np.inf
    for i in range(n):
        r = np.linalg.norm(pts[i])
        margin = 1.0 - r
        if margin < min_margin:
            min_margin = margin
    return min(min_pair / 2.0, min_margin)


def _hexagonal_init(n):
    """Generate a hexagonal lattice packing inside the unit disk."""
    # Try different spacings to get at least n points
    for spacing in np.linspace(0.25, 0.08, 80):
        # Generate hex lattice points
        points = []
        k_max = int(1.0 / spacing) + 3
        for j in range(-k_max, k_max + 1):
            for i in range(-k_max, k_max + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing / 2.0:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= n:
            # Sort by radius and take closest to center
            radii = np.linalg.norm(points, axis=1)
            idx = np.argsort(radii)[:n]
            return points[idx]
    # Fallback: random
    theta = np.random.uniform(0, 2 * np.pi, n)
    r = np.sqrt(np.random.uniform(0, 1, n)) * 0.8
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def _grid_neighbors(pts, cell_size):
    """Build a grid-based spatial hash for O(1) neighbor lookups."""
    n = len(pts)
    grid = {}
    for i in range(n):
        cx = int(pts[i, 0] / cell_size)
        cy = int(pts[i, 1] / cell_size)
        key = (cx, cy)
        if key not in grid:
            grid[key] = []
        grid[key].append(i)
    return grid


def _get_nearby(grid, i, cell_size):
    """Get all points within 2*cell_size of point i."""
    cx = int(pts[i, 0] / cell_size)
    cy = int(pts[i, 1] / cell_size)
    nearby = []
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            key = (cx + dx, cy + dy)
            if key in grid:
                nearby.extend(grid[key])
    return nearby


def _physics_relax(pts, max_time, initial_temp=0.01):
    """Physics-inspired relaxation with grid-based neighbor lookup."""
    global pts
    n = len(pts)
    start_time = time.time()
    deadline = start_time + max_time
    
    # Initial grid
    cell_size = 0.3
    grid = _grid_neighbors(pts, cell_size)
    
    best_score = _compute_score(pts)
    best_pts = pts.copy()
    
    # Simulated annealing with directed moves
    temp = initial_temp
    cooling = 0.9995
    
    for iteration in range(200000):
        if time.time() > deadline:
            break
        
        # Pick a random point to move
        i = np.random.randint(n)
        
        # Get nearby points for efficient force calculation
        nearby = _get_nearby(grid, i, cell_size)
        
        # Compute repulsive force from nearby points
        fx, fy = 0.0, 0.0
        for j in nearby:
            if j == i:
                continue
            dx = pts[i, 0] - pts[j, 0]
            dy = pts[i, 1] - pts[j, 1]
            d2 = dx * dx + dy * dy
            d = np.sqrt(d2) if d2 > 1e-12 else 1e-12
            # Repulsive force ~ 1/d^2, capped
            if d < 0.5:
                force = 1.0 / (d * d + 0.01)
                fx += force * dx / d
                fy += force * dy / d
        
        # Boundary repulsion
        r = np.hypot(pts[i, 0], pts[i, 1])
        if r > 0.8:
            force = (r - 0.8) * 5.0
            if r > 1e-6:
                fx -= force * pts[i, 0] / r
                fy -= force * pts[i, 1] / r
        
        # Random perturbation
        fx += np.random.normal(0, temp)
        fy += np.random.normal(0, temp)
        
        # Move
        old_x, old_y = pts[i, 0], pts[i, 1]
        step = 0.02
        new_x = old_x + step * fx
        new_y = old_y + step * fy
        
        # Keep inside unit circle
        new_r = np.hypot(new_x, new_y)
        if new_r > 0.95:
            new_x *= 0.95 / new_r
            new_y *= 0.95 / new_r
        
        pts[i, 0] = new_x
        pts[i, 1] = new_y
        
        # Update grid
        old_cx = int(old_x / cell_size)
        old_cy = int(old_y / cell_size)
        new_cx = int(new_x / cell_size)
        new_cy = int(new_y / cell_size)
        
        old_key = (old_cx, old_cy)
        new_key = (new_cx, new_cy)
        
        if old_key != new_key:
            if old_key in grid:
                grid[old_key].remove(i)
                if not grid[old_key]:
                    del grid[old_key]
            if new_key not in grid:
                grid[new_key] = []
            grid[new_key].append(i)
        
        # Periodically check score
        if iteration % 500 == 0:
            score = _compute_score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        
        temp *= cooling
    
    return best_pts, best_score


def _local_polish(pts, max_time):
    """High-precision local optimization using gradient descent on the min-objective."""
    global pts
    n = len(pts)
    start_time = time.time()
    deadline = start_time + max_time
    
    best_score = _compute_score(pts)
    best_pts = pts.copy()
    
    # Identify the binding constraint (which pair or boundary limits the radius)
    for iteration in range(50000):
        if time.time() > deadline:
            break
        
        # Find the minimum distance pair and minimum boundary margin
        min_pair_dist = np.inf
        min_pair_idx = (0, 1)
        min_boundary_margin = np.inf
        min_boundary_idx = 0
        
        for i in range(n):
            r = np.linalg.norm(pts[i])
            margin = 1.0 - r
            if margin < min_boundary_margin:
                min_boundary_margin = margin
                min_boundary_idx = i
            for j in range(i + 1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_pair_dist:
                    min_pair_dist = d
                    min_pair_idx = (i, j)
        
        current_radius = min(min_pair_dist / 2.0, min_boundary_margin)
        
        if current_radius > best_score:
            best_score = current_radius
            best_pts = pts.copy()
        
        # Determine which constraint is binding
        if min_pair_dist / 2.0 < min_boundary_margin:
            # Pair constraint is binding: push the two points apart
            i, j = min_pair_idx
            dx = pts[i, 0] - pts[j, 0]
            dy = pts[i, 1] - pts[j, 1]
            d = np.hypot(dx, dy)
            if d > 1e-9:
                dx /= d
                dy /= d
                step = 0.005
                pts[i, 0] += step * dx
                pts[i, 1] += step * dy
                pts[j, 0] -= step * dx
                pts[j, 1] -= step * dy
        else:
            # Boundary constraint is binding: push point inward
            i = min_boundary_idx
            r = np.linalg.norm(pts[i])
            if r > 1e-9:
                step = 0.005
                pts[i, 0] -= step * pts[i, 0] / r
                pts[i, 1] -= step * pts[i, 1] / r
        
        # Ensure all points stay inside unit circle
        for k in range(n):
            rk = np.linalg.norm(pts[k])
            if rk > 0.999:
                pts[k, 0] *= 0.999 / rk
                pts[k, 1] *= 0.999 / rk
    
    return best_pts, best_score


# EVOLVE-BLOCK-START
def construct():
    """Construct 50 equal circles inside unit circle, maximizing common radius."""
    start_time = time.time()
    remaining = DEADLINE - SAFE_MARGIN - (time.time() - start_time)
    
    # Phase 1: Hexagonal initialization
    pts = _hexagonal_init(N)
    score = _compute_score(pts)
    best_pts = pts.copy()
    best_score = score
    
    # Phase 2: Physics-inspired relaxation with grid-based neighbors
    time_left = remaining - (time.time() - start_time)
    if time_left > 2.0:
        pts, score = _physics_relax(pts, max_time=time_left * 0.5)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    # Phase 3: Local polish on the best configuration
    time_left = remaining - (time.time() - start_time)
    if time_left > 1.0:
        pts, score = _local_polish(best_pts, max_time=time_left * 0.8)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    # Final check
    final_score = _compute_score(best_pts)
    if final_score > best_score:
        best_score = final_score
    
    return best_pts
# EVOLVE-BLOCK-END
