"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay

DEADLINE = 150.0
START_TIME = time.time()

def _remaining():
    return DEADLINE - (time.time() - START_TIME) - 5.0

def _valid_check(centers, radii, tol=1e-6):
    """Check if packing is valid."""
    n = len(radii)
    for i in range(n):
        if radii[i] < -tol:
            return False
        if centers[i, 0] - radii[i] < -tol:
            return False
        if 1.0 - centers[i, 0] - radii[i] < -tol:
            return False
        if centers[i, 1] - radii[i] < -tol:
            return False
        if 1.0 - centers[i, 1] - radii[i] < -tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
            if d < radii[i] + radii[j] - tol:
                return False
    return True

def _get_neighbors(centers, radii, margin=0.05):
    """Get Delaunay neighbors for efficient constraint handling."""
    n = len(centers)
    if n < 3:
        return set()
    try:
        tri = Delaunay(centers)
        neighbors = set()
        for simplex in tri.simplices:
            for i in simplex:
                for j in simplex:
                    if i != j:
                        neighbors.add((min(i, j), max(i, j)))
        return neighbors
    except Exception:
        # Fallback to all pairs if Delaunay fails (e.g., collinear points)
        neighbors = set()
        for i in range(n):
            for j in range(i+1, n):
                neighbors.add((i, j))
        return neighbors

def _force_step(centers, radii, target_radii, neighbors, step_size=0.01, wall_stiffness=1.0, circle_stiffness=1.0):
    """
    Perform a single force-directed relaxation step.
    Forces:
    1. Push circles apart if overlapping (based on target radii).
    2. Push circles away from walls if penetrating (based on target radii).
    3. Pull circles towards contact if close (optional, for stability).
    """
    n = len(centers)
    forces = np.zeros_like(centers)
    
    # 1. Wall forces
    for i in range(n):
        r = target_radii[i]
        # Left
        if centers[i, 0] < r:
            forces[i, 0] += wall_stiffness * (r - centers[i, 0])
        # Right
        if centers[i, 0] > 1.0 - r:
            forces[i, 0] -= wall_stiffness * (centers[i, 0] - (1.0 - r))
        # Bottom
        if centers[i, 1] < r:
            forces[i, 1] += wall_stiffness * (r - centers[i, 1])
        # Top
        if centers[i, 1] > 1.0 - r:
            forces[i, 1] -= wall_stiffness * (centers[i, 1] - (1.0 - r))
            
    # 2. Circle-circle forces
    # Use neighbors from Delaunay for efficiency, but check all pairs if list is empty or small
    # Actually, Delaunay neighbors might miss some close pairs if geometry is distorted.
    # For n=26, O(N^2) is fast enough (~325 pairs). Let's just do all pairs for robustness.
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            d = np.sqrt(dx*dx + dy*dy)
            if d < 1e-8:
                # Perturb randomly if coincident
                dx = np.random.uniform(-0.01, 0.01)
                dy = np.random.uniform(-0.01, 0.01)
                d = np.sqrt(dx*dx + dy*dy)
            
            required_dist = target_radii[i] + target_radii[j]
            
            if d < required_dist:
                # Overlap: push apart
                overlap = required_dist - d
                # Force is proportional to overlap
                fx = (dx / d) * overlap * circle_stiffness
                fy = (dy / d) * overlap * circle_stiffness
                forces[i, 0] += fx
                forces[i, 1] += fy
                forces[j, 0] -= fx
                forces[j, 1] -= fy
            else:
                # No overlap: no force (or slight attraction to contact? 
                # Let's stick to repulsion only for stability, growth handles filling)
                pass
                
    # Apply forces with damping
    centers += step_size * forces
    
    # Clip to square to prevent flying out
    centers[:, 0] = np.clip(centers[:, 0], 0.0, 1.0)
    centers[:, 1] = np.clip(centers[:, 1], 0.0, 1.0)
    
    return centers

def _grow_and_relax(centers_init, radii_init, max_iter=2000, growth_factor=1.005):
    """
    Grow radii iteratively and relax positions.
    """
    n = len(radii_init)
    centers = centers_init.copy()
    radii = radii_init.copy()
    
    # Start with small radii to ensure valid start
    # Ensure initial validity
    for i in range(n):
        max_r = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        radii[i] = min(radii[i], max_r * 0.9)
        
    # Calculate max possible radius based on current positions
    def _max_possible_radius(centers, radii):
        max_r = 0.5
        for i in range(n):
            # Distance to walls
            d_wall = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
            # Distance to other circles
            for j in range(n):
                if i == j: continue
                d_c = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
                d_avail = d_c - radii[j]
                if d_avail < d_wall:
                    d_wall = d_avail
            if d_wall > max_r:
                max_r = d_wall
        return max_r

    best_sum = np.sum(radii)
    best_centers = centers.copy()
    best_radii = radii.copy()
    
    target_radii = radii.copy()
    
    for it in range(max_iter):
        if _remaining() < 2.0:
            break
            
        # Grow target radii
        # Scale all radii by growth factor, but cap by max possible
        # Simple strategy: uniform growth
        target_radii *= growth_factor
        
        # Cap target radii by local maximum possible radius
        # This is expensive O(N^2), but N=26 is small
        for i in range(n):
            # Distance to walls
            d_wall = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
            for j in range(n):
                if i == j: continue
                d_c = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
                d_avail = d_c - target_radii[j]
                if d_avail < d_wall:
                    d_wall = d_avail
            target_radii[i] = min(target_radii[i], d_wall)
            
        # Relax positions to fit target_radii
        # Use smaller steps as we get closer to solution
        step = 0.05 if it < 100 else 0.01
        for _ in range(50): # Inner relaxation iterations
            centers = _force_step(centers, radii, target_radii, set(), step_size=step, wall_stiffness=1.0, circle_stiffness=1.0)
            
        # Check if we can accept this state
        # If positions are valid for target_radii, update radii
        if _valid_check(centers, target_radii, tol=1e-5):
            radii = target_radii.copy()
            current_sum = np.sum(radii)
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = centers.copy()
                best_radii = radii.copy()
        else:
            # If invalid, shrink target_radii slightly to recover
            target_radii *= 0.95
            # Try to relax again with smaller target
            for _ in range(20):
                centers = _force_step(centers, radii, target_radii, set(), step_size=0.01, wall_stiffness=1.0, circle_stiffness=1.0)
            if _valid_check(centers, target_radii, tol=1e-5):
                radii = target_radii.copy()
            else:
                # Keep old radii, positions might have moved
                pass
                
    return best_centers, best_radii, best_sum

def _hex_initial(n_circles, jitter=0.0):
    """Generate a hexagonal packing initialization."""
    rows = 5
    cols = 6
    dx = 1.0 / (cols + 0.5)
    dy = 1.0 / (rows + 0.5)
    centers = []
    for r in range(rows):
        for c in range(cols):
            x = (c + 0.5) * dx + (0.5 * dx if r % 2 == 1 else 0)
            y = (r + 0.5) * dy
            if jitter > 0:
                x += np.random.uniform(-jitter, jitter)
                y += np.random.uniform(-jitter, jitter)
            centers.append((x, y))
    centers = centers[:n_circles]
    return np.array(centers)

def _random_initial(n_circles, seed=None):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    centers = []
    min_dist = 0.15
    attempts = 0
    while len(centers) < n_circles and attempts < 20000:
        attempts += 1
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        ok = True
        for cx, cy in centers:
            if (x - cx)**2 + (y - cy)**2 < min_dist**2:
                ok = False
                break
        if ok:
            centers.append((x, y))
    while len(centers) < n_circles:
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        centers.append((x, y))
    return np.array(centers)

def _error_repair(centers, radii):
    """
    Structural error repair: 
    1. Identify all violations.
    2. For each violation, shrink the radius of the circle that has the most 'space' 
       relative to its current radius (i.e., it's not constrained by walls/others tightly) 
       OR simply shrink the smaller circle.
    3. Move centers slightly to reduce overlap.
    """
    n = len(radii)
    centers = centers.copy()
    radii = radii.copy()
    
    # Iterative repair
    for _ in range(50):
        changed = False
        # Check walls
        for i in range(n):
            # Left
            if centers[i, 0] < radii[i]:
                radii[i] = max(0, centers[i, 0] * 0.99)
                changed = True
            # Right
            if centers[i, 0] > 1.0 - radii[i]:
                radii[i] = max(0, (1.0 - centers[i, 0]) * 0.99)
                changed = True
            # Bottom
            if centers[i, 1] < radii[i]:
                radii[i] = max(0, centers[i, 1] * 0.99)
                changed = True
            # Top
            if centers[i, 1] > 1.0 - radii[i]:
                radii[i] = max(0, (1.0 - centers[i, 1]) * 0.99)
                changed = True
        
        # Check overlaps
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                d = np.sqrt(dx*dx + dy*dy)
                req = radii[i] + radii[j]
                if d < req:
                    # Shrink the smaller one
                    if radii[i] < radii[j]:
                        radii[i] = max(0, (d - radii[j]) * 0.99)
                    else:
                        radii[j] = max(0, (d - radii[i]) * 0.99)
                    changed = True
                    
        if not changed:
            break
            
    return centers, radii

def construct():
    n = N
    best_centers = None
    best_radii = None
    best_sum = 0.0
    
    # Generate multiple initializations
    inits = []
    
    # Hexagonal
    for jitter in [0.0, 0.01, 0.02, 0.03]:
        c = _hex_initial(n, jitter=jitter)
        r = np.array([min(c[i,0], 1-c[i,0], c[i,1], 1-c[i,1]) * 0.8 for i in range(n)])
        inits.append((c, r))
        
    # Random
    for seed in [42, 123, 7, 99, 555, 1234, 4321, 777]:
        c = _random_initial(n, seed=seed)
        r = np.array([min(c[i,0], 1-c[i,0], c[i,1], 1-c[i,1]) * 0.7 for i in range(n)])
        inits.append((c, r))
        
    # Optimize from each start
    for idx, (c_init, r_init) in enumerate(inits):
        if _remaining() < 10.0:
            break
            
        centers, radii, total = _grow_and_relax(c_init, r_init, max_iter=500)
        
        # Apply error repair to ensure validity
        centers, radii = _error_repair(centers, radii)
        
        if _valid_check(centers, radii, tol=1e-5):
            total = np.sum(radii)
            if total > best_sum:
                best_sum = total
                best_centers = centers
                best_radii = radii
                
    # If no valid result found, fall back to simple grid
    if best_centers is None:
        side = int(np.ceil(np.sqrt(n)))
        dx = 1.0 / side
        dy = 1.0 / side
        centers = []
        radii = []
        for j in range(side):
            for i in range(side):
                if len(centers) >= n:
                    break
                x = (i + 0.5) * dx
                y = (j + 0.5) * dy
                r = 0.5 * dx - 1e-3
                centers.append((x, y))
                radii.append(r)
        best_centers = np.array(centers[:n])
        best_radii = np.array(radii[:n])
        best_sum = np.sum(best_radii)
        
    result = np.column_stack([best_centers, best_radii])
    return result

# EVOLVE-BLOCK-END
