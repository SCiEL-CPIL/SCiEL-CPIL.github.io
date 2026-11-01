"""Optimized packing of 50 equal circles in a unit square."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 115.0  # seconds, safety margin before 120s limit


def _compute_radius(pts):
    """Compute the maximum valid radius for the given centers."""
    if len(pts) == 0:
        return 0.0
    pts = np.asarray(pts, dtype=np.float64)
    # Wall distances: min over all points of min(0.5 - x, 0.5 - y, 0.5 + x, 0.5 + y)
    wall_min = min(
        np.min(0.5 - pts[:, 0]),
        np.min(0.5 - pts[:, 1]),
        np.min(0.5 + pts[:, 0]),
        np.min(0.5 + pts[:, 1]),
    )
    # Pairwise distances
    tree = cKDTree(pts)
    # Use query_pairs to get all pairs within a reasonable distance
    # For 50 points, just compute all pairwise distances
    from scipy.spatial.distance import pdist
    dists = pdist(pts, metric='euclidean')
    if len(dists) > 0:
        pair_min = np.min(dists) / 2.0
    else:
        pair_min = np.inf
    return min(wall_min, pair_min)


def _soft_violation(pts, r_target):
    """
    Compute soft violation: sum of squared overlaps with walls and between circles.
    Returns (violation, gradient) for use in optimization.
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    violation = 0.0
    grad = np.zeros_like(pts)
    
    # Wall constraints: for each point, distance to each wall must be >= r_target
    # Left wall: x >= -0.5 + r_target => violation if x < -0.5 + r_target
    # Right wall: x <= 0.5 - r_target
    # Bottom wall: y >= -0.5 + r_target
    # Top wall: y <= 0.5 - r_target
    
    # Left
    v = r_target + 0.5 - pts[:, 0]
    mask = v > 0
    if np.any(mask):
        violation += np.sum(v[mask] ** 2)
        grad[mask, 0] -= 2 * v[mask]
    
    # Right
    v = pts[:, 0] - (0.5 - r_target)
    mask = v > 0
    if np.any(mask):
        violation += np.sum(v[mask] ** 2)
        grad[mask, 0] += 2 * v[mask]
    
    # Bottom
    v = r_target + 0.5 - pts[:, 1]
    mask = v > 0
    if np.any(mask):
        violation += np.sum(v[mask] ** 2)
        grad[mask, 1] -= 2 * v[mask]
    
    # Top
    v = pts[:, 1] - (0.5 - r_target)
    mask = v > 0
    if np.any(mask):
        violation += np.sum(v[mask] ** 2)
        grad[mask, 1] += 2 * v[mask]
    
    # Pairwise constraints: |p_i - p_j| >= 2*r_target
    for i in range(n):
        for j in range(i + 1, n):
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                # Perturb
                d = np.array([1e-8, 0.0])
                dist = 1e-8
            required = 2 * r_target
            if dist < required:
                v = required - dist
                violation += v ** 2
                # Gradient: d(v^2)/dp_i = 2*v * d(required - dist)/dp_i
                # d(dist)/dp_i = d/dist (where d = p_i - p_j)
                # d(required - dist)/dp_i = -d/dist
                grad[i] -= 2 * v * d / dist
                grad[j] += 2 * v * d / dist
    
    return violation, grad


def _hex_init():
    """Hexagonal close-packed initial configuration."""
    # Start with a hexagonal lattice, then clip to the square
    # For 50 circles, try different hexagonal arrangements
    
    best_pts = None
    best_radius = -1.0
    
    # Try different hexagonal densities
    for rows in range(5, 10):
        for cols in range(5, 10):
            if rows * cols < N:
                continue
            # Hexagonal spacing
            # In a hexagonal lattice, horizontal spacing = 2r, vertical spacing = 2r * sqrt(3)/2 = r*sqrt(3)
            # Let's parameterize by the horizontal spacing
            # We want to fit rows x cols in the square [-0.5, 0.5]^2
            # Width needed: (cols-1) * dx + 2r <= 1
            # Height needed: (rows-1) * dy + 2r <= 1
            # Where dx = 2r, dy = r*sqrt(3)
            
            # Try to maximize r
            # (cols-1)*2r + 2r <= 1 => r * (2*cols) <= 1 => r <= 1/(2*cols)
            # (rows-1)*r*sqrt(3) + 2r <= 1 => r * (sqrt(3)*(rows-1) + 2) <= 1 => r <= 1/(sqrt(3)*(rows-1)+2)
            
            r_x = 1.0 / (2.0 * cols)
            r_y = 1.0 / (np.sqrt(3) * (rows - 1) + 2)
            r = min(r_x, r_y)
            
            dx = 2 * r
            dy = r * np.sqrt(3)
            
            # Generate hexagonal lattice points
            pts = []
            for row in range(rows):
                for col in range(cols):
                    x = -0.5 + r + col * dx + (row % 2) * dx / 2
                    y = -0.5 + r + row * dy
                    # Center the pattern
                    if row % 2 == 1:
                        x -= dx / 2  # Shift odd rows
                    pts.append([x, y])
            
            pts = np.array(pts[:N])
            
            # Check if all points are within the square
            if np.all(pts[:, 0] >= -0.5 + r - 1e-9) and np.all(pts[:, 0] <= 0.5 - r + 1e-9) and \
               np.all(pts[:, 1] >= -0.5 + r - 1e-9) and np.all(pts[:, 1] <= 0.5 - r + 1e-9):
                rad = _compute_radius(pts)
                if rad > best_radius:
                    best_radius = rad
                    best_pts = pts.copy()
    
    if best_pts is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = grid[:N].copy()
        best_radius = _compute_radius(best_pts)
    
    return best_pts, best_radius


def _simulated_annealing(pts, r_target, time_budget=30.0):
    """
    Simulated annealing to find a configuration where all circles of radius r_target
    fit without overlap. Returns the best feasible configuration found.
    """
    pts = np.array(pts, dtype=np.float64).copy()
    n = len(pts)
    
    start_time = time.time()
    current_violation, _ = _soft_violation(pts, r_target)
    best_pts = pts.copy()
    best_violation = current_violation
    
    T = 0.01
    T_min = 1e-8
    alpha = 0.999
    steps = 0
    max_steps = 500000
    
    while time.time() - start_time < time_budget and steps < max_steps:
        steps += 1
        # Random perturbation
        idx = np.random.randint(n)
        delta = np.random.randn(2) * T * 2
        new_pts = pts.copy()
        new_pts[idx] += delta
        
        # Keep within bounds
        new_pts[:, 0] = np.clip(new_pts[:, 0], -0.5 + r_target, 0.5 - r_target)
        new_pts[:, 1] = np.clip(new_pts[:, 1], -0.5 + r_target, 0.5 - r_target)
        
        new_violation, _ = _soft_violation(new_pts, r_target)
        
        if new_violation < current_violation or np.random.rand() < np.exp(-(new_violation - current_violation) / max(T, 1e-12)):
            pts = new_pts
            current_violation = new_violation
            if current_violation < best_violation:
                best_violation = current_violation
                best_pts = pts.copy()
        
        T *= alpha
        if T < T_min:
            T = T_min
    
    return best_pts, best_violation


def _lbfgs_polish(pts, r_target, time_budget=10.0):
    """
    Use L-BFGS to minimize the soft violation for a given target radius.
    """
    pts = np.array(pts, dtype=np.float64).copy()
    n = len(pts)
    
    def objective_and_grad(flat_pts):
        p = flat_pts.reshape(n, 2)
        v, g = _soft_violation(p, r_target)
        return v, g
    
    start_time = time.time()
    try:
        result = minimize(
            objective_and_grad,
            pts.flatten(),
            method='L-BFGS-B',
            bounds=[(-0.5 + r_target, 0.5 - r_target)] * (2 * n),
            options={'maxiter': 200, 'maxfun': 2000, 'disp': False},
        )
        if time.time() - start_time > time_budget:
            return pts  # Return original if timeout
        new_pts = result.x.reshape(n, 2)
        v_new, _ = _soft_violation(new_pts, r_target)
        v_old, _ = _soft_violation(pts, r_target)
        if v_new < v_old:
            return new_pts
    except Exception:
        pass
    
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Step 1: Get a good initial configuration
    pts, init_radius = _hex_init()
    best_pts = pts.copy()
    best_radius = init_radius
    
    # Step 2: Binary search on radius with optimization
    # The known best is ~0.071377103865
    # Start with a reasonable range
    r_low = best_radius * 0.95
    r_high = 0.075
    
    # First, try to improve the initial radius via local optimization
    for attempt in range(5):
        if time.time() > deadline - 30:
            break
        # Try to polish the current configuration
        polished = _lbfgs_polish(best_pts, best_radius, time_budget=5.0)
        new_r = _compute_radius(polished)
        if new_r > best_radius:
            best_pts = polished
            best_radius = new_r
    
    # Step 3: Binary search for maximum radius
    # We'll do a few iterations of binary search
    for _ in range(8):
        if time.time() > deadline - 20:
            break
        
        r_mid = (r_low + r_high) / 2
        
        # Try to find a configuration for radius r_mid
        # Start from best_pts and do SA + L-BFGS
        candidate = best_pts.copy()
        
        # SA to try to fit r_mid
        sa_pts, sa_violation = _simulated_annealing(candidate, r_mid, time_budget=8.0)
        
        # Polish with L-BFGS
        polished = _lbfgs_polish(sa_pts, r_mid, time_budget=5.0)
        
        actual_r = _compute_radius(polished)
        
        if actual_r >= r_mid - 1e-10:
            # Feasible
            r_low = r_mid
            if actual_r > best_radius:
                best_radius = actual_r
                best_pts = polished.copy()
        else:
            r_high = r_mid
    
    # Final polish
    if time.time() < deadline - 10:
        final_polished = _lbfgs_polish(best_pts, best_radius, time_budget=8.0)
        final_r = _compute_radius(final_polished)
        if final_r > best_radius:
            best_radius = final_r
            best_pts = final_polished
    
    # Ensure validity: clip to square
    best_pts[:, 0] = np.clip(best_pts[:, 0], -0.5, 0.5)
    best_pts[:, 1] = np.clip(best_pts[:, 1], -0.5, 0.5)
    
    return best_pts
