"""Rewritten program: circle packing via gradient descent with multi-start and simulated annealing."""
import time
import numpy as np
from scipy.spatial.distance import cdist

N = 200
DEADLINE = 150.0  # seconds wall-clock limit


def _compute_radius(pts):
    """Compute the maximum common radius for the given centres."""
    # Distance to boundary: min over coordinates of (0.5 - |coord|)
    boundary_min = np.min(0.5 - np.abs(pts))
    # Pairwise distances
    d = cdist(pts, pts)
    # Zero out diagonal
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    r = min(min_pair, 2.0 * boundary_min) / 2.0
    return r


def _hex_init(n, rng):
    """Initialize n points in a hexagonal lattice pattern inside the unit square."""
    # Approximate hex packing density: sqrt(3)/2 * (2r)^2 per circle
    # For N circles in unit square, roughly r ~ 0.5 / sqrt(N) * correction
    # Use a hex grid with spacing s
    # In a hex lattice, each point has 6 neighbors at distance s
    # Area per point in hex = s^2 * sqrt(3)/2
    # N * s^2 * sqrt(3)/2 <= 1  =>  s <= sqrt(2 / (N * sqrt(3)))
    s = np.sqrt(2.0 / (N * np.sqrt(3.0)))
    # Generate hex lattice points
    dx = s
    dy = s * np.sqrt(3.0) / 2.0
    rows = int(np.ceil(1.0 / dy)) + 2
    cols = int(np.ceil(1.0 / dx)) + 2
    pts = []
    for i in range(rows):
        y = -0.5 + i * dy
        offset = 0.0 if i % 2 == 0 else dx / 2.0
        for j in range(cols):
            x = -0.5 + j * dx + offset
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                pts.append((x, y))
    pts = np.array(pts)
    if len(pts) >= n:
        # Randomly select n points
        idx = rng.choice(len(pts), size=n, replace=False)
        return pts[idx]
    else:
        # Fallback: use all and pad with random
        result = np.empty((n, 2))
        result[:len(pts)] = pts
        result[len(pts):] = rng.uniform(-0.45, 0.45, size=(n - len(pts), 2))
        return result


def _random_init(n, rng):
    """Random initialization inside the unit square."""
    return rng.uniform(-0.45, 0.45, size=(n, 2))


def _gradient_step(pts, lr, rng):
    """One gradient descent step to maximize minimum distance."""
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    
    # Find the closest pair
    min_idx = np.unravel_index(np.argmin(d), d.shape)
    i, j = min_idx
    
    # Also consider boundary proximity
    boundary_dists = 0.5 - np.abs(pts)
    min_boundary = np.min(boundary_dists)
    
    # Gradient: push points apart
    grad = np.zeros_like(pts)
    
    # For the closest pair, push them apart
    diff = pts[i] - pts[j]
    dist = np.linalg.norm(diff)
    if dist > 1e-12:
        direction = diff / dist
        grad[i] += direction
        grad[j] -= direction
    else:
        # Random direction if coincident
        direction = rng.normal(size=2)
        direction /= np.linalg.norm(direction)
        grad[i] += direction
        grad[j] -= direction
    
    # For boundary: push points inward if too close to boundary
    for k in range(len(pts)):
        for dim in range(2):
            if pts[k, dim] > 0:
                if 0.5 - pts[k, dim] < 0.05:
                    grad[k, dim] -= 1.0
            else:
                if pts[k, dim] + 0.5 < 0.05:
                    grad[k, dim] += 1.0
    
    # Apply gradient step with learning rate
    pts_new = pts + lr * grad
    
    # Clip to stay inside square with small margin
    pts_new = np.clip(pts_new, -0.499, 0.499)
    
    return pts_new


def _simulated_annealing(pts, rng, time_limit):
    """Simulated annealing to maximize packing radius."""
    current_pts = pts.copy()
    current_r = _compute_radius(current_pts)
    best_pts = current_pts.copy()
    best_r = current_r
    
    t0 = time.time()
    temp = 0.01
    cooling = 0.9995
    lr = 0.005
    
    while time.time() - t0 < time_limit:
        # Generate a random perturbation
        perturbation = rng.normal(0, temp, size=(len(current_pts), 2))
        new_pts = current_pts + perturbation
        new_pts = np.clip(new_pts, -0.499, 0.499)
        
        new_r = _compute_radius(new_pts)
        
        # Accept if better, or with probability exp((new_r - current_r) / temp)
        if new_r >= current_r or rng.random() < np.exp((new_r - current_r) / temp):
            current_pts = new_pts
            current_r = new_r
            if new_r > best_r:
                best_pts = new_pts.copy()
                best_r = new_r
        
        # Cool down
        temp *= cooling
        if temp < 1e-6:
            temp = 1e-6
    
    return best_pts, best_r


def construct():
    """Construct 200 circle centres in the unit square maximizing common radius."""
    start_time = time.time()
    deadline = start_time + DEADLINE - 5.0  # 5 second safety margin
    
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_r = -np.inf
    
    # Multi-start with different initializations
    n_starts = 8
    
    for start_idx in range(n_starts):
        if time.time() > deadline:
            break
        
        # Alternate between hex and random initialization
        if start_idx % 2 == 0:
            pts = _hex_init(N, rng)
        else:
            pts = _random_init(N, rng)
        
        # Initial gradient descent phase
        gd_time = min(10.0, (deadline - time.time()) / (n_starts - start_idx))
        if gd_time <= 0:
            break
        
        t0 = time.time()
        lr = 0.01
        for step in range(1000):
            if time.time() - t0 > gd_time:
                break
            pts = _gradient_step(pts, lr, rng)
            lr *= 0.999
        
        # Compute radius after gradient descent
        r = _compute_radius(pts)
        
        # Simulated annealing phase
        sa_time = min(5.0, (deadline - time.time()) / (n_starts - start_idx))
        if sa_time > 1.0:
            pts, r = _simulated_annealing(pts, rng, sa_time)
        
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    # If no valid result, return a simple grid
    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return best_pts
