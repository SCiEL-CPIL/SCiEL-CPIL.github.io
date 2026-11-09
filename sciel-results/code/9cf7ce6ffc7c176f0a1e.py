"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import cKDTree

N = 100

# EVOLVE-BLOCK-START


def _score(pts):
    """Compute the packing radius for a set of points in the unit cube [-0.5, 0.5]^3."""
    # Check for non-finite values
    if not np.all(np.isfinite(pts)):
        return -1.0
    # Check boundaries
    if np.any(pts < -0.5 - 1e-12) or np.any(pts > 0.5 + 1e-12):
        return -1.0
    # Distance to walls
    wall_dist = np.min(np.minimum(0.5 - np.abs(pts), 0.5))
    # Pairwise distances
    tree = cKDTree(pts)
    pairs = tree.query_pairs(2.0)  # upper bound on distance
    if len(pairs) == 0:
        return wall_dist
    dists = []
    for i, j in pairs:
        d = np.linalg.norm(pts[i] - pts[j])
        dists.append(d)
    min_pair_dist = np.min(dists)
    return min(wall_dist, min_pair_dist / 2.0)


def _random_init(rng):
    """Generate random initial configuration."""
    return rng.uniform(-0.5, 0.5, size=(N, 3))


def _grid_init():
    """Generate grid-based initial configuration."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N]


def _hill_climb(pts, rng, max_iters=5000, time_limit=None, start_time=None):
    """
    Perform hill climbing to improve the packing.
    Uses repulsion-based moves.
    """
    if start_time is None:
        start_time = time.time()
    current_score = _score(pts)
    best_pts = pts.copy()
    best_score = current_score
    
    # Adaptive step size
    step_size = 0.05
    
    for it in range(max_iters):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        # Randomly pick a point to move
        idx = rng.randint(N)
        
        # Generate random displacement
        disp = rng.normal(0, step_size, size=3)
        
        # Clip to stay in bounds
        new_pts = pts.copy()
        new_pts[idx] = np.clip(pts[idx] + disp, -0.5, 0.5)
        
        new_score = _score(new_pts)
        
        if new_score > current_score:
            pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
            # Slightly increase step size on success
            step_size = min(step_size * 1.05, 0.2)
        else:
            # Slightly decrease step size on failure
            step_size = max(step_size * 0.95, 0.005)
    
    return best_pts, best_score


def _repulsion_optimize(pts, rng, max_iters=3000, time_limit=None, start_time=None):
    """
    Optimize using repulsion forces between spheres and walls.
    """
    if start_time is None:
        start_time = time.time()
    current_score = _score(pts)
    best_pts = pts.copy()
    best_score = current_score
    
    step_size = 0.02
    
    for it in range(max_iters):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        # Compute repulsion forces
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                d = pts[i] - pts[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    d = rng.normal(0, 0.001, size=3)
                    dist = np.linalg.norm(d)
                if dist < 0.3:  # Only interact when close
                    force = d / dist * (0.3 - dist)
                    forces[i] += force
                    forces[j] -= force
        
        # Wall repulsion
        for i in range(N):
            for k in range(3):
                if pts[i, k] > 0.2:
                    forces[i, k] -= (pts[i, k] - 0.2)
                elif pts[i, k] < -0.2:
                    forces[i, k] -= (pts[i, k] + 0.2)
        
        # Apply forces with random perturbation
        new_pts = pts.copy()
        for i in range(N):
            new_pts[i] = pts[i] + step_size * forces[i] + rng.normal(0, step_size * 0.5, size=3)
            new_pts[i] = np.clip(new_pts[i], -0.5, 0.5)
        
        new_score = _score(new_pts)
        
        if new_score >= current_score:
            pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
            step_size = min(step_size * 1.02, 0.1)
        else:
            step_size = max(step_size * 0.98, 0.005)
    
    return best_pts, best_score


def construct():
    """
    Construct the best packing of 100 equal spheres in a unit cube.
    Returns array of shape (N, 3) with centres.
    """
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    # Start with grid initialization
    pts = _grid_init()
    best_pts = pts.copy()
    best_score = _score(pts)
    
    rng = np.random.default_rng(42)
    
    # Try multiple restarts with different strategies
    for restart in range(20):
        if (time.time() - start_time) > time_limit:
            break
        
        # Alternate between random and perturbed grid starts
        if restart % 2 == 0:
            pts = _random_init(rng)
        else:
            pts = _grid_init()
            pts += rng.normal(0, 0.05, size=(N, 3))
            pts = np.clip(pts, -0.5, 0.5)
        
        # Hill climb
        pts, score = _hill_climb(pts, rng, max_iters=3000, time_limit=time_limit, start_time=start_time)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
        
        # Repulsion optimize
        pts, score = _repulsion_optimize(pts, rng, max_iters=2000, time_limit=time_limit, start_time=start_time)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
        
        # Final hill climb polish
        pts, score = _hill_climb(pts, rng, max_iters=2000, time_limit=time_limit, start_time=start_time)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    return best_pts


# EVOLVE-BLOCK-END
