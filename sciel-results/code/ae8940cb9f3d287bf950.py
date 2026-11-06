"""Rewritten program: Lennard-Jones style optimization for 100 spheres in a unit cube."""
import numpy as np
import time

N = 100
CUBE_HALF = 0.5

def _score(pts):
    """Compute the radius score for a configuration of N points in [-0.5, 0.5]^3."""
    if pts.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Check inside cube
    if np.any(pts < -CUBE_HALF) or np.any(pts > CUBE_HALF):
        return 0.0
    # Compute min pair distance
    # Use scipy for efficient computation
    from scipy.spatial.distance import pdist
    d = pdist(pts, metric='euclidean')
    min_pair = np.min(d)
    # Min distance to walls
    wall_dists = np.min(np.abs(pts - CUBE_HALF), axis=1)  # min distance to any wall
    min_wall = np.min(wall_dists)
    r = min(min_pair / 2.0, min_wall)
    return r


def _random_init(seed):
    """Generate random initial points inside the cube with a small margin."""
    rng = np.random.RandomState(seed)
    margin = 0.02
    pts = rng.uniform(-CUBE_HALF + margin, CUBE_HALF - margin, size=(N, 3))
    return pts


def _lj_minimize(pts, max_iters=2000, lr_init=0.001, lr_decay=0.999, epsilon=0.001):
    """
    Minimize a Lennard-Jones-like potential with wall repulsion.
    The potential pushes points apart and away from walls.
    We use a smooth approximation to avoid singularities.
    """
    from scipy.spatial.distance import pdist, squareform
    
    pts = pts.copy()
    lr = lr_init
    
    for it in range(max_iters):
        # Compute pairwise distances
        d = pdist(pts, metric='euclidean')
        d_sq = d ** 2
        
        # Avoid zero distance
        d_sq_safe = np.maximum(d_sq, 1e-12)
        
        # LJ-like force: repulsive only (we only care about pushing apart)
        # Use a soft repulsion: F ~ 1/d^4 for d < some threshold
        # Gradient of (1/d^2) w.r.t. position is -2 * (r_vec) / d^4
        
        # Compute forces
        D = squareform(d_sq_safe)
        np.fill_diagonal(D, 0.0)
        
        # Force magnitude: proportional to 1/d^4, direction along r_vec
        # F_ij = (1/d^4) * (r_i - r_j) for repulsion
        # Sum over j for each i
        
        # Vectorized: for each pair (i,j), force on i is (1/d_ij^4) * (p_i - p_j)
        # We can compute this using broadcasting
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
        d_sq_full = np.sum(diff ** 2, axis=2)  # (N, N)
        np.fill_diagonal(d_sq_full, 1.0)  # avoid zero
        
        # Force = diff / d^4
        force_mag = 1.0 / (d_sq_full ** 2)  # (N, N)
        np.fill_diagonal(force_mag, 0.0)
        
        forces = np.sum(force_mag[:, :, np.newaxis] * diff, axis=1)  # (N, 3)
        
        # Wall repulsion: push away from walls
        # For each coordinate, if close to -0.5, push in + direction; if close to 0.5, push in - direction
        wall_force = np.zeros_like(pts)
        for k in range(3):
            # Distance to -0.5 wall
            d_left = pts[:, k] - (-CUBE_HALF)
            d_right = CUBE_HALF - pts[:, k]
            # Repulsion: strong when close
            # Use smooth: 1/d^2 for d < 0.1
            d_left_safe = np.maximum(d_left, 0.01)
            d_right_safe = np.maximum(d_right, 0.01)
            wall_force[:, k] += 0.5 / d_left_safe ** 2  # push right
            wall_force[:, k] -= 0.5 / d_right_safe ** 2  # push left
        
        total_forces = forces + wall_force
        
        # Update
        pts += lr * total_forces
        
        # Clamp to cube
        pts = np.clip(pts, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)
        
        # Decay learning rate
        lr *= lr_decay
        
        # Check convergence
        if it % 100 == 0 and it > 0:
            current_score = _score(pts)
    
    return pts


def _pairwise_push(pts, max_iters=1000, epsilon=0.001):
    """
    Greedy pairwise push: repeatedly find the closest pair and push them apart.
    Also handle wall violations.
    """
    from scipy.spatial.distance import pdist, cdist
    
    pts = pts.copy()
    
    for it in range(max_iters):
        # Check for wall violations
        for k in range(3):
            # Points too close to left wall
            mask_left = pts[:, k] < -CUBE_HALF + 1e-9
            if np.any(mask_left):
                pts[mask_left, k] = -CUBE_HALF + 1e-9
            # Points too close to right wall
            mask_right = pts[:, k] > CUBE_HALF - 1e-9
            if np.any(mask_right):
                pts[mask_right, k] = CUBE_HALF - 1e-9
        
        # Find closest pair
        d = pdist(pts, metric='euclidean')
        min_idx = np.argmin(d)
        min_dist = d[min_idx]
        
        # Convert min_idx to pair indices
        n = N
        i = int((np.sqrt(1 + 8 * min_idx) - 1) / 2)
        j = int(min_idx - i * (i + 1) / 2)
        i, j = int(i), int(j)
        
        if min_dist < epsilon:
            # Push apart
            diff = pts[i] - pts[j]
            norm = np.linalg.norm(diff)
            if norm < 1e-12:
                # Random direction
                diff = np.random.randn(3)
                norm = np.linalg.norm(diff)
            diff = diff / norm
            push = (epsilon - min_dist) / 2.0 + 0.001
            pts[i] += push * diff
            pts[j] -= push * diff
            # Clamp
            pts = np.clip(pts, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)
        else:
            break
    
    return pts


def construct():
    """Construct N sphere centers in the unit cube to maximize the minimum radius."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_pts = None
    best_score = 0.0
    
    # Strategy: multiple random starts, each followed by LJ minimization + pairwise push
    seeds = [42, 123, 456, 789, 1024, 2048, 3141, 5678, 9999, 12345]
    
    for seed in seeds:
        if time.time() > deadline:
            break
        
        # Random initialization
        pts = _random_init(seed)
        
        # LJ minimization
        pts = _lj_minimize(pts, max_iters=1500, lr_init=0.0005, lr_decay=0.9995)
        
        # Pairwise push to resolve close pairs
        pts = _pairwise_push(pts, max_iters=500, epsilon=0.005)
        
        # Another round of LJ with smaller lr
        pts = _lj_minimize(pts, max_iters=1000, lr_init=0.0001, lr_decay=0.9999)
        
        # Final pairwise push
        pts = _pairwise_push(pts, max_iters=300, epsilon=0.001)
        
        score = _score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
        
        # Check time
        if time.time() > deadline:
            break
    
    # If we haven't found anything good, use a simple grid as fallback
    if best_pts is None or best_score < 0.05:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_pts = grid[:N]
        best_score = _score(best_pts)
    
    return best_pts
