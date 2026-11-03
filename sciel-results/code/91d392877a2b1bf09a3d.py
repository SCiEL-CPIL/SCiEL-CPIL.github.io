import numpy as np
from scipy.spatial.distance import cdist
import time

N = 200
_TIME_LIMIT = 145  # Leave 5s margin for 150s kill


def _score(pts):
    """Compute the maximum radius for a set of N points in [-0.5, 0.5]^2."""
    if pts.shape[0] != N or not np.all(np.isfinite(pts)):
        return 0.0
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        return 0.0
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D)
    wall = np.min([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1]),
    ])
    r = min(min_pair / 2.0, wall)
    return r if r > 0 else 0.0


def _initial_hex_grid(nrows, ncols, jitter=0.0):
    """Generate a hexagonal lattice of nrows*ncols points, then pick N closest to origin."""
    pts = []
    for i in range(nrows):
        y = -0.45 + i * (0.9 / max(nrows - 1, 1))
        shift = (0.5 * 0.9 / max(ncols - 1, 1)) if (i % 2 == 1) else 0.0
        for j in range(ncols):
            x = -0.45 + j * (0.9 / max(ncols - 1, 1)) + shift
            if jitter > 0:
                x += np.random.uniform(-jitter, jitter)
                y += np.random.uniform(-jitter, jitter)
            pts.append([x, y])
    pts = np.array(pts)
    if len(pts) > N:
        d2 = np.sum(pts ** 2, axis=1)
        idx = np.argsort(d2)[:N]
        pts = pts[idx]
    elif len(pts) < N:
        # Pad with random points
        extra = np.random.uniform(-0.4, 0.4, size=(N - len(pts), 2))
        pts = np.vstack([pts, extra])
    return pts


def _simulate(pts, n_steps, dt, k_rep, k_wall, damping):
    """
    Rigid-body dynamics: repulsive forces between circles + wall repulsion.
    pts: (N, 2) array of positions
    Returns final positions after n_steps of integration.
    """
    pts = pts.copy()
    vel = np.zeros_like(pts)
    
    for step in range(n_steps):
        # Compute pairwise forces
        D = cdist(pts, pts)
        # Avoid division by zero
        D[D < 1e-10] = 1e-10
        
        # Repulsive force: F = k_rep / d^2 along direction
        # Vectorized: for each pair (i,j), force on i from j
        # diff[i,j] = pts[j] - pts[i]
        diff = pts[None, :, :] - pts[:, None, :]  # (N, N, 2)
        # Force magnitude on i from j: k_rep / D[i,j]^2
        # Direction: normalize diff[i,j]
        norms = np.sqrt(np.sum(diff ** 2, axis=2))  # (N, N)
        norms[norms < 1e-10] = 1e-10
        
        # Unit vectors
        unit = diff / norms[:, :, None]
        
        # Force magnitude: k_rep / d^2, but only if d < some cutoff
        # Use inverse square: F = k_rep / d^2
        force_mag = k_rep / (norms ** 2)
        # Zero out self-interaction
        np.fill_diagonal(force_mag, 0.0)
        
        # Total force on each particle: sum over j of force_mag[i,j] * unit[i,j]
        forces = np.sum(force_mag[:, :, None] * unit, axis=1)  # (N, 2)
        
        # Wall repulsion: strong force pushing particles back inside
        # For each coordinate, if close to wall, add force
        # Left wall: x < -0.5 + margin -> push right
        # Right wall: x > 0.5 - margin -> push left
        # Similarly for y
        
        wall_margin = 0.05
        wall_force = np.zeros_like(pts)
        
        # X walls
        left_dist = pts[:, 0] - (-0.5)
        right_dist = 0.5 - pts[:, 0]
        
        # Strong repulsion when close to wall
        # Use exponential or polynomial repulsion
        # If left_dist < wall_margin, push right with force proportional to (wall_margin - left_dist)^2
        mask_left = left_dist < wall_margin
        if np.any(mask_left):
            wall_force[mask_left, 0] += k_wall * (wall_margin - left_dist[mask_left]) ** 2
        
        mask_right = right_dist < wall_margin
        if np.any(mask_right):
            wall_force[mask_right, 0] -= k_wall * (wall_margin - right_dist[mask_right]) ** 2
        
        # Y walls
        bottom_dist = pts[:, 1] - (-0.5)
        top_dist = 0.5 - pts[:, 1]
        
        mask_bottom = bottom_dist < wall_margin
        if np.any(mask_bottom):
            wall_force[mask_bottom, 1] += k_wall * (wall_margin - bottom_dist[mask_bottom]) ** 2
        
        mask_top = top_dist < wall_margin
        if np.any(mask_top):
            wall_force[mask_top, 1] -= k_wall * (wall_margin - top_dist[mask_top]) ** 2
        
        # Update velocity with damping
        vel = vel * damping + (forces + wall_force) * dt
        
        # Update position
        pts = pts + vel * dt
        
        # Hard clamp to stay within bounds (prevent escape)
        pts[:, 0] = np.clip(pts[:, 0], -0.499, 0.499)
        pts[:, 1] = np.clip(pts[:, 1], -0.499, 0.499)
        
        # Check time
        if time.time() > _TIME_LIMIT - 5:
            break
    
    return pts


def _local_refine(pts, n_iters=500, lr=1e-4):
    """
    Gradient-free local refinement: iteratively push apart the closest pairs
    and pull inward if too close to walls.
    """
    pts = pts.copy()
    for it in range(n_iters):
        if time.time() > _TIME_LIMIT - 5:
            break
        
        D = cdist(pts, pts)
        np.fill_diagonal(D, np.inf)
        
        # Find the minimum distance pair
        min_idx = np.unravel_index(np.argmin(D), D.shape)
        i, j = min_idx
        d = D[i, j]
        
        if d < 1e-10:
            # Coincident points: separate them
            pts[i, 0] += 0.001
            pts[j, 0] -= 0.001
            continue
        
        # Push apart the closest pair
        diff = pts[j] - pts[i]
        norm = np.linalg.norm(diff)
        unit = diff / norm
        
        # Also check wall distances
        wall_dists = np.array([
            pts[:, 0] + 0.5,
            0.5 - pts[:, 0],
            pts[:, 1] + 0.5,
            0.5 - pts[:, 1],
        ])
        min_wall_idx = np.argmin(wall_dists)
        min_wall_val = np.min(wall_dists)
        
        # If the limiting constraint is a pair, push them apart
        # If it's a wall, push that particle inward
        if min_pair_limit(d / 2.0, min_wall_val):
            # Push pair apart
            step = lr * (1.0 / d)
            pts[i] -= step * unit
            pts[j] += step * unit
        else:
            # Find which particle is closest to the limiting wall
            # min_wall_idx: 0=left, 1=right, 2=bottom, 3=top
            if min_wall_idx == 0:  # left wall: min of pts[:,0] + 0.5
                wall_particle_idx = np.argmin(pts[:, 0] + 0.5)
                pts[wall_particle_idx, 0] += lr
            elif min_wall_idx == 1:  # right wall: min of 0.5 - pts[:,0]
                wall_particle_idx = np.argmin(0.5 - pts[:, 0])
                pts[wall_particle_idx, 0] -= lr
            elif min_wall_idx == 2:  # bottom wall: min of pts[:,1] + 0.5
                wall_particle_idx = np.argmin(pts[:, 1] + 0.5)
                pts[wall_particle_idx, 1] += lr
            else:  # top wall: min of 0.5 - pts[:,1]
                wall_particle_idx = np.argmin(0.5 - pts[:, 1])
                pts[wall_particle_idx, 1] -= lr
        
        # Clamp
        pts[:, 0] = np.clip(pts[:, 0], -0.499, 0.499)
        pts[:, 1] = np.clip(pts[:, 1], -0.499, 0.499)
    
    return pts


def min_pair_limit(pair_r, wall_r):
    """Return True if the pair distance is the limiting constraint."""
    return pair_r <= wall_r


def construct():
    t0 = time.time()
    best_r = 0.0
    best_pts = None
    
    # Try multiple initial configurations
    configs = [
        (14, 15, 0.0),   # hex grid, no jitter
        (15, 14, 0.0),
        (14, 15, 0.01),  # hex grid with jitter
        (15, 14, 0.01),
        (13, 16, 0.0),
        (16, 13, 0.0),
    ]
    
    for nrows, ncols, jitter in configs:
        if time.time() - t0 > _TIME_LIMIT - 20:
            break
        
        # Generate initial points
        pts = _initial_hex_grid(nrows, ncols, jitter)
        
        # Run dynamics simulation
        # Parameters: n_steps, dt, k_rep, k_wall, damping
        n_steps = 2000
        dt = 0.001
        k_rep = 0.001
        k_wall = 1.0
        damping = 0.95
        
        pts_sim = _simulate(pts, n_steps, dt, k_rep, k_wall, damping)
        
        r = _score(pts_sim)
        if r > best_r:
            best_r = r
            best_pts = pts_sim.copy()
        
        # Local refinement
        pts_ref = _local_refine(pts_sim, n_iters=300, lr=1e-4)
        r_ref = _score(pts_ref)
        if r_ref > best_r:
            best_r = r_ref
            best_pts = pts_ref.copy()
    
    # If dynamics didn't find anything good, try a simpler approach
    if best_pts is None or best_r < 0.03:
        # Fallback: simple hex grid
        pts = _initial_hex_grid(14, 15, 0.0)
        r = _score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts
    
    if best_pts is None:
        # Last resort: uniform grid
        rows, cols = 14, 15
        xs = np.linspace(-0.5 + 0.5/cols, 0.5 - 0.5/cols, cols)
        ys = np.linspace(-0.5 + 0.5/rows, 0.5 - 0.5/rows, rows)
        grid = np.array([(x, y) for y in ys for x in xs])
        best_pts = grid[:N]
    
    return best_pts
