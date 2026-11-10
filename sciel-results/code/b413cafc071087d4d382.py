"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def _gen_hex_lattice(N, r):
    """Generate hex lattice points inside unit disk, return first N."""
    spacing = 2.0 * r
    k = int(1.0 / spacing) + 2
    pts = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - r:
                pts.append((x, y))
    pts = np.array(pts)
    if len(pts) < N:
        return None
    # Sort by distance from center, take N closest
    dists = np.hypot(pts[:, 0], pts[:, 1])
    idx = np.argsort(dists, kind="stable")[:N]
    return pts[idx]

def _min_dist(pts):
    """Compute minimum pairwise distance using vectorized approach."""
    n = len(pts)
    min_d = np.inf
    for i in range(n):
        d = np.hypot(pts[i+1:, 0] - pts[i, 0], pts[i+1:, 1] - pts[i, 1])
        if d.size > 0:
            min_d = min(min_d, np.min(d))
    return min_d

def _boundary_violation(pts, r):
    """Check if any point violates the boundary constraint."""
    dists = np.hypot(pts[:, 0], pts[:, 1])
    return np.max(dists) - (1.0 - r)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 130s deadline, 20s safety margin
    
    # Start with hex lattice
    r_init = 0.052
    pts = _gen_hex_lattice(N, r_init)
    if pts is None:
        # Fallback: simple grid
        pts = np.zeros((N, 2))
        for i in range(N):
            angle = 2 * np.pi * i / N
            rad = 0.5 * (i / N)
            pts[i, 0] = rad * np.cos(angle)
            pts[i, 1] = rad * np.sin(angle)
    
    best_pts = pts.copy()
    best_r = _min_dist(best_pts) / 2.0
    best_r = min(best_r, 1.0 - np.max(np.hypot(best_pts[:, 0], best_pts[:, 1])))
    
    # Force-directed optimization
    lr = 0.001
    decay = 0.995
    min_lr = 1e-6
    
    while time.time() < deadline and lr > min_lr:
        # Compute forces
        forces = np.zeros_like(pts)
        n = len(pts)
        
        # Pairwise repulsion
        for i in range(n):
            d = pts[i+1:] - pts[i]
            dist = np.hypot(d[:, 0], d[:, 1])
            # Avoid division by zero
            valid = dist > 1e-10
            if np.any(valid):
                # Force proportional to 1/dist^2, pushing apart
                f = d[valid] / (dist[valid] ** 2 + 1e-10)[:, np.newaxis]
                forces[i] -= np.sum(f, axis=0)
                forces[i+1:][valid] += f
        
        # Boundary constraint: push inward if too close to edge
        dists = np.hypot(pts[:, 0], pts[:, 1])
        margin = 1.0 - dists
        # If margin < r, push inward
        too_close = margin < best_r
        if np.any(too_close):
            inward = -pts[too_close] / dists[too_close, np.newaxis]
            forces[too_close] += inward * 0.5
        
        # Update positions
        pts += lr * forces
        
        # Clamp to stay inside boundary
        dists = np.hypot(pts[:, 0], pts[:, 1])
        too_far = dists > 1.0 - best_r * 0.5
        if np.any(too_far):
            pts[too_far] *= (1.0 - best_r * 0.5) / dists[too_far, np.newaxis]
        
        # Evaluate
        curr_min_d = _min_dist(pts) / 2.0
        curr_r = min(curr_min_d, 1.0 - np.max(np.hypot(pts[:, 0], pts[:, 1])))
        
        if curr_r > best_r:
            best_r = curr_r
            best_pts = pts.copy()
            lr = max(lr * 1.1, 0.01)  # Increase learning rate on improvement
        else:
            lr *= decay
        
        # Early exit if no improvement for many iterations
        if time.time() > deadline - 5.0:
            break
    
    return best_pts
# EVOLVE-BLOCK-END
