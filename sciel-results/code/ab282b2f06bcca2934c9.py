"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin before 150s kill
    
    # Initial configuration: 4x4x4 grid, take first 50
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    pts = grid[:N].astype(np.float64).copy()
    
    def compute_score(pts):
        # min pairwise distance / 2
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist) / 2.0
        # wall clearance
        wall = np.min(0.5 - np.abs(pts), axis=1)
        min_wall = np.min(wall)
        return min(min_pair, min_wall)
    
    best = pts.copy()
    best_score = compute_score(best)
    
    # Repulsion-based relaxation
    # Each sphere repelled by others (1/d^2) and by walls
    n_iter = 200000
    step = 0.001
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
        
        # Compute forces
        forces = np.zeros_like(pts)
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Repulsion: force = diff / dist^3 (so magnitude 1/dist^2)
        # Only consider close pairs to save time? With 50, 50*49=2450 pairs, fine.
        inv_d3 = np.zeros_like(dist)
        mask = dist > 1e-12
        inv_d3[mask] = 1.0 / (dist[mask] ** 3)
        forces += np.sum(diff * inv_d3[:, :, None], axis=1)
        
        # Wall repulsion: if close to wall, push inward
        # Left wall (x=-0.5): if pts[i,0] < -0.5 + margin, push +x
        margin = 0.1
        for axis in range(3):
            # Left wall
            close_left = pts[:, axis] < -0.5 + margin
            if np.any(close_left):
                forces[close_left, axis] += 0.5 * (1.0 - (pts[close_left, axis] + 0.5) / margin)
            # Right wall
            close_right = pts[:, axis] > 0.5 - margin
            if np.any(close_right):
                forces[close_right, axis] -= 0.5 * (1.0 - (0.5 - pts[close_right, axis]) / margin)
        
        # Normalize forces to avoid large steps
        norms = np.linalg.norm(forces, axis=1)
        max_norm = np.max(norms) if np.max(norms) > 0 else 1.0
        forces = forces / max_norm * step
        
        pts += forces
        # Clamp to cube
        pts = np.clip(pts, -0.5 + 1e-9, 0.5 - 1e-9)
        
        # Check score periodically
        if it % 500 == 0:
            s = compute_score(pts)
            if s > best_score:
                best_score = s
                best = pts.copy()
    
    return best
# EVOLVE-BLOCK-END
