"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _score(pts):
    """Compute the maximum radius for given centers."""
    n = len(pts)
    # Min distance between any two centers
    dmin = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < dmin:
                dmin = d
    # Min distance to walls
    wmin = np.inf
    for i in range(n):
        for k in range(3):
            w = 0.5 - abs(pts[i, k])
            if w < wmin:
                wmin = w
    return min(dmin / 2.0, wmin)


def _optimize(pts, max_iter=2000, step=0.05, deadline=None):
    """Local optimization: push spheres apart."""
    pts = pts.copy()
    best_score = _score(pts)
    best_pts = pts.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        n = len(pts)
        # Compute forces: each sphere is pushed away from others and toward center from walls
        forces = np.zeros_like(pts)
        
        for i in range(n):
            for j in range(i + 1, n):
                d = pts[i] - pts[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    # Random nudge for coincident points
                    d = np.random.randn(3) * 0.01
                    dist = np.linalg.norm(d)
                # Push apart if too close
                if dist < 2 * best_score + 0.1:
                    force = d / dist * (1.0 / (dist + 1e-6))
                    forces[i] += force
                    forces[j] -= force
            
            # Wall forces: push away from walls if too close
            for k in range(3):
                margin = 0.5 - abs(pts[i, k])
                if margin < best_score + 0.05:
                    if pts[i, k] > 0:
                        forces[i, k] -= 1.0 / (margin + 1e-6)
                    else:
                        forces[i, k] += 1.0 / (margin + 1e-6)
        
        # Normalize forces
        fnorm = np.linalg.norm(forces, axis=1, keepdims=True)
        fnorm = np.maximum(fnorm, 1e-10)
        forces = forces / fnorm
        
        # Update positions
        new_pts = pts + step * forces
        
        # Clamp to cube
        new_pts = np.clip(new_pts, -0.5, 0.5)
        
        # Check if valid and better
        new_score = _score(new_pts)
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            # Reduce step
            step *= 0.95
            if step < 1e-6:
                break
    
    return best_pts, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with a reasonable initial packing
    # Use a face-centered cubic-like arrangement
    np.random.seed(42)
    
    best_pts = None
    best_score = -1
    
    # Try multiple random restarts
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Generate initial positions: random with some structure
        # Use a jittered grid as starting point
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        
        if len(grid) < N:
            # Pad with random points
            extra = np.random.randn(N - len(grid), 3) * 0.1
            pts_init = np.vstack([grid, extra])
        else:
            pts_init = grid[:N].copy()
        
        # Add jitter
        pts_init = pts_init + np.random.randn(N, 3) * 0.02
        pts_init = np.clip(pts_init, -0.49, 0.49)
        
        # Optimize
        pts_opt, score = _optimize(pts_init, max_iter=1000, step=0.03, deadline=deadline)
        
        if score > best_score:
            best_score = score
            best_pts = pts_opt
    
    if best_pts is None:
        # Fallback: simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    return best_pts
# EVOLVE-BLOCK-END
