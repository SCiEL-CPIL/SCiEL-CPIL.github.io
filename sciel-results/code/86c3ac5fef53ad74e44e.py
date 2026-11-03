"""Pack 50 equal circles in unit disk — ring seed + SA/gradient refinement."""
import time
import numpy as np
from math import sqrt, cos, sin, pi

N = 50
T_START = time.time()
TIME_LIMIT = 115.0  # safety margin before 120s kill

def _time_left():
    return TIME_LIMIT - (time.time() - T_START)

def _score(pts):
    """Return the achievable common radius for a placement."""
    # boundary constraint
    r_bound = 1.0 - np.linalg.norm(pts, axis=1)
    if np.any(r_bound < -1e-12):
        return -1.0
    # pairwise
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    r_pair = dist.min(axis=1) / 2.0
    return float(min(r_bound.min(), r_pair.min()))

def _build_ring_seed():
    """Build the known best-known packing structure: center + concentric rings."""
    # Known best for n=50: 1 center, then rings.
    # Common good structure: 1, 6, 12, 18, 13 = 50
    # Let's use a well-known configuration and let SA optimize positions.
    # Start with rings at approximate optimal radii.
    rings = [(6, 0.22), (12, 0.44), (18, 0.66), (13, 0.87)]
    pts = [np.array([0.0, 0.0])]
    for count, radius in rings:
        for i in range(count):
            angle = 2.0 * pi * i / count
            # stagger alternate rings
            x = radius * cos(angle)
            y = radius * sin(angle)
            pts.append(np.array([x, y]))
    return np.array(pts)

def _sa_optimize(pts, time_budget):
    """Simulated annealing with momentum-like moves to maximize min radius."""
    pts = pts.copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    current_score = best_score
    
    # Temperature schedule
    t0 = 0.05
    t_end = 0.0001
    # We'll do adaptive steps based on time
    alpha = 0.99995  # cooling factor per step
    
    rng = np.random.default_rng(42)
    
    # Precompute for speed
    n = len(pts)
    
    step = 0
    while _time_left() > 5.0:
        # Pick a random point to move
        idx = rng.integers(n)
        
        # Random perturbation
        scale = t0 * (t_end / t0) ** (step * 0.001)
        dx = rng.normal(0, scale)
        dy = rng.normal(0, scale)
        
        new_pts = pts.copy()
        new_pts[idx, 0] += dx
        new_pts[idx, 1] += dy
        
        new_score = _score(new_pts)
        
        if new_score >= current_score or (new_score > 0 and rng.random() < np.exp((new_score - current_score) / max(t0 * (t_end/t0)**(step*0.001), 1e-10))):
            pts = new_pts
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
        
        step += 1
        
        # Adapt scale
        t0 = max(t0 * 0.9999, t_end)
    
    return best_pts, best_score

def _gradient_polish(pts, iterations=500):
    """Gradient-based local optimization using subgradient of min distance."""
    pts = pts.copy()
    n = len(pts)
    best_score = _score(pts)
    best_pts = pts.copy()
    
    lr = 0.001
    for it in range(iterations):
        if _time_left() < 3.0:
            break
        
        # Compute all pairwise distances
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Find the minimum distance pair
        min_idx = np.unravel_index(np.argmin(dist), dist.shape)
        i, j = min_idx
        min_dist = dist[i, j]
        
        # Also check boundary
        norms = np.linalg.norm(pts, axis=1)
        min_norm = norms.min()
        boundary_idx = np.argmin(norms)
        
        # Determine which constraint is binding
        r_candidate = min(min_dist / 2.0, 1.0 - min_norm)
        
        if min_dist / 2.0 <= 1.0 - min_norm:
            # Pairwise constraint binding: push i and j apart
            if min_dist < 1e-10:
                direction = np.array([1.0, 0.0])
            else:
                direction = (pts[i] - pts[j]) / min_dist
            grad = np.zeros_like(pts)
            grad[i] = direction * lr
            grad[j] = -direction * lr
        else:
            # Boundary constraint binding: push inward
            if min_norm < 1e-10:
                direction = np.array([1.0, 0.0])
            else:
                direction = pts[boundary_idx] / min_norm
            grad = np.zeros_like(pts)
            grad[boundary_idx] = -direction * lr
        
        pts = pts + grad
        
        # Project back: ensure inside disk
        norms = np.linalg.norm(pts, axis=1)
        for k in range(n):
            if norms[k] > 0.999:
                pts[k] = pts[k] * (0.999 / norms[k])
        
        new_score = _score(pts)
        if new_score > best_score:
            best_score = new_score
            best_pts = pts.copy()
        
        # Decay learning rate
        lr *= 0.999
    
    return best_pts, best_score

def construct():
    """Construct 50 circle centers in unit disk maximizing common radius."""
    # Build initial seed from ring structure
    pts = _build_ring_seed()
    
    # Verify initial validity
    init_score = _score(pts)
    if init_score <= 0:
        # Fallback: simple ring placement with safer radii
        pts = np.zeros((N, 2))
        ring_configs = [(6, 0.20), (12, 0.40), (18, 0.60), (13, 0.80)]
        idx = 1
        for count, radius in ring_configs:
            for i in range(count):
                angle = 2.0 * pi * i / count
                pts[idx, 0] = radius * cos(angle)
                pts[idx, 1] = radius * sin(angle)
                idx += 1
        init_score = _score(pts)
    
    best_pts = pts.copy()
    best_score = init_score
    
    # Phase 1: Simulated Annealing
    if _time_left() > 20.0:
        sa_pts, sa_score = _sa_optimize(pts, _time_left() - 10.0)
        if sa_score > best_score:
            best_pts = sa_pts
            best_score = sa_score
    
    # Phase 2: Gradient polish
    if _time_left() > 5.0:
        for _ in range(3):  # multiple restarts of polish
            if _time_left() < 3.0:
                break
            pol_pts, pol_score = _gradient_polish(best_pts, iterations=300)
            if pol_score > best_score:
                best_pts = pol_pts
                best_score = pol_score
    
    # Final validation
    final_score = _score(best_pts)
    if final_score <= 0:
        # Absolute fallback: very conservative hexagonal packing
        spacing = 0.2
        points = []
        for j in range(-5, 6):
            for i in range(-5, 6):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * sqrt(3) / 2
                if x*x + y*y <= (1 - spacing/2)**2:
                    points.append([x, y])
        points = np.array(points[:N])
        if len(points) < N:
            # Pad with center
            extra = N - len(points)
            extra_pts = np.zeros((extra, 2))
            for k in range(extra):
                extra_pts[k] = [0.01 * (k+1), 0.0]
            points = np.vstack([points, extra_pts])
        return points
    
    return best_pts
