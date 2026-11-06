"""Improved packing: hexagonal init + force-based relaxation with deadline guard."""
import numpy as np
import time

N = 300
DEADLINE = 110.0  # safety margin before 120s limit


def _compute_radius(pts):
    """Compute the feasible common radius for a set of centres."""
    if len(pts) < 2:
        return 0.0
    # Boundary constraint
    r_boundary = 1.0 - np.max(np.linalg.norm(pts, axis=1))
    if r_boundary < 0:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    return min(r_boundary, min_dist / 2.0)


def _hex_init(n):
    """Generate a hexagonal lattice configuration inside the unit disk."""
    # Start with a reasonable spacing
    # For n circles in unit disk, rough spacing ~ 2/sqrt(n)
    spacing = 2.0 / (np.sqrt(n) * 1.1)
    points = []
    # Generate hex lattice points
    k_max = int(1.0 / spacing) + 3
    for j in range(-k_max, k_max + 1):
        for i in range(-k_max, k_max + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - spacing * 0.6:
                points.append([x, y])
    points = np.array(points)
    if len(points) == 0:
        return np.zeros((n, 2))
    # Sort by distance from origin and take first n
    radii = np.linalg.norm(points, axis=1)
    idx = np.argsort(radii)[:n]
    return points[idx]


def _force_relax(pts, max_iters=2000, deadline=None):
    """
    Force-based relaxation to maximize the minimum radius.
    Uses a smooth surrogate: maximize t subject to constraints,
    approximated by gradient ascent on a soft objective.
    """
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    
    if deadline is None:
        deadline = time.time() + 5.0
    
    # Initial radius estimate
    r = _compute_radius(pts)
    if r <= 0:
        return pts
    
    best_pts = pts.copy()
    best_r = r
    
    # Parameters
    lr = 0.01  # learning rate
    lr_decay = 0.999
    beta = 10.0  # sharpness of soft-min
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        # Compute pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Compute boundary distances
        boundary_dists = 1.0 - np.linalg.norm(pts, axis=1)
        
        # Current feasible radius
        r = min(np.min(boundary_dists), np.min(dists) / 2.0)
        
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
        
        if r <= 0:
            break
        
        # Gradient of soft objective
        # We want to maximize the minimum of: boundary_dists, dists/2
        # Use soft-min via log-sum-exp for smoothness
        
        # For pairwise constraints: gradient w.r.t. pts[i]
        grad = np.zeros_like(pts)
        
        # Boundary gradient: push away from boundary
        # d/dpts[i] (1 - |pts[i]|) = -pts[i]/|pts[i]|
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1e-10
        boundary_grad = -pts / norms  # gradient of (1-|x|) w.r.t. x
        
        # Weight by how close to boundary
        boundary_weights = np.exp(-beta * (boundary_dists - r))
        boundary_weights = np.clip(boundary_weights, 0, 1)
        
        # Pairwise gradients
        # d/dpts[i] (|pts[i]-pts[j]|/2) = (pts[i]-pts[j])/(2*|pts[i]-pts[j]|)
        # We want to increase distances, so push apart
        for i in range(n):
            for j in range(i + 1, n):
                d = dists[i, j]
                if d < 1e-10:
                    continue
                weight = np.exp(-beta * (d / 2.0 - r))
                weight = np.clip(weight, 0, 1)
                direction = (pts[i] - pts[j]) / d
                grad[i] += weight * direction * 0.5
                grad[j] -= weight * direction * 0.5
        
        # Combine gradients
        total_grad = boundary_weights[:, np.newaxis] * boundary_grad + grad
        total_grad *= 0.5
        
        # Update
        pts += lr * total_grad
        
        # Project back: ensure inside disk with margin
        norms = np.linalg.norm(pts, axis=1)
        for i in range(n):
            if norms[i] > 1.0 - 1e-6:
                pts[i] *= (1.0 - 1e-6) / norms[i]
        
        # Decay learning rate
        lr *= lr_decay
    
    return best_pts


def _local_optimize(pts, max_iters=500, deadline=None):
    """
    Local optimization: iteratively move the most constrained circle
    to improve the minimum radius.
    """
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    
    if deadline is None:
        deadline = time.time() + 5.0
    
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    
    step_size = 0.01
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        # Compute current radius
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        boundary_dists = 1.0 - np.linalg.norm(pts, axis=1)
        r = min(np.min(boundary_dists), np.min(dists) / 2.0)
        
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
        
        if r <= 0:
            break
        
        # Find the most constrained circle
        # Score each circle by how much it limits the radius
        scores = np.full(n, np.inf)
        for i in range(n):
            # Boundary constraint
            scores[i] = min(scores[i], boundary_dists[i])
            # Pairwise constraints
            min_pair = np.min(dists[i, :]) / 2.0
            scores[i] = min(scores[i], min_pair)
        
        # Move the most constrained circle
        idx = np.argmin(scores)
        
        # Try small random moves
        improved = False
        for _ in range(20):
            if deadline is not None and time.time() > deadline - 5.0:
                break
            delta = np.random.randn(2) * step_size
            new_pt = pts[idx].copy()
            new_pt += delta
            
            # Check if it's inside the disk
            if np.linalg.norm(new_pt) > 1.0 - 1e-8:
                continue
            
            pts[idx] = new_pt
            
            # Quick check
            new_boundary = 1.0 - np.linalg.norm(new_pt)
            new_min_pair = np.min(np.linalg.norm(pts[idx] - pts, axis=1)) / 2.0
            # Exclude self
            new_min_pair = np.min(np.delete(np.linalg.norm(pts[idx] - pts, axis=1), idx)) / 2.0
            
            new_r = min(new_boundary, new_min_pair)
            
            if new_r > r + 1e-10:
                improved = True
                break
            else:
                pts[idx] = best_pts[idx].copy() if False else pts[idx] - delta  # revert
                pts[idx] -= delta  # revert
        
        if not improved:
            # Try smaller step
            step_size *= 0.95
            if step_size < 1e-6:
                break
    
    return best_pts


def construct():
    start_time = time.time()
    
    # Phase 1: Hexagonal initialization
    pts = _hex_init(N)
    r = _compute_radius(pts)
    
    # Phase 2: Force-based relaxation
    deadline = start_time + DEADLINE
    pts = _force_relax(pts, max_iters=3000, deadline=deadline)
    r = _compute_radius(pts)
    
    # Phase 3: Local optimization if time permits
    if time.time() < deadline - 10.0:
        pts = _local_optimize(pts, max_iters=200, deadline=deadline)
        r = _compute_radius(pts)
    
    # Ensure validity
    if not np.all(np.isfinite(pts)) or pts.shape != (N, 2):
        # Fallback to hex init
        pts = _hex_init(N)
    
    # Final check
    norms = np.linalg.norm(pts, axis=1)
    if np.any(norms > 1.0):
        # Scale down slightly
        max_norm = np.max(norms)
        if max_norm > 0:
            pts *= (1.0 - 1e-8) / max_norm
    
    return pts
