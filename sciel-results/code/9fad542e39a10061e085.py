"""Packing 50 equal circles in a unit circle - vectorized force optimization."""
import time
import numpy as np

N = 50
_DEADLINE = time.time() + 140.0  # 10s safety margin before 150s kill

def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    dists = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(dists > 1.0):
        return 0.0
    r_cont = 1.0 - np.max(dists)
    if r_cont < 0:
        return 0.0
    if N < 2:
        return r_cont
    # Pairwise distances using broadcasting
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dmat = np.sqrt(dx ** 2 + dy ** 2)
    np.fill_diagonal(dmat, np.inf)
    min_pair = np.min(dmat)
    r_pair = min_pair / 2.0
    return min(r_cont, r_pair)

def _hex_lattice_init(spacing):
    """Generate points on a hexagonal lattice inside the unit disk."""
    k = int(1.0 / spacing) + 2
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    mask = radius <= 1.0 - spacing / 2.0
    points = points[mask]
    return points

def _ring_init():
    """Initialize points on concentric rings, mimicking the known optimal structure."""
    pts = []
    # Central point
    pts.append([0.0, 0.0])
    
    # Ring 1: 6 points
    r1 = 0.2
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pts.append([r1 * np.cos(angle), r1 * np.sin(angle)])
        
    # Ring 2: 12 points
    r2 = 0.4
    for i in range(12):
        angle = 2 * np.pi * i / 12
        pts.append([r2 * np.cos(angle), r2 * np.sin(angle)])
        
    # Ring 3: 18 points
    r3 = 0.6
    for i in range(18):
        angle = 2 * np.pi * i / 18
        pts.append([r3 * np.cos(angle), r3 * np.sin(angle)])
        
    # Ring 4: 13 points (50 - 1 - 6 - 12 - 18 = 13)
    r4 = 0.8
    for i in range(13):
        angle = 2 * np.pi * i / 13
        pts.append([r4 * np.cos(angle), r4 * np.sin(angle)])
        
    return np.array(pts)

def _random_init():
    """Initialize points randomly inside the unit circle."""
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 1, N))
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts

def _optimize(pts, max_iters=5000):
    """
    Vectorized force-based optimization.
    Uses repulsion between overlapping circles and attraction to center.
    """
    pts = pts.copy()
    n = len(pts)
    
    # Pre-allocate arrays for vectorization
    dx = np.zeros((n, n))
    dy = np.zeros((n, n))
    dist = np.zeros((n, n))
    forces = np.zeros((n, 2))
    
    # Parameters
    repulse_strength = 10.0
    attract_strength = 0.01
    boundary_strength = 50.0
    step_size = 0.01
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 2.0:
            break
        
        # Calculate pairwise differences
        dx = pts[:, 0:1] - pts[0:1, 0]
        dy = pts[:, 1:2] - pts[0:1, 1]
        dist = np.sqrt(dx**2 + dy**2)
        
        # Avoid division by zero
        np.fill_diagonal(dist, 1.0)
        
        # Repulsion force: strong when dist < 2r (overlap)
        # We want to maximize the minimum distance.
        # Force is proportional to 1/dist^2 or similar, but we need to balance with boundary.
        # Let's use a potential energy approach: U = sum(1/d^2) for d < threshold?
        # Actually, a simpler approach for packing:
        # If dist < 2*r_target, push apart.
        # But we don't know r_target. We want to maximize min(dist/2, 1-r).
        
        # Let's use a gradient ascent on the "minimum distance" proxy.
        # This is tricky. Let's stick to force-based.
        # Repulsion: F_rep = k_rep * (1/dist^2) * direction
        # Attraction: F_att = -k_att * position (spring to center)
        # Boundary: F_bnd = -k_bnd * position if |pos| > 1 - r_est
        
        # Estimate current radius to adjust boundary
        r_est = min(np.min(dist)/2.0, 1.0 - np.max(np.hypot(pts[:,0], pts[:,1])))
        
        # Repulsion
        # Only repel if close. Threshold = 2 * r_est + 0.1
        threshold = 2 * r_est + 0.1
        mask = dist < threshold
        mask = mask & (dist > 1e-6)
        
        # Force magnitude: (threshold - dist) / dist^2
        # This creates a strong repulsion when overlapping
        f_mag = np.zeros_like(dist)
        f_mag[mask] = (threshold - dist[mask]) / (dist[mask]**2)
        
        # Direction is (dx, dy) / dist
        # Force on i from j is f_mag * (dx_ij, dy_ij) / dist_ij
        # Sum over j
        fx = np.sum(f_mag * dx / dist, axis=1)
        fy = np.sum(f_mag * dy / dist, axis=1)
        
        forces[:, 0] = repulse_strength * fx
        forces[:, 1] = repulse_strength * fy
        
        # Attraction to center
        forces[:, 0] -= attract_strength * pts[:, 0]
        forces[:, 1] -= attract_strength * pts[:, 1]
        
        # Boundary constraint
        radii = np.hypot(pts[:, 0], pts[:, 1])
        # If radius + r_est > 1, push in
        # Effective radius for boundary: 1 - r_est
        # If radii > 1 - r_est, force inward
        bnd_mask = radii > (1.0 - r_est)
        if np.any(bnd_mask):
            # Force is proportional to penetration depth
            penetration = radii[bnd_mask] - (1.0 - r_est)
            # Direction is inward: -pos / |pos|
            forces[bnd_mask, 0] -= boundary_strength * penetration * (pts[bnd_mask, 0] / radii[bnd_mask])
            forces[bnd_mask, 1] -= boundary_strength * penetration * (pts[bnd_mask, 1] / radii[bnd_mask])
        
        # Update positions
        # Adaptive step size: reduce if forces are large to prevent explosion
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force > 1e-6:
            current_step = min(step_size, 0.1 / max_force)
        else:
            current_step = step_size
            
        pts += current_step * forces
        
        # Clamp to unit circle (safety)
        radii = np.hypot(pts[:, 0], pts[:, 1])
        mask_out = radii > 0.99
        if np.any(mask_out):
            pts[mask_out] *= 0.99 / radii[mask_out]
            
        # Decay step size slightly
        if it % 100 == 0:
            step_size *= 0.95
            
    return pts

def construct():
    """Construct optimal packing of 50 circles in unit circle."""
    best_pts = None
    best_score = 0.0
    
    # List of initializations
    inits = []
    
    # 1. Hex lattice with various spacings
    for spacing in [0.25, 0.28, 0.30]:
        points = _hex_lattice_init(spacing)
        if len(points) >= N:
            radii = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(radii, kind='stable')[:N]
            inits.append(points[idx].copy())
        else:
            # Pad with random points if not enough
            pts = points.copy()
            while len(pts) < N:
                angle = np.random.uniform(0, 2*np.pi)
                r = np.sqrt(np.random.uniform(0, 1))
                pts = np.vstack([pts, [[r*np.cos(angle), r*np.sin(angle)]]])
            inits.append(pts[:N].copy())
            
    # 2. Ring initialization
    inits.append(_ring_init())
    
    # 3. Random initializations
    for _ in range(3):
        inits.append(_random_init())
        
    # Optimize each initialization
    for init_pts in inits:
        if time.time() > _DEADLINE - 10.0:
            break
            
        # Run optimization
        opt_pts = _optimize(init_pts, max_iters=3000)
        
        # Score
        score = _score(opt_pts)
        
        if score > best_score:
            best_score = score
            best_pts = opt_pts.copy()
            
    # Final polish on the best result if time permits
    if best_pts is not None and time.time() < _DEADLINE - 5.0:
        # Run a shorter, more refined optimization on the best point
        # Maybe with a smaller step size?
        # For now, just run again
        polished = _optimize(best_pts, max_iters=1000)
        pol_score = _score(polished)
        if pol_score > best_score:
            best_pts = polished
            best_score = pol_score
            
    # Fallback if something went wrong
    if best_pts is None:
        best_pts = _ring_init()
        
    return best_pts
