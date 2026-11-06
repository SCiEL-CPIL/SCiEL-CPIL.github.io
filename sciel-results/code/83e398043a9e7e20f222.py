"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(points):
    """Compute the minimum radius for a set of points."""
    if points is None or len(points) != N:
        return 0.0
    
    # Check for non-finite values
    if not np.all(np.isfinite(points)):
        return 0.0
        
    # Distance to boundary
    norms = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - norms)
    
    # If any point is outside, the radius is effectively negative or invalid for packing
    # But the formula says min(1 - |c_i|). If |c_i| > 1, this is negative.
    # The task says "A centre on or outside the container ... make the candidate invalid."
    # So we should ensure all points are strictly inside for a valid score, 
    # or return a very low score.
    if min_boundary < -1e-9:
        return 0.0

    # Pairwise distances
    # Use broadcasting for efficiency
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_boundary, min_pair)

def _energy_and_grad(x):
    """
    Compute energy and gradient for the repulsion model.
    E = sum_{i<j} 1/d_ij^2 + lambda * sum_i max(0, |c_i| - 1)^2
    """
    pts = x.reshape(N, 2)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    
    # Avoid division by zero
    dists_sq_safe = np.maximum(dists_sq, 1e-12)
    
    # Energy term: sum 1/d^2
    # We sum over i < j
    iu = np.triu_indices(N, k=1)
    e_pair = np.sum(1.0 / dists_sq_safe[iu])
    
    # Gradient of pair term
    # grad_i = sum_{j != i} -2 * (x_i - x_j) / d_ij^4
    # F[i,j] = -2 * (x_i - x_j) / d_ij^4
    # Note: d_ij^4 = (d_ij^2)^2
    mask = ~np.eye(N, dtype=bool)
    # Create force matrix
    # We need to be careful with the shape. diff is (N, N, 2)
    # dists_sq_safe is (N, N)
    # factor[i,j] = -2 / dists_sq_safe[i,j]^2
    factor = np.zeros((N, N))
    factor[mask] = -2.0 / (dists_sq_safe[mask] ** 2)
    
    # grad_pair[i] = sum_j factor[i,j] * diff[i,j]
    grad_pair = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    
    # Boundary penalty
    lambda_bound = 100.0
    norms = np.linalg.norm(pts, axis=1)
    outside = norms > 1.0
    # Penalty term
    e_bound = 0.0
    grad_bound = np.zeros_like(pts)
    
    if np.any(outside):
        # Only compute for outside points
        out_norms = norms[outside]
        e_bound = lambda_bound * np.sum((out_norms - 1.0)**2)
        
        # Gradient: lambda * 2 * (r - 1) * (x / r)
        # d/dr (r-1)^2 = 2(r-1)
        # d x_i / dr = x_i / r
        grad_vals = 2.0 * lambda_bound * (out_norms - 1.0) / out_norms
        grad_bound[outside] = grad_vals[:, np.newaxis] * pts[outside]
        
    total_e = e_pair + e_bound
    total_grad = grad_pair + grad_bound
    
    return total_e, total_grad.flatten()

def _seed_hexagonal():
    """Generate a hexagonal lattice seed."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        # Filter points inside unit circle (with margin)
        points = points[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            # Sort by radius to pick central ones first? 
            # Actually for optimization, any valid set is fine.
            # Picking closest to center might help stability.
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98

def _seed_random():
    """Generate a random seed within the unit circle."""
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.95, N)) # Keep slightly inside
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return points

def _seed_rings():
    """Generate seeds on concentric rings."""
    # Approximate number of circles per ring
    # Center: 1
    # Ring 1: 6
    # Ring 2: 12
    # Ring 3: 18
    # Ring 4: 24
    # Total: 1+6+12+18 = 37. Need 13 more.
    # Let's just distribute them somewhat evenly.
    
    # Simple heuristic:
    # Place 1 at center
    pts = []
    if N > 0:
        pts.append([0.0, 0.0])
    
    remaining = N - 1
    ring_idx = 1
    while remaining > 0:
        # Estimate capacity of this ring
        # Circumference ~ 2 * pi * r
        # Spacing ~ 2 * r_circle
        # This is circular. Let's just use fixed radii.
        r = ring_idx * 0.2
        if r > 0.9: break
        
        # Max circles on this ring
        max_c = int(2 * np.pi * r / 0.25) # Rough estimate
        if max_c < 1: max_c = 1
        
        count = min(remaining, max_c)
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        ring_pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        pts.extend(ring_pts)
        remaining -= count
        ring_idx += 1
        
    pts = np.array(pts)
    if len(pts) < N:
        # Fill remaining with random
        extra = _seed_random()[:N - len(pts)]
        pts = np.vstack([pts, extra])
    elif len(pts) > N:
        pts = pts[:N]
        
    # Ensure inside
    norms = np.linalg.norm(pts, axis=1)
    pts[norms > 0.99] *= 0.99 / norms[norms > 0.99, np.newaxis]
    
    return pts

def _optimize_from_seed(seed, deadline):
    """Optimize from a seed using L-BFGS-B."""
    x0 = seed.flatten()
    
    # Check time
    if time.time() > deadline:
        return seed, _score(seed)
        
    try:
        res = minimize(_energy_and_grad, x0, method='L-BFGS-B', 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.success or res.status == 0:
            pts = res.x.reshape(N, 2)
            # Clip to ensure validity if L-BFGS-B drifted slightly
            norms = np.linalg.norm(pts, axis=1)
            if np.any(norms > 1.0):
                pts[norms > 1.0] *= 0.9999 / norms[norms > 1.0, np.newaxis]
            return pts, _score(pts)
        else:
            return seed, _score(seed)
            
    except Exception:
        return seed, _score(seed)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    best_points = None
    best_score = -1.0
    
    seeds = []
    
    # 1. Hexagonal
    try:
        seeds.append(_seed_hexagonal())
    except:
        pass
        
    # 2. Random seeds
    for _ in range(10):
        seeds.append(_seed_random())
        
    # 3. Ring seeds
    for _ in range(3):
        seeds.append(_seed_rings())
        
    # Shuffle seeds to ensure diversity in time allocation if we run out
    np.random.shuffle(seeds)
    
    for seed in seeds:
        if time.time() > deadline:
            break
            
        pts, score = _optimize_from_seed(seed, deadline)
        
        if score > best_score:
            best_score = score
            best_points = pts
            
    if best_points is None:
        # Fallback
        best_points = _seed_hexagonal()
        
    return best_points
# EVOLVE-BLOCK-END
