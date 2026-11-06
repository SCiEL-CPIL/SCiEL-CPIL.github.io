"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 130.0  # Safety margin before 150s kill


def _elapsed():
    return time.time() - _START_TIME


def _remaining():
    return max(0.0, _DEADLINE - time.time())


def compute_radius(pts):
    """Compute the maximum common radius for given centers."""
    if len(pts) < 2:
        return 0.0
    d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min() / 2.0
    min_dist_to_edge = (1.0 - np.linalg.norm(pts, axis=1)).min()
    return min(min_pair, min_dist_to_edge)


def soft_min(values, beta=100.0):
    """Log-sum-exp approximation to min for smooth optimization."""
    m = np.min(values)
    return m - np.log(np.sum(np.exp(beta * (values - m)))) / beta


def objective(pts_flat):
    """Smooth objective to minimize (negative of radius)."""
    pts = pts_flat.reshape(-1, 2)
    n = len(pts)
    
    # Pairwise distances
    d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    
    # Distance to boundary
    dist_to_edge = 1.0 - np.linalg.norm(pts, axis=1)
    
    # Collect all constraints
    pair_dists = d[d > 0] / 2.0
    all_constraints = np.concatenate([pair_dists, dist_to_edge])
    
    # Soft min
    return -soft_min(all_constraints, beta=80.0)


def seed_hexagonal(n_points):
    """Generate hexagonal lattice points within unit circle."""
    spacing = 2.0 / (np.sqrt(n_points) + 1.5)
    points = []
    k = int(1.0 / spacing) + 3
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) < n_points:
        # Fallback: random points
        angles = np.random.uniform(0, 2 * np.pi, n_points)
        radii = np.sqrt(np.random.uniform(0, 1, n_points))
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        return points
    
    # Sort by distance from center, take closest n_points
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n_points]
    return points[idx]


def seed_concentric_rings(n_points):
    """Generate points on concentric rings."""
    points = [[0.0, 0.0]]
    remaining = n_points - 1
    
    ring_radius = 0.15
    ring_spacing = 0.12
    
    while remaining > 0 and ring_radius < 0.95:
        circumference = 2 * np.pi * ring_radius
        max_on_ring = max(1, int(circumference / (2 * ring_radius * 0.8)))
        count = min(remaining, max_on_ring)
        
        if count > 1:
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            for a in angles:
                if remaining <= 0:
                    break
                points.append([ring_radius * np.cos(a), ring_radius * np.sin(a)])
                remaining -= 1
        else:
            points.append([ring_radius, 0.0])
            remaining -= 1
        
        ring_radius += ring_spacing
    
    # If we still need more points, add random ones
    while remaining > 0:
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.sqrt(np.random.uniform(0, 0.9))
        points.append([r * np.cos(angle), r * np.sin(angle)])
        remaining -= 1
    
    return np.array(points[:n_points])


def seed_random(n_points, seed=42):
    """Generate random points."""
    rng = np.random.RandomState(seed)
    angles = rng.uniform(0, 2 * np.pi, n_points)
    radii = np.sqrt(rng.uniform(0, 0.95, n_points))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def seed_optimal_known(n_points):
    """
    Generate an initial configuration based on known good packings.
    For 50 circles, use a structured ring-based approach that approximates
    the Graham et al. solution.
    """
    # The best known packing for 50 circles has radius ~0.1258254895304
    # This implies diameter ~0.2516509790608
    # Let's create a configuration with rings
    
    points = []
    
    # Center point
    points.append([0.0, 0.0])
    
    # Ring 1: 6 points at radius ~0.25
    r1 = 0.2516509790608
    for i in range(6):
        angle = i * 2 * np.pi / 6
        points.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    
    # Ring 2: 12 points at radius ~0.5
    r2 = 0.5033019581216
    for i in range(12):
        angle = i * 2 * np.pi / 12
        points.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    
    # Ring 3: 18 points at radius ~0.75
    r3 = 0.7549529371824
    for i in range(18):
        angle = i * 2 * np.pi / 18
        points.append([r3 * np.cos(angle), r3 * np.sin(angle)])
    
    # Ring 4: 14 points at radius ~0.95
    r4 = 0.95
    for i in range(14):
        angle = i * 2 * np.pi / 14
        points.append([r4 * np.cos(angle), r4 * np.sin(angle)])
    
    # Trim to exactly n_points
    points = np.array(points[:n_points])
    
    # If we have too many, trim; if too few, add random
    if len(points) < n_points:
        remaining = n_points - len(points)
        rng = np.random.RandomState(12345)
        for _ in range(remaining):
            angle = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(0.85, 0.98)
            points = np.vstack([points, [r * np.cos(angle), r * np.sin(angle)]])
    
    return points[:n_points]


def optimize_from_seed(seed_pts, max_time=25.0):
    """Optimize starting from a seed using L-BFGS-B with adaptive beta."""
    n = len(seed_pts)
    x0 = seed_pts.flatten()
    
    start = time.time()
    timeout = start + max_time
    
    # First pass: moderate beta
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        options={'maxiter': 300, 'maxfun': 1500, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    pts = result.x.reshape(-1, 2)
    
    # Check time
    if time.time() > timeout - 5.0:
        return pts
    
    # Second pass: higher beta for sharper minimum
    def objective_sharp(pts_flat):
        pts = pts_flat.reshape(-1, 2)
        d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        dist_to_edge = 1.0 - np.linalg.norm(pts, axis=1)
        pair_dists = d[d > 0] / 2.0
        all_constraints = np.concatenate([pair_dists, dist_to_edge])
        return -soft_min(all_constraints, beta=150.0)
    
    result2 = minimize(
        objective_sharp,
        pts.flatten(),
        method='L-BFGS-B',
        options={'maxiter': 200, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    return result2.x.reshape(-1, 2)


def local_refinement(pts, max_time=10.0):
    """Local refinement with very tight tolerance and high beta."""
    n = len(pts)
    x0 = pts.flatten()
    
    def objective_tight(pts_flat):
        pts = pts_flat.reshape(-1, 2)
        d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        dist_to_edge = 1.0 - np.linalg.norm(pts, axis=1)
        pair_dists = d[d > 0] / 2.0
        all_constraints = np.concatenate([pair_dists, dist_to_edge])
        return -soft_min(all_constraints, beta=300.0)
    
    result = minimize(
        objective_tight,
        x0,
        method='L-BFGS-B',
        options={'maxiter': 500, 'maxfun': 3000, 'ftol': 1e-14, 'gtol': 1e-12}
    )
    
    return result.x.reshape(-1, 2)


def search_portfolio():
    """Search with multiple seeds and strategies."""
    best_pts = None
    best_radius = 0.0
    
    # Try multiple seed strategies, including the known-optimal-like configuration
    seeds = [
        seed_optimal_known(N),
        seed_hexagonal(N),
        seed_concentric_rings(N),
        seed_random(N, seed=42),
        seed_random(N, seed=123),
        seed_random(N, seed=999),
    ]
    
    for i, seed in enumerate(seeds):
        if _elapsed() > _DEADLINE - 20.0:
            break
        
        # Optimize from this seed
        opt_pts = optimize_from_seed(seed, max_time=15.0)
        
        # Compute radius
        r = compute_radius(opt_pts)
        
        if r > best_radius:
            best_radius = r
            best_pts = opt_pts.copy()
        
        # Local refinement
        if _remaining() > 10.0:
            refined = local_refinement(opt_pts, max_time=5.0)
            r_refined = compute_radius(refined)
            
            if r_refined > best_radius:
                best_radius = r_refined
                best_pts = refined.copy()
    
    # Final polish if time allows
    if best_pts is not None and _remaining() > 8.0:
        final = local_refinement(best_pts, max_time=min(6.0, _remaining() - 3.0))
        r_final = compute_radius(final)
        if r_final > best_radius:
            best_pts = final
            best_radius = r_final
    
    if best_pts is None:
        best_pts = seed_hexagonal(N)
    
    return best_pts


def construct():
    """Construct centers of N equal circles inside unit circle."""
    if _elapsed() > _DEADLINE - 10.0:
        # Not enough time, return hexagonal seed
        return seed_hexagonal(N)
    
    pts = search_portfolio()
    
    # Ensure validity: check all points are inside unit circle
    dists = np.linalg.norm(pts, axis=1)
    if np.any(dists > 1.0):
        # Project back inside
        mask = dists > 1.0
        pts[mask] = pts[mask] / dists[mask, np.newaxis] * 0.999
    
    # Final sanity check: ensure no coincident points
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-10:
                pts[j] += [1e-6, 0]
    
    return pts
# EVOLVE-BLOCK-END
