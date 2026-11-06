"""Program for packing 50 equal circles in a unit circle; combined structural seed + gradient refinement."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 50
DEADLINE = time.time() + 140.0  # safety margin out of 150s


def compute_radius(points):
    """Compute the packing radius for given centres."""
    points = np.asarray(points, dtype=np.float64)
    # Distance to boundary
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    return min(boundary_dists.min(), min_pair_dist / 2.0)


def seed_hexagonal():
    """Generate a hexagonal lattice packing as a seed using the structural approach."""
    # Use the structural idea from inspiration 1: hexagonal lattice cut to unit disk
    # Try different spacings to find one that gives at least N points
    best_pts = None
    best_r = 0.0
    
    for spacing in np.linspace(0.18, 0.12, 100):
        k = int(1.0 / spacing) + 2
        rows = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing / 2.0:
                    rows.append([x, y])
        
        if len(rows) >= N:
            pts = np.array(rows)
            # Sort by distance from center, take closest N
            dists = np.hypot(pts[:, 0], pts[:, 1])
            idx = np.argsort(dists, kind="stable")[:N]
            pts = pts[idx]
            r = compute_radius(pts)
            if r > best_r:
                best_r = r
                best_pts = pts
    
    if best_pts is not None:
        return best_pts
    
    # Fallback: simple hexagonal grid
    spacing = 0.2
    points_list = []
    for j in range(-5, 6):
        for i in range(-5, 6):
            x = i * spacing + (0.5 * spacing if j % 2 else 0)
            y = j * spacing * np.sqrt(3) / 2
            if np.sqrt(x * x + y * y) < 1.0:
                points_list.append([x, y])
    
    points = np.array(points_list)
    dists = np.linalg.norm(points, axis=1)
    points = points[np.argsort(dists)]
    
    if len(points) >= N:
        selected = [points[np.argmin(dists)]]
        while len(selected) < N:
            sel = np.array(selected)
            min_dists = np.min(np.linalg.norm(points[:, np.newaxis, :] - sel[np.newaxis, :, :], axis=2), axis=1)
            selected.append(points[np.argmax(min_dists)])
        return np.array(selected)
    return points


def seed_ring():
    """Generate points on concentric rings."""
    pts = [[0.0, 0.0]]
    r1 = 0.25
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pts.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    r2 = 0.50
    for i in range(12):
        angle = 2 * np.pi * i / 12 + np.pi / 12
        pts.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    r3 = 0.75
    for i in range(18):
        angle = 2 * np.pi * i / 18
        pts.append([r3 * np.cos(angle), r3 * np.sin(angle)])
    r4 = 0.95
    for i in range(13):
        angle = 2 * np.pi * i / 13 + np.pi / 13
        pts.append([r4 * np.cos(angle), r4 * np.sin(angle)])
    
    pts = np.array(pts[:N])
    return pts


def seed_random():
    """Generate random points inside the unit circle."""
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.random.uniform(0, 0.9, N)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts


def optimize_from_seed(seed_points, max_iter=3000):
    """Local optimization using L-BFGS-B to maximize the packing radius with smooth approximation."""
    start_time = time.time()
    
    beta = 80.0  # sharpness of log-sum-exp

    def objective(x):
        points = x.reshape(N, 2)
        boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Smooth min via log-sum-exp
        b_min = -np.log(np.sum(np.exp(-beta * boundary_dists))) / beta
        d_min = -np.log(np.sum(np.exp(-beta * dists))) / beta
        
        r = min(b_min, d_min / 2.0)
        return -r
    
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    result = minimize(
        objective,
        seed_points.flatten(),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'maxfun': max_iter * 2}
    )
    
    optimized = result.x.reshape(N, 2)
    
    # Ensure points are strictly inside
    norms = np.linalg.norm(optimized, axis=1)
    too_far = norms >= 1.0
    if np.any(too_far):
        optimized[too_far] *= 0.99 / norms[too_far]
    
    return optimized


def local_refinement(points, n_steps=2000, step_size=0.005):
    """Coordinate descent local refinement."""
    points = points.copy()
    old_r = compute_radius(points)
    
    for step in range(n_steps):
        if time.time() > DEADLINE - 10.0:
            break
        i = random.randint(0, N - 1)
        for dx in [-step_size, 0, step_size]:
            for dy in [-step_size, 0, step_size]:
                if dx == 0 and dy == 0:
                    continue
                new_point = points[i].copy()
                new_point[0] += dx
                new_point[1] += dy
                if np.linalg.norm(new_point) >= 1.0:
                    continue
                points[i] = new_point
                new_r = compute_radius(points)
                if new_r > old_r:
                    old_r = new_r
                else:
                    points[i] -= np.array([dx, dy])
    return points


def search_portfolio():
    """Search approach: generate varied seeds and keep the best."""
    best_points = None
    best_radius = 0.0
    
    # Start with hexagonal seed (structural approach)
    seed_hex = seed_hexagonal()
    r_hex = compute_radius(seed_hex)
    best_points = seed_hex.copy()
    best_radius = r_hex
    
    # Try ring seed
    if time.time() < DEADLINE - 60.0:
        seed_ring_pts = seed_ring()
        r_ring = compute_radius(seed_ring_pts)
        if r_ring > best_radius:
            best_points = seed_ring_pts.copy()
            best_radius = r_ring
    
    # Generate a few random seeds
    n_random_seeds = 3
    for i in range(n_random_seeds):
        if time.time() > DEADLINE - 80.0:
            break
        seed = seed_random()
        r = compute_radius(seed)
        if r > best_radius:
            best_points = seed.copy()
            best_radius = r
    
    # Try optimizing from several seeds
    seeds_to_optimize = [best_points.copy()]
    seeds_to_optimize.append(seed_hex.copy())
    
    for i in range(3):
        if time.time() > DEADLINE - 90.0:
            break
        s = seed_random()
        seeds_to_optimize.append(s)
    
    for seed in seeds_to_optimize:
        if time.time() > DEADLINE - 25.0:
            break
        try:
            optimized = optimize_from_seed(seed, max_iter=2000)
            r = compute_radius(optimized)
            if r > best_radius:
                best_points = optimized.copy()
                best_radius = r
        except Exception:
            pass
    
    # Final local refinement
    if time.time() < DEADLINE - 10.0:
        best_points = local_refinement(best_points, n_steps=1000, step_size=0.003)
        best_radius = compute_radius(best_points)
    
    return best_points


def construct():
    """Construct centres of N equal circles inside the unit circle."""
    global DEADLINE
    DEADLINE = time.time() + 140.0
    
    np.random.seed(42)
    random.seed(42)
    
    points = search_portfolio()
    
    # Final validation and clipping
    points = np.asarray(points, dtype=np.float64)
    norms = np.linalg.norm(points, axis=1)
    for i in range(N):
        if norms[i] >= 1.0:
            points[i] *= 0.999 / norms[i]
    
    if not np.all(np.isfinite(points)):
        points = seed_hexagonal()
    
    return points


if __name__ == "__main__":
    points = construct()
    print("Final radius:", compute_radius(points))
