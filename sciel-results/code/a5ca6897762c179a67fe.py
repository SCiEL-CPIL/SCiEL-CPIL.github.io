"""Rewritten program: hexagonal lattice init + Nelder-Mead optimization on radius."""
import numpy as np
import time
import math

N = 300

# EVOLVE-BLOCK-START
def _hex_lattice_init():
    """Generate a hexagonal lattice of points inside the unit disk, return first N."""
    # We want ~300 points in a hex lattice inside unit disk.
    # Area of unit disk = pi. Each hex cell area = sqrt(3)/2 * s^2.
    # For 300 points: 300 * sqrt(3)/2 * s^2 = pi => s^2 = 2*pi/(300*sqrt(3)) ≈ 0.01209
    # s ≈ 0.1099
    # But boundary effects mean we need slightly smaller spacing to get >= 300 points.
    
    best_points = None
    for s in [0.10, 0.102, 0.104, 0.106, 0.108, 0.110, 0.112, 0.115, 0.12]:
        k = int(1.0 / s) + 3
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * math.sqrt(3.0) / 2.0
                if x * x + y * y <= 1.0:
                    points.append((x, y))
        points = np.array(points)
        if len(points) >= N:
            # Keep the N closest to center for a compact packing
            r = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(r, kind='stable')[:N]
            pts = points[idx]
            if best_points is None or len(pts) > len(best_points):
                best_points = pts
                if len(pts) == N:
                    break
    if best_points is not None and len(best_points) == N:
        return best_points
    
    # Fallback: very fine grid
    s = 0.05
    k = int(1.0 / s) + 3
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * math.sqrt(3.0) / 2.0
            if x * x + y * y <= 1.0:
                points.append((x, y))
    points = np.array(points)
    r = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(r, kind='stable')[:N]
    return points[idx]


def _score(points):
    """Compute the maximum radius r for given centers."""
    # Distance to boundary
    dist_boundary = 1.0 - np.hypot(points[:, 0], points[:, 1])
    min_boundary = np.min(dist_boundary)
    
    # Pairwise distances
    # Use efficient computation
    dx = points[:, 0:1] - points[0:1, 0]  # This won't work for all pairs
    # Better: compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    # Set diagonal to inf
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_boundary, min_pair / 2.0)


def _optimize(points, time_limit):
    """Run Nelder-Mead optimization to maximize the radius."""
    start_time = time.time()
    
    # Objective: maximize r = min(min_pairwise_dist/2, min_boundary_dist)
    # Minimize negative of that
    
    def objective(flat_points):
        pts = flat_points.reshape(N, 2)
        # Boundary constraint
        dist_boundary = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
        if np.any(dist_boundary < 0):
            return -1e6  # Penalty for being outside
        min_boundary = np.min(dist_boundary)
        
        # Pairwise distances - use a faster method
        # For 300 points, 300*299/2 = 44850 pairs
        # Vectorized
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        r = min(min_boundary, min_pair / 2.0)
        return -r
    
    # Initial simplex: perturb the initial points
    x0 = points.copy()
    
    # Use scipy if available, otherwise implement Nelder-Mead
    try:
        from scipy.optimize import minimize
        res = minimize(objective, x0.flatten(), method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10,
                               'adaptive': True})
        if time.time() - start_time > time_limit * 0.7:
            # Check if current best is better
            best_pts = res.x.reshape(N, 2)
            best_score = -res.fun
            init_score = _score(x0)
            if best_score > init_score:
                return best_pts
            return x0
        best_pts = res.x.reshape(N, 2)
        best_score = -res.fun
        init_score = _score(x0)
        if best_score > init_score:
            return best_pts
        return x0
    except ImportError:
        pass
    except Exception:
        pass
    
    # Fallback: simple coordinate descent / gradient-free
    # Just return initial if optimization fails
    return x0


def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin from 120s
    
    # Generate initial hexagonal lattice
    points = _hex_lattice_init()
    if points is None or len(points) != N:
        # Emergency fallback
        points = np.zeros((N, 2))
        for i in range(N):
            angle = 2 * np.pi * i / N
            r = 0.5
            points[i, 0] = r * np.cos(angle)
            points[i, 1] = r * np.sin(angle)
    
    init_score = _score(points)
    
    # Run optimization if time permits
    elapsed = time.time() - start_time
    if elapsed < time_limit * 0.5:
        remaining = time_limit - elapsed - 5.0  # 5s safety
        if remaining > 10:
            points = _optimize(points, remaining)
    
    # Final check: ensure validity
    final_score = _score(points)
    
    # Ensure all points are inside the unit disk
    r = np.hypot(points[:, 0], points[:, 1])
    if np.any(r > 1.0):
        # Clip to boundary
        mask = r > 1.0
        points[mask] = points[mask] / r[mask, np.newaxis] * (1.0 - 1e-10)
    
    # Ensure no coincident points
    # Check for very close points
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-10:
        # Separate coincident points
        for i in range(N):
            for j in range(i + 1, N):
                if np.linalg.norm(points[i] - points[j]) < 1e-10:
                    points[j, 0] += 1e-6
    
    return points

# EVOLVE-BLOCK-END
