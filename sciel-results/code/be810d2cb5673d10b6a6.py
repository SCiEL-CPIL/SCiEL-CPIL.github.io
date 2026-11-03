"""Packing 50 equal circles in a unit circle using concentric rings structure."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0  # safety margin before 150s kill


def _pairwise_indices(n):
    """Return array of (i, j) pairs for i < j."""
    iu = np.triu_indices(n, k=1)
    return np.column_stack((iu[0], iu[1]))


def _rings_score(radii, counts, angles_offset=None):
    """Compute the max common radius for given ring radii and counts.
    
    radii: array of ring radii (distance from center)
    counts: array of number of circles on each ring
    angles_offset: array of angular offsets for each ring
    """
    if angles_offset is None:
        angles_offset = [0.0] * len(counts)
    n_rings = len(counts)
    pts = []
    for k in range(n_rings):
        R = radii[k]
        n = counts[k]
        if n == 0:
            continue
        if n == 1:
            pts.append([R * np.cos(angles_offset[k]), R * np.sin(angles_offset[k])])
        else:
            for i in range(n):
                theta = 2 * np.pi * i / n + angles_offset[k]
                pts.append([R * np.cos(theta), R * np.sin(theta)])
    
    pts = np.array(pts)
    if len(pts) != N:
        return 0.0
    
    # Compute minimum distance between any two centers
    pair_idx = _pairwise_indices(N)
    ci = pts[pair_idx[:, 0]]
    cj = pts[pair_idx[:, 1]]
    d = np.sqrt(np.sum((ci - cj) ** 2, axis=1))
    min_dist = np.min(d)
    
    # Compute minimum distance to boundary
    r2 = np.hypot(pts[:, 0], pts[:, 1])
    min_boundary = np.min(1 - r2)
    
    return min(min_dist / 2, min_boundary)


def _optimize_ring_structure(radii0, counts, angles_offset0=None, time_limit=30.0):
    """Optimize ring radii and angles for fixed counts."""
    if angles_offset0 is None:
        angles_offset0 = [0.0] * len(counts)
    
    n_params = len(counts) * 2  # radii and angles
    
    def objective(params):
        radii = params[:len(counts)]
        angles = params[len(counts):]
        # Ensure radii are non-negative
        radii = np.maximum(radii, 0.0)
        return -_rings_score(radii, counts, angles)
    
    x0 = np.concatenate([radii0, angles_offset0])
    
    t_start = time.time()
    try:
        res = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12},
            callback=lambda x: None if time.time() - t_start < time_limit else (_ for _ in ()).throw(Exception("timeout"))
        )
        if res.success:
            radii_opt = np.maximum(res.x[:len(counts)], 0.0)
            angles_opt = res.x[len(counts):]
            return radii_opt, angles_opt
    except Exception:
        pass
    
    return radii0, angles_offset0


def _local_optimize(c, time_limit=10.0):
    """Local optimization using L-BFGS-B with soft-min objective."""
    pair_idx = _pairwise_indices(N)
    
    def neg_soft_min(x):
        c = x.reshape(N, 2)
        ci = c[pair_idx[:, 0]]
        cj = c[pair_idx[:, 1]]
        d = np.sqrt(np.sum((ci - cj) ** 2, axis=1)) / 2
        b = 1 - np.hypot(c[:, 0], c[:, 1])
        all_vals = np.concatenate([d, b])
        beta = 200.0
        return np.log(np.sum(np.exp(-beta * all_vals))) / beta
    
    t_start = time.time()
    try:
        res = minimize(
            neg_soft_min,
            c.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-14},
            callback=lambda x: None if time.time() - t_start < time_limit else (_ for _ in ()).throw(Exception("timeout"))
        )
        if res.success:
            return res.x.reshape(N, 2)
    except Exception:
        pass
    
    return c


def _compute_r(c):
    """Compute the max common radius for given centers."""
    c = c.reshape(N, 2)
    pair_idx = _pairwise_indices(N)
    ci = c[pair_idx[:, 0]]
    cj = c[pair_idx[:, 1]]
    d = np.sqrt(np.sum((ci - cj) ** 2, axis=1))
    min_dist = np.min(d)
    r2 = np.hypot(c[:, 0], c[:, 1])
    min_boundary = np.min(1 - r2)
    return min(min_dist / 2, min_boundary)


def construct():
    t_start = time.time()
    
    # Known good structure for N=50: 1 + 6 + 12 + 18 + 13 or similar ring structures
    # Let's try several ring structures and optimize them
    
    # Structure 1: 1 + 6 + 12 + 18 + 13 = 50
    # Structure 2: 1 + 6 + 12 + 14 + 17 = 50
    # Structure 3: 1 + 6 + 10 + 14 + 19 = 50
    # Structure 4: 1 + 6 + 12 + 16 + 15 = 50
    
    ring_structures = [
        [1, 6, 12, 18, 13],
        [1, 6, 12, 14, 17],
        [1, 6, 10, 14, 19],
        [1, 6, 12, 16, 15],
        [1, 6, 10, 12, 21],
        [1, 6, 8, 14, 21],
    ]
    
    best_c = None
    best_r = 0.0
    
    # For each ring structure, try to optimize
    for counts in ring_structures:
        if time.time() - t_start > DEADLINE:
            break
        
        n_rings = len(counts)
        
        # Initial radii: evenly spaced from 0 to ~0.85
        if n_rings == 1:
            radii0 = np.array([0.0])
        else:
            # Center circle at 0, others spaced out
            radii0 = np.linspace(0, 0.85, n_rings)
            radii0[0] = 0.0
        
        angles0 = [0.0] * n_rings
        
        # Optimize ring structure
        radii_opt, angles_opt = _optimize_ring_structure(
            radii0, counts, angles0,
            time_limit=min(20.0, DEADLINE - (time.time() - t_start))
        )
        
        # Build centers from optimized ring structure
        pts = []
        for k in range(n_rings):
            R = radii_opt[k]
            n = counts[k]
            if n == 0:
                continue
            if n == 1:
                theta = angles_opt[k]
                pts.append([R * np.cos(theta), R * np.sin(theta)])
            else:
                for i in range(n):
                    theta = 2 * np.pi * i / n + angles_opt[k]
                    pts.append([R * np.cos(theta), R * np.sin(theta)])
        
        c = np.array(pts)
        if len(c) != N:
            continue
        
        r = _compute_r(c)
        
        # Local optimization
        if time.time() - t_start < DEADLINE - 5:
            c_opt = _local_optimize(c, time_limit=min(10.0, DEADLINE - (time.time() - t_start)))
            r_opt = _compute_r(c_opt)
            if r_opt > r:
                c = c_opt
                r = r_opt
        
        if r > best_r:
            best_r = r
            best_c = c.copy()
    
    # If we still don't have a good solution, try a random start with local optimization
    if best_c is None or best_r < 0.1:
        np.random.seed(42)
        for _ in range(5):
            if time.time() - t_start > DEADLINE:
                break
            angles = np.random.uniform(0, 2 * np.pi, N)
            radii = np.sqrt(np.random.uniform(0, 0.9, N))
            c = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            r = _compute_r(c)
            if time.time() - t_start < DEADLINE - 5:
                c_opt = _local_optimize(c, time_limit=min(10.0, DEADLINE - (time.time() - t_start)))
                r_opt = _compute_r(c_opt)
                if r_opt > r:
                    c = c_opt
                    r = r_opt
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    # Final polish with more iterations if time permits
    if best_c is not None and time.time() - t_start < DEADLINE - 3:
        c_final = _local_optimize(best_c, time_limit=min(8.0, DEADLINE - (time.time() - t_start) - 2))
        r_final = _compute_r(c_final)
        if r_final > best_r:
            best_c = c_final
            best_r = r_final
    
    # Ensure validity: clamp points that might be slightly outside
    c_final = best_c.reshape(N, 2)
    r2 = np.hypot(c_final[:, 0], c_final[:, 1])
    for i in range(N):
        if r2[i] > 1 - best_r + 1e-10:
            scale = (1 - best_r) / r2[i]
            c_final[i] *= scale
    
    return c_final
