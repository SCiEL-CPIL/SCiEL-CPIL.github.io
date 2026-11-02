"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import random
import numpy as np
import time
from scipy.optimize import minimize

S3H = np.sqrt(3) / 2
TIME_LIMIT = 120  # seconds


def _min_area_raw(points):
    """Compute the minimum triangle area among all triples of points."""
    n = len(points)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = points[j, 0] - points[i, 0]
                dy1 = points[j, 1] - points[i, 1]
                dx2 = points[k, 0] - points[i, 0]
                dy2 = points[k, 1] - points[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2) * 0.5
                if cross < min_a:
                    min_a = cross
    return min_a


def _penalized_min_area_flat(x):
    """Objective: maximize min triangle area, penalize points outside triangle.
    x is a flat array of shape (2*N,). Returns negative of (min_area - penalty)."""
    pts = x.reshape(N, 2)
    # Check if points are inside the triangle: (0,0), (1,0), (0.5, sqrt(3)/2)
    # Barycentric coordinates: for point (px, py),
    # w0 = 1 - px - py/S3H, w1 = px - py/(2*S3H), w2 = py/S3H
    # All must be >= 0 (with small tolerance)
    py = pts[:, 1]
    px = pts[:, 0]
    w0 = 1.0 - px - py / S3H
    w1 = px - py / (2.0 * S3H)
    w2 = py / S3H
    # Penalty for being outside
    penalty = np.sum(np.maximum(0, -w0)) + np.sum(np.maximum(0, -w1)) + np.sum(np.maximum(0, -w2))
    min_a = _min_area_raw(pts)
    return -(min_a - 10.0 * penalty)


def _local_polish(x0, rng, end_time):
    """Run Nelder-Mead from initial point x0, return best result.
    Nelder-Mead is chosen for better handling of non-smooth objectives."""
    # Nelder-Mead can be sensitive to initial simplex size.
    # We provide a small perturbation to start the simplex.
    res = minimize(_penalized_min_area_flat, x0, method='Nelder-Mead',
                   options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
    
    best_x = res.x.copy()
    best_val = -res.fun
    
    return best_x, best_val


def construct():
    rng = random.Random(42)
    start_time = time.time()
    end_time = start_time + TIME_LIMIT
    
    best_points = None
    best_score = 0.0
    
    # Multiple random starts with local polish
    # Reduced starts to allow more iterations per polish if needed, 
    # but Nelder-Mead is slower per iteration than L-BFGS-B in low dims? 
    # Actually NM is derivative free, so it's more robust but potentially slower.
    # We will try 20 starts to be safe within time limit.
    n_starts = 20
    
    for s in range(n_starts):
        if time.time() > end_time:
            break
        
        # Generate random initial configuration inside triangle
        pts = np.zeros((N, 2))
        for i in range(N):
            while True:
                u, v = rng.random(), rng.random()
                if u + v < 0.95 and u > 0.02 and v > 0.02:
                    break
            pts[i, 0] = u + 0.5 * v
            pts[i, 1] = S3H * v
        
        x0 = pts.flatten()
        
        # Local polish
        polished_x, polished_val = _local_polish(x0, rng, end_time)
        
        # Convert back and check
        pts_polished = polished_x.reshape(N, 2)
        
        # Check if all points are inside triangle
        px = pts_polished[:, 0]
        py = pts_polished[:, 1]
        w0 = 1.0 - px - py / S3H
        w1 = px - py / (2.0 * S3H)
        w2 = py / S3H
        if np.min(w0) < -1e-6 or np.min(w1) < -1e-6 or np.min(w2) < -1e-6:
            # Clamp to boundary
            for i in range(N):
                if w0[i] < 0:
                    w0[i] = 0
                if w1[i] < 0:
                    w1[i] = 0
                if w2[i] < 0:
                    w2[i] = 0
                # Renormalize
                total = w0[i] + w1[i] + w2[i]
                if total == 0:
                    continue
                w0[i] /= total
                w1[i] /= total
                w2[i] /= total
                pts_polished[i, 0] = w1[i] + 0.5 * w2[i]
                pts_polished[i, 1] = S3H * w2[i]
        
        # Compute actual min area
        min_a = _min_area_raw(pts_polished)
        normalized = min_a / (S3H / 2)  # Divide by triangle area sqrt(3)/4
        
        if normalized > best_score:
            best_score = normalized
            best_points = pts_polished.copy()
    
    if best_points is None:
        # Fallback: random placement
        points = []
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                points.append([u + 0.5 * v, S3H * v])
        best_points = np.array(points)
    
    return best_points
# EVOLVE-BLOCK-END
