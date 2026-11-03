"""
Strategy: Global smooth optimization using scipy.optimize.least_squares.
Instead of local pairwise fixes, we formulate the packing problem as minimizing
a smooth "soft-min" of the minimum distances (pairwise and to boundary).
The trust-region reflective algorithm uses gradient information to move all
300 circles simultaneously in a coordinated way, which can escape local minima
more effectively than pairwise adjustments.
"""
import numpy as np
import time
import math
from scipy.optimize import least_squares

N = 300
DEADLINE = 115.0  # Safety margin from 150s limit

def _hex_lattice_init():
    """Generate a hexagonal lattice of points inside the unit disk, return first N."""
    best_points = None
    # Try a few spacings to find a good initial configuration
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
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_boundary, min_pair / 2.0)

def _soft_min_resid(x, sharpness):
    """
    Compute residuals for least_squares.
    We want to maximize min(dist_boundary, min_pair/2).
    This is equivalent to minimizing -min(...).
    We use a soft-min approximation:
    soft_min(a, b) = -1/s * log(exp(-s*a) + exp(-s*b))
    Gradient is smooth.
    
    However, for least_squares, we need a vector of residuals.
    A common trick for max-min problems is to introduce a variable r
    and minimize (r - d_i)^2 for all constraints, but here we have
    pairwise distances which are O(N^2).
    
    Alternative: Minimize the sum of squared "violations" of a target radius r.
    But r is not a variable here.
    
    Let's try a different formulation:
    Minimize F(x) = sum_{i<j} max(0, 2*r - d_ij)^2 + sum_i max(0, r - (1-|c_i|))^2
    But r is unknown.
    
    Better: Minimize the "energy" E = sum_{i<j} 1/d_ij^2 + sum_i 1/(1-|c_i|)^2
    Minimizing this tends to maximize the minimum distance.
    This is a smooth, differentiable function.
    Residuals can be the square roots of the energy terms.
    
    Let's use:
    Residuals for pairs: 1/d_ij
    Residuals for boundary: 1/(1-|c_i|)
    
    This drives distances to be large.
    """
    pts = x.reshape(-1, 2)
    
    # Boundary residuals
    r_pts = np.hypot(pts[:, 0], pts[:, 1])
    # Avoid division by zero
    r_safe = np.maximum(r_pts, 1e-6)
    boundary_res = 1.0 / (1.0 - r_safe)
    
    # Pairwise residuals
    # To keep the residual vector size manageable, we might subsample or use all.
    # N=300 -> ~45000 pairs. Residual vector size ~45300.
    # This is large but least_squares can handle it.
    # However, computing all distances is O(N^2).
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # We only care about close pairs? No, the 1/d^2 energy penalizes all,
    # but strongly penalizes close ones.
    # To avoid huge values for distant pairs, we can clip or use a soft version.
    # Actually, 1/d is fine.
    
    # Get upper triangle indices
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    # Clip small distances to avoid inf
    pair_dists_safe = np.maximum(pair_dists, 1e-6)
    pair_res = 1.0 / pair_dists_safe
    
    # Combine
    residuals = np.concatenate([boundary_res, pair_res])
    
    return residuals

def _optimize_least_squares(initial_points, time_limit, sharpness_factor=1.0):
    """
    Run least_squares optimization.
    """
    start_time = time.time()
    x0 = initial_points.flatten()
    
    # Define objective
    def fun(x):
        return _soft_min_resid(x, sharpness_factor)
    
    # Run optimization
    try:
        result = least_squares(
            fun, 
            x0, 
            method='trf', 
            max_nfev=100, 
            # xtol=1e-8, 
            # ftol=1e-8,
            # gtol=1e-8
        )
    except Exception as e:
        # If optimization fails, return initial
        return initial_points, _score(initial_points)
    
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        return initial_points, _score(initial_points)
    
    optimized_points = result.x.reshape(-1, 2)
    
    # Ensure points are inside the unit circle
    r = np.hypot(optimized_points[:, 0], optimized_points[:, 1])
    mask = r > 1.0
    if np.any(mask):
        optimized_points[mask] = optimized_points[mask] / r[mask, np.newaxis] * (1.0 - 1e-9)
    
    score = _score(optimized_points)
    return optimized_points, score

def construct():
    start_time = time.time()
    
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
    
    best_points = points.copy()
    best_score = _score(points)
    
    # Multi-restart strategy
    # Restart 0: Direct optimization from hex lattice
    # Restart 1+: Perturb best found and optimize
    
    np.random.seed(42)
    
    max_restarts = 5
    
    for restart in range(max_restarts):
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed - 5.0  # 5s safety margin
        if remaining < 10.0:
            break
        
        if restart == 0:
            current = points.copy()
            time_limit = min(remaining * 0.5, 20.0)
        else:
            # Perturb from best found
            magnitude = 0.02 * (restart + 1)
            current = best_points + np.random.normal(0, magnitude, size=best_points.shape)
            # Clamp to disk
            r = np.hypot(current[:, 0], current[:, 1])
            mask = r > 0.98
            current[mask] = current[mask] / r[mask, np.newaxis] * 0.98
            time_limit = min(remaining * 0.4, 15.0)
        
        optimized, score = _optimize_least_squares(current, time_limit)
        
        if score > best_score:
            best_score = score
            best_points = optimized.copy()
            
            # If we are very close to the best known, we can try a finer polish
            if best_score > 0.0531:
                # Do a final polish with tighter tolerance?
                # least_squares already does this.
                pass
        
        # Check time
        if time.time() - start_time > DEADLINE - 10:
            break
    
    # Final check: ensure validity
    r = np.hypot(best_points[:, 0], best_points[:, 1])
    if np.any(r > 1.0):
        mask = r > 1.0
        best_points[mask] = best_points[mask] / r[mask, np.newaxis] * (1.0 - 1e-10)
    
    # Ensure no coincident points
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-10:
        for i in range(N):
            for j in range(i + 1, N):
                if np.linalg.norm(best_points[i] - best_points[j]) < 1e-10:
                    best_points[j, 0] += 1e-6
    
    return best_points
