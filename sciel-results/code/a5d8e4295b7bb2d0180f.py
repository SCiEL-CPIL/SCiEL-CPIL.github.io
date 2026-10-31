"""Improved program: optimize 16 points in 2D to minimize (d_max/d_min)^2."""
import time
import itertools
import numpy as np
from scipy.optimize import minimize

N = 16
D = 2
DEADLINE = 110.0  # seconds, safety margin before 120s limit

def _score(points):
    """Compute (max_dist / min_dist)^2."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    # zero out diagonal
    np.fill_diagonal(dists_sq, 0)
    min_d2 = np.min(dists_sq[dists_sq > 0])
    max_d2 = np.max(dists_sq)
    if min_d2 <= 0:
        return float('inf')
    return (np.sqrt(max_d2) / np.sqrt(min_d2))**2

def _initial_hex19():
    """Generate 19 points on triangular lattice in a hexagon."""
    pts = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            # Axial to Cartesian for triangular lattice
            x = 1.5 * q
            y = np.sqrt(3) / 2 * q + np.sqrt(3) * r
            if q**2 + q*r + r**2 <= 4:  # hexagonal constraint (radius 2)
                pts.append((x, y))
    return np.array(pts[:19], dtype=float)

def _best_subset_of_19():
    """Find best 16-point subset from 19-point hexagon."""
    hex_pts = _initial_hex19()
    if len(hex_pts) != 19:
        return hex_pts  # fallback
    
    best_score = float('inf')
    best_subset = None
    indices = list(range(19))
    
    for remove in itertools.combinations(indices, 3):
        remove_set = set(remove)
        subset = [hex_pts[i] for i in indices if i not in remove_set]
        arr = np.array(subset, dtype=float)
        s = _score(arr)
        if s < best_score:
            best_score = s
            best_subset = arr
    
    if best_subset is not None:
        return best_subset
    return hex_pts[:16]

def _grid_init():
    """Initial grid placement."""
    side = 4
    grid = list(itertools.product(range(side), repeat=2))[:16]
    return np.array(grid, dtype=float)

def _optimize_from(points0, max_time=None):
    """Run local optimization from initial points."""
    def objective(params):
        pts = params.reshape(N, 2)
        # Subtract center to keep numerical stability
        pts = pts - pts.mean(axis=0, keepdims=True)
        s = _score(pts)
        return s
    
    def gradient(params):
        pts = params.reshape(N, 2)
        pts = pts - pts.mean(axis=0, keepdims=True)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, 0)
        dists = np.sqrt(dists_sq)
        
        # Avoid division by zero
        dists_safe = np.where(dists > 1e-10, dists, 1.0)
        min_d = np.min(dists[dists > 1e-10])
        max_d = np.max(dists)
        
        if min_d <= 0 or max_d <= 0:
            return np.zeros_like(params)
        
        ratio_sq = (max_d / min_d)**2
        # d(ratio_sq)/d(max_d) = 2*max_d/min_d^2
        # d(ratio_sq)/d(min_d) = -2*max_d^2/min_d^3
        
        # For each pair, compute contribution
        # This is complex; use numerical gradient as fallback
        return None
    
    # Use Nelder-Mead for simplicity and robustness (no gradient needed)
    x0 = points0.flatten()
    
    result = minimize(
        objective,
        x0,
        method='Nelder-Mead',
        options={'maxiter': 5000, 'fatol': 1e-12, 'xatol': 1e-12, 'adaptive': True}
    )
    
    if result.success or hasattr(result, 'fun'):
        return result.x.reshape(N, 2)
    return points0

def construct():
    start_time = time.time()
    
    # Generate initial candidates
    candidates = []
    
    # 1. Best 16-subset from 19-point hexagon
    hex_subset = _best_subset_of_19()
    candidates.append(hex_subset)
    
    # 2. Grid
    grid = _grid_init()
    candidates.append(grid)
    
    # 3. Slightly perturbed hex subsets
    np.random.seed(42)
    for i in range(5):
        perturbed = hex_subset + np.random.randn(N, 2) * 0.1
        candidates.append(perturbed)
    
    # Optimize each candidate
    best_points = None
    best_score = float('inf')
    
    for i, init in enumerate(candidates):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            break
        
        # Center and normalize initial points for better optimization
        pts_centered = init - init.mean(axis=0, keepdims=True)
        scale = np.max(np.linalg.norm(pts_centered, axis=1))
        if scale > 0:
            pts_centered = pts_centered / scale
        
        optimized = _optimize_from(pts_centered)
        optimized = optimized - optimized.mean(axis=0, keepdims=True)
        
        s = _score(optimized)
        if s < best_score:
            best_score = s
            best_points = optimized
        
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            break
    
    # Final polish: run Nelder-Mead with tighter tolerances if time permits
    elapsed = time.time() - start_time
    if best_points is not None and elapsed < DEADLINE - 15:
        # Try multiple local optimizations from best point with small perturbations
        for trial in range(10):
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - 15:
                break
            if trial == 0:
                start_pt = best_points
            else:
                start_pt = best_points + np.random.randn(N, 2) * 0.01
                start_pt = start_pt - start_pt.mean(axis=0, keepdims=True)
            
            polished = _optimize_from(start_pt)
            polished = polished - polished.mean(axis=0, keepdims=True)
            s = _score(polished)
            if s < best_score:
                best_score = s
                best_points = polished
    
    if best_points is None:
        best_points = _grid_init()
    
    # Ensure distinct points
    return np.array(best_points, dtype=float)

# Pre-compute at import time if needed, but construct does the work
