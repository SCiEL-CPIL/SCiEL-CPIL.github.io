"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import sys

def _score(pts):
    """Compute the radius score for a configuration of points."""
    if pts.shape[0] != N:
        return -1.0
    if not np.all(np.isfinite(pts)):
        return -1.0
    # Check bounds
    if np.any(pts[:, 0] < -0.5) or np.any(pts[:, 0] > 0.5) or \
       np.any(pts[:, 1] < -0.5) or np.any(pts[:, 1] > 0.5):
        return -1.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Distance to boundary
    d_bound = np.minimum(
        np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5),
        np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5)
    )
    min_pair = np.min(dists[np.triu_indices(N, k=1)])
    min_bound = np.min(d_bound)
    return min(min_pair, 2 * min_bound) / 2.0

def _initial_hexagonal():
    """Create a hexagonal packing initial configuration."""
    # For 50 circles, use a hexagonal arrangement
    # Start with a grid and perturb
    m = 7  # 7x7 = 49, close to 50
    # Use hexagonal lattice
    pts = []
    # Hexagonal lattice with spacing
    # Try different spacings
    for row in range(8):
        for col in range(8):
            x = col * 0.14 - 0.45
            y = row * 0.12 - 0.42
            if row % 2 == 1:
                x += 0.07
            if x > -0.5 and x < 0.5 and y > -0.5 and y < 0.5:
                pts.append([x, y])
    pts = np.array(pts[:N])
    if len(pts) < N:
        # Fill remaining with random
        while len(pts) < N:
            pts = np.vstack([pts, np.array([random.uniform(-0.4, 0.4), random.uniform(-0.4, 0.4)])])
    return pts

def _initial_random():
    """Create a random initial configuration."""
    return np.random.uniform(-0.45, 0.45, (N, 2))

def _initial_grid():
    """Create a grid initial configuration."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]

def _optimize_from(pts0, max_iter=5000, tol=1e-12):
    """Optimize a configuration using gradient-based method."""
    # Use a smooth objective: maximize minimum distance
    # We'll use a soft-min approach with iterative refinement
    
    best_pts = pts0.copy()
    best_score = _score(pts0)
    
    # Use Nelder-Mead or BFGS with multiple restarts
    # For this problem, we'll use a custom approach
    
    # First, try to improve using local search with small perturbations
    for iteration in range(max_iter):
        # Compute current score
        curr_score = _score(best_pts)
        
        # If score improved, update best
        if curr_score > best_score:
            best_score = curr_score
        
        # Generate perturbation
        step_size = 0.01 * (1 - iteration / max_iter) + 0.001
        noise = np.random.normal(0, step_size, (N, 2))
        new_pts = best_pts + noise
        
        # Clamp to bounds
        new_pts = np.clip(new_pts, -0.5, 0.5)
        
        new_score = _score(new_pts)
        if new_score > curr_score:
            best_pts = new_pts
    
    return best_pts

def _optimize_scipy(pts0, method='Nelder-Mead', maxiter=2000):
    """Optimize using scipy's minimize."""
    def neg_score(pts):
        pts = pts.reshape(N, 2)
        return -_score(pts)
    
    x0 = pts0.flatten()
    
    result = minimize(neg_score, x0, method=method, 
                      options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12})
    
    return result.x.reshape(N, 2)

def construct():
    """Construct the optimal packing configuration."""
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    best_pts = None
    best_score = -1.0
    
    # Try multiple initializations and optimization strategies
    initializations = [
        ('hex', _initial_hexagonal()),
        ('random1', _initial_random()),
        ('random2', _initial_random()),
        ('grid', _initial_grid()),
    ]
    
    # Add more random inits
    for i in range(3):
        initializations.append((f'rand{i}', _initial_random()))
    
    for name, pts0 in initializations:
        if time.time() > deadline - 20:
            break
        
        # First try local search
        pts_opt = _optimize_from(pts0, max_iter=3000)
        score = _score(pts_opt)
        
        if score > best_score:
            best_score = score
            best_pts = pts_opt.copy()
        
        # Try scipy optimization if time allows
        if time.time() < deadline - 10:
            try:
                pts_scipy = _optimize_scipy(pts_opt, method='Nelder-Mead', maxiter=500)
                score_scipy = _score(pts_scipy)
                if score_scipy > best_score:
                    best_score = score_scipy
                    best_pts = pts_scipy.copy()
            except Exception:
                pass
    
    # Final polish: try to improve with smaller steps
    if best_pts is not None and time.time() < deadline - 5:
        # Gradient-based refinement
        for _ in range(100):
            if time.time() > deadline - 3:
                break
            curr_score = _score(best_pts)
            step_size = 0.002
            noise = np.random.normal(0, step_size, (N, 2))
            new_pts = np.clip(best_pts + noise, -0.5, 0.5)
            new_score = _score(new_pts)
            if new_score > curr_score:
                best_pts = new_pts
    
    if best_pts is None:
        # Fallback to grid
        best_pts = _initial_grid()
    
    # Ensure validity
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts
# EVOLVE-BLOCK-END
