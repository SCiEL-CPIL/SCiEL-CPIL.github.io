"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def _covering_radius_deg(pts):
    """Compute covering radius in degrees using ConvexHull circumradii."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.inf
    radii = hull.circumradii
    # Filter out non-finite
    radii = radii[np.isfinite(radii)]
    if len(radii) == 0:
        return np.inf
    # Circumradii are angular distances in radians for unit sphere
    max_rad = np.max(radii)
    return np.degrees(max_rad)

def _fast_covering_radius_deg(pts, n_samples=20000):
    """Fast approximate covering radius using random sampling."""
    # Generate random points on sphere
    theta = np.random.uniform(0, 2*np.pi, n_samples)
    phi = np.arccos(2*np.random.uniform(0, 1, n_samples) - 1)
    samples = np.column_stack([
        np.sin(phi)*np.cos(theta),
        np.sin(phi)*np.sin(theta),
        np.cos(phi)
    ])
    # Compute min distance to each sample point
    # dist^2 = 2 - 2*(dot product)
    dots = samples @ pts.T  # (n_samples, N)
    min_dots = np.max(dots, axis=1)  # max dot = min angular distance
    min_dots = np.clip(min_dots, -1, 1)
    max_angle = np.arccos(min_dots)
    return np.degrees(np.max(max_angle))

def _objective(pts):
    """Objective function: covering radius in degrees."""
    return _covering_radius_deg(pts)

def _optimize_from_start(pts0, time_budget=100.0):
    """Local optimization using gradient-free methods."""
    start_time = time.time()
    best_pts = pts0.copy()
    best_score = _objective(best_pts)
    
    if not np.isfinite(best_score):
        return best_pts, best_score
    
    # Use Nelder-Mead style or coordinate descent
    # We'll use a simple iterative approach: move each point to reduce max facet
    n_iter = 0
    step_size = 0.05
    patience = 0
    max_patience = 50
    
    while time.time() - start_time < time_budget:
        n_iter += 1
        improved = False
        
        for i in range(N):
            if time.time() - start_time > time_budget:
                    break
            
            # Try to move point i in a few random directions
            best_local = best_pts[i].copy()
            best_local_score = best_score
            
            for _ in range(20):
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
                for ss in [step_size, step_size*0.5, step_size*0.1]:
                    new_pt = best_pts[i] + ss * direction
                    # Normalize
                    norm = np.linalg.norm(new_pt)
                    if norm < 1e-10:
                        continue
                    new_pt /= norm
                    
                    trial_pts = best_pts.copy()
                    trial_pts[i] = new_pt
                    
                    # Quick check: ensure no coincidence
                    if np.min(np.linalg.norm(trial_pts - trial_pts[i], axis=1)) < 1e-6:
                        continue
                    
                    score = _fast_covering_radius_deg(trial_pts, n_samples=5000)
                    if score < best_local_score - 1e-10:
                        best_local_score = score
                        best_local = new_pt
                        improved = True
            
            if np.linalg.norm(best_local - best_pts[i]) > 1e-10:
                best_pts[i] = best_local
        
        # Recompute exact score
        new_score = _objective(best_pts)
        if new_score < best_score - 1e-10:
            best_score = new_score
            patience = 0
        else:
            patience += 1
        
        if patience > max_patience:
            break
        
        # Reduce step size occasionally
        if n_iter % 50 == 0:
            step_size *= 0.8
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Multiple starts
    best_pts = None
    best_score = np.inf
    
    # Start 1: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts1 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Start 2: Random
    np.random.seed(42)
    pts2 = np.random.randn(N, 3)
    pts2 /= np.linalg.norm(pts2, axis=1, keepdims=True)
    
    # Start 3: Another random
    np.random.seed(123)
    pts3 = np.random.randn(N, 3)
    pts3 /= np.linalg.norm(pts3, axis=1, keepdims=True)
    
    # Start 4: Grid-based
    pts4 = np.zeros((N, 3))
    idx = 0
    for lat in np.linspace(-np.pi/2 + 0.1, np.pi/2 - 0.1, 7):
        for lon in np.linspace(0, 2*np.pi, 8, endpoint=False):
            if idx >= N:
                break
            pts4[idx, 0] = np.cos(lat) * np.cos(lon)
            pts4[idx, 1] = np.cos(lat) * np.sin(lon)
            pts4[idx, 2] = np.sin(lat)
            idx += 1
    if idx < N:
        # Fill remaining with random
        remaining = N - idx
        extra = np.random.randn(remaining, 3)
        extra /= np.linalg.norm(extra, axis=1, keepdims=True)
        pts4[idx:] = extra
    
    starts = [pts1, pts2, pts3, pts4]
    
    for pts in starts:
        if time.time() > deadline - 20:
            break
        
        # Evaluate initial
        score = _objective(pts)
        if score < best_score:
            best_score = score
            best_pts = pts.copy()
        
        # Optimize
        time_left = deadline - time.time() - 5
        if time_left > 10:
            opt_pts, opt_score = _optimize_from_start(pts, time_budget=time_left)
            if opt_score < best_score:
                best_score = opt_score
                best_pts = opt_pts
    
    # Final polish with exact evaluation
    if best_pts is not None and time.time() < deadline - 2:
        final_pts, final_score = _optimize_from_start(best_pts, time_budget=deadline - time.time() - 1)
        if final_score < best_score:
            best_pts = final_pts
            best_score = final_score
    
    if best_pts is None:
        best_pts = pts1
    
    # Ensure normalization
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    return best_pts
# EVOLVE-BLOCK-END
