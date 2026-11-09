# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools

def _score(points: np.ndarray) -> float:
    """Compute A_min / A_hull for a set of points."""
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    if not np.all(np.isfinite(points)):
        return 0.0
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 1e-12:
        return 0.0
        
    idx = np.arange(n)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    i_flat = i.flatten()
    j_flat = j.flatten()
    k_flat = k.flatten()
    mask = (i_flat < j_flat) & (j_flat < k_flat)
    i_sel = i_flat[mask]
    j_sel = j_flat[mask]
    k_sel = k_flat[mask]
    
    p1 = points[i_sel]
    p2 = points[j_sel]
    p3 = points[k_sel]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = np.abs(cross) * 0.5
    min_area = np.min(areas)
    
    if min_area < 1e-12:
        return 0.0
        
    return min_area / A_hull

def _batch_score(points: np.ndarray, i_sel: np.ndarray, j_sel: np.ndarray, k_sel: np.ndarray) -> np.ndarray:
    """
    Vectorized score calculation for a batch of points.
    points: (B, N, 2)
    Returns: (B,)
    """
    B = points.shape[0]
    scores = np.zeros(B)
    
    # Vectorized triangle areas
    # i_sel, j_sel, k_sel are arrays of length M (286)
    p1 = points[:, i_sel, :] # (B, M, 2)
    p2 = points[:, j_sel, :]
    p3 = points[:, k_sel, :]
    
    cross = (p2[:, :, 0] - p1[:, :, 0]) * (p3[:, :, 1] - p1[:, :, 1]) - \
            (p2[:, :, 1] - p1[:, :, 1]) * (p3[:, :, 0] - p1[:, :, 0])
    areas = np.abs(cross) * 0.5 # (B, M)
    min_areas = np.min(areas, axis=1) # (B,)
    
    # Hull areas (Loop is necessary for ConvexHull)
    for i in range(B):
        try:
            hull = ConvexHull(points[i])
            A_hull = hull.volume
            if A_hull > 1e-12 and min_areas[i] > 1e-12:
                scores[i] = min_areas[i] / A_hull
            else:
                scores[i] = 0.0
        except Exception:
            scores[i] = 0.0
            
    return scores

def _local_search(points: np.ndarray, i_sel: np.ndarray, j_sel: np.ndarray, k_sel: np.ndarray, time_limit: float, batch_size: int = 100) -> np.ndarray:
    """
    Vectorized stochastic local search.
    """
    best_pts = points.copy()
    best_score = _score(best_pts)
    start_time = time.time()
    rng = np.random.default_rng(int(time.time() * 1000) % 100000) # Random seed for search
    
    noise_scale = 0.1
    stagnant_counter = 0
    
    while time.time() < time_limit:
        # Generate batch of candidates
        noise = rng.normal(0, noise_scale, (batch_size, 13, 2))
        candidates = best_pts[None, :, :] + noise
        candidates = np.clip(candidates, 0.01, 0.99)
        
        scores = _batch_score(candidates, i_sel, j_sel, k_sel)
        best_idx = np.argmax(scores)
        
        if scores[best_idx] > best_score:
            best_score = scores[best_idx]
            best_pts = candidates[best_idx].copy()
            stagnant_counter = 0
            # Shrink noise on success
            noise_scale *= 0.9
        else:
            stagnant_counter += 1
            # Expand noise if stuck
            if stagnant_counter > 10:
                noise_scale *= 1.1
                stagnant_counter = 0
            # Occasional reset to a random point if very stuck?
            # Maybe just let noise grow.
            
        # Ensure noise doesn't get too small or too large
        noise_scale = np.clip(noise_scale, 0.001, 0.5)
        
    return best_pts

def heilbronn_convex13() -> np.ndarray:
    n = 13
    start_time = time.time()
    deadline = start_time + 590
    
    # Precompute triangle indices
    idx = np.arange(n)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    i_flat = i.flatten()
    j_flat = j.flatten()
    k_flat = k.flatten()
    mask = (i_flat < j_flat) & (j_flat < k_flat)
    i_sel = i_flat[mask]
    j_sel = j_flat[mask]
    k_sel = k_flat[mask]
    
    best_points = None
    best_score = -1.0
    
    # Initial configurations
    # 1. Grid-like
    grid = np.array([[i/3, j/3] for i in range(4) for j in range(4) if not (i==3 and j==3)])
    grid = np.clip(grid, 0.05, 0.95)
    
    # 2. Hexagonal packing (approx)
    hex_pts = np.array([
        [0.2, 0.2], [0.5, 0.2], [0.8, 0.2],
        [0.35, 0.45], [0.65, 0.45],
        [0.2, 0.7], [0.5, 0.7], [0.8, 0.7],
        [0.35, 0.9], [0.65, 0.9],
        [0.5, 0.35], [0.35, 0.6], [0.65, 0.6]
    ])
    
    # 3. Regular polygon + center
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    radius = 0.4
    poly_pts = np.column_stack([0.5 + radius*np.cos(angles), 0.5 + radius*np.sin(angles)])
    poly_pts = np.vstack([poly_pts, [[0.5, 0.5]]])
    
    # 4. Random seeds
    rng = np.random.default_rng(42)
    random_starts = [rng.random((13, 2)) * 0.9 + 0.05 for _ in range(2)]
    
    starts = [grid, hex_pts, poly_pts] + random_starts
    
    # Allocate time to each start
    num_starts = len(starts)
    time_per_start = (deadline - time.time()) / num_starts
    
    for i, pts in enumerate(starts):
        # Check if we have time
        if time.time() > deadline - 10:
            break
            
        # Score the initial point
        score = _score(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            
        # Run local search
        # Leave some time for the next iterations
        search_deadline = time.time() + time_per_start - 5
        improved = _local_search(pts, i_sel, j_sel, k_sel, search_deadline, batch_size=100)
        improved_score = _score(improved)
        
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved.copy()
            
    if best_points is None:
        best_points = rng.random((13, 2)) * 0.9 + 0.05
        
    return best_points
# EVOLVE-BLOCK-END
