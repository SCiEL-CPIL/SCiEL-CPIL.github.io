# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from itertools import combinations

def _score_fast(points, tri_idx):
    """Vectorized score computation for speed."""
    n = points.shape[0]
    i, j, k = tri_idx
    # Compute cross products for all triangles at once
    x1, y1 = points[i, 0], points[i, 1]
    x2, y2 = points[j, 0], points[j, 1]
    x3, y3 = points[k, 0], points[k, 1]
    areas = np.abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) / 2.0
    a_min = np.min(areas)
    if a_min <= 1e-15:
        return 0.0
    try:
        hull = ConvexHull(points)
        a_hull = hull.volume
    except:
        return 0.0
    if a_hull <= 1e-15:
        return 0.0
    return a_min / a_hull

def _score(points, tri_indices):
    """Compute A_min / A_hull ratio using vectorized triangle areas."""
    n = points.shape[0]
    # Vectorized computation of all 286 triangle areas
    i, j, k = zip(*tri_indices)
    i = np.array(i)
    j = np.array(j)
    k = np.array(k)
    
    x1, y1 = points[i, 0], points[i, 1]
    x2, y2 = points[j, 0], points[j, 1]
    x3, y3 = points[k, 0], points[k, 1]
    
    areas = np.abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) / 2.0
    a_min = np.min(areas)
    if a_min <= 1e-15:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        a_hull = hull.volume
    except:
        return 0.0
    
    if a_hull <= 1e-15:
        return 0.0
    
    return a_min / a_hull

def _generate_initializations(n, rng):
    """Generate diverse initial configurations."""
    inits = []
    
    # 1: Regular n-gon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2: Concentric rings (1 center + 4 inner + 8 outer)
    pts = np.zeros((n, 2))
    pts[0] = [0, 0]
    inner_angles = np.linspace(0, 2*np.pi, 4, endpoint=False)
    pts[1:5] = np.column_stack([0.5*np.cos(inner_angles), 0.5*np.sin(inner_angles)])
    outer_angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    pts[5:] = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inits.append(pts)
    
    # 3: Two rings (7 outer + 6 inner)
    pts = np.zeros((n, 2))
    outer_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    pts[:7] = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inner_angles = np.linspace(np.pi/6, 2*np.pi + np.pi/6, 6, endpoint=False)
    pts[7:] = np.column_stack([0.6*np.cos(inner_angles), 0.6*np.sin(inner_angles)])
    inits.append(pts)
    
    # 4: Grid pattern
    grid = np.array([
        [-1, -1], [0, -1], [1, -1],
        [-1, 0], [0, 0], [1, 0],
        [-1, 1], [0, 1], [1, 1],
        [-0.5, -1.5], [0.5, -1.5],
        [-0.5, 1.5], [0.5, 1.5]
    ])
    inits.append(grid)
    
    # 5-10: Random initializations
    for _ in range(6):
        pts = rng.random((n, 2)) * 2 - 1
        inits.append(pts)
    
    return inits

def _local_search(points, tri_indices, start_time, deadline, rng):
    """Simulated annealing-like local search."""
    current = points.copy()
    current_score = _score(current, tri_indices)
    best = current.copy()
    best_score = current_score
    
    T = 0.01  # Initial temperature
    cooling = 0.995
    
    while time.time() < deadline - 10:
        # Generate a random perturbation
        noise_scale = 0.05 + 0.05 * T
        trial = current + rng.normal(0, noise_scale, size=current.shape)
        
        # Keep points bounded
        trial = np.clip(trial, -5, 5)
        
        trial_score = _score(trial, tri_indices)
        
        # Acceptance criterion (simulated annealing)
        if trial_score >= current_score or rng.random() < np.exp((trial_score - current_score) / max(T, 1e-10)):
            current = trial
            current_score = trial_score
            
            if current_score > best_score:
                best_score = current_score
                best = current.copy()
        
        T *= cooling
        if T < 1e-6:
            break
    
    return best, best_score

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 600.0  # 600 second deadline with safety margin
    
    # Precompute all triangle indices
    tri_indices = list(combinations(range(n), 3))
    
    # Generate initial configurations
    rng = np.random.default_rng(seed=42)
    inits = _generate_initializations(n, rng)
    
    best_points = None
    best_score = 0.0
    
    # Evaluate initial configurations
    for init in inits:
        score = _score(init, tri_indices)
        if score > best_score:
            best_score = score
            best_points = init.copy()
    
    # Multi-start local search with simulated annealing
    n_restarts = 50
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        # Create perturbed starting point
        if best_points is not None:
            rng_r = np.random.default_rng(seed=1000 + restart)
            perturbation = rng_r.normal(0, 0.1, size=(n, 2))
            current = best_points + perturbation
        else:
            current = inits[restart % len(inits)].copy()
        
        # Run local search
        local_best, local_score = _local_search(current, tri_indices, start_time, deadline, rng)
        
        if local_score > best_score:
            best_score = local_score
            best_points = local_best.copy()
    
    # Final refinement with smaller steps
    if best_points is not None and time.time() < deadline - 20:
        current = best_points.copy()
        current_score = _score(current, tri_indices)
        
        # Fine-tuning with very small steps
        for _ in range(1000):
            if time.time() > deadline - 15:
                break
            
            improved = False
            for i in range(n):
                for d in range(2):
                    for direction in [1, -1]:
                        trial = current.copy()
                        trial[i, d] += direction * 0.005
                        
                        trial_score = _score(trial, tri_indices)
                        if trial_score > current_score:
                            current = trial
                            current_score = trial_score
                            improved = True
            
            if not improved:
                break
            
            if current_score > best_score:
                best_score = current_score
                best_points = current.copy()
    
    # Ensure valid output
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Verify the result is valid
    if not np.all(np.isfinite(best_points)):
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points

# EVOLVE-BLOCK-END
