"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools
import time

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (N, 2):
        return 0.0
    
    # Check for degenerate hull
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area < 1e-10:
        return 0.0
        
    # Calculate all triangle areas
    indices = list(itertools.combinations(range(N), 3))
    if not indices:
        return 0.0
    i, j, k = zip(*indices)
    
    p_i = points[i]
    p_j = points[j]
    p_k = points[k]
    
    # Area = 0.5 * | (x_j - x_i)(y_k - y_i) - (x_k - x_i)(y_j - y_i) |
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - (p_k[:, 0] - p_i[:, 0]) * (p_j[:, 1] - p_i[:, 1])
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    if min_area < 1e-10:
        return 0.0
        
    return min_area / hull_area

def _neg_score(flat_points):
    """
    Objective function for minimization (negated score).
    """
    points = flat_points.reshape(N, 2)
    return -_score(points)

def _generate_initial_configs():
    configs = []
    
    # 1. Regular Polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    poly = np.column_stack([np.cos(angles), np.sin(angles)])
    configs.append(poly.flatten())
    
    # 2. 4x4 Grid (16 points), remove 2
    grid_4x4 = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
    grid_4x4 = grid_4x4 / 3.0
    
    # Remove 2 random points
    for _ in range(10):
        idx_to_remove = np.random.choice(16, 2, replace=False)
        mask = np.ones(16, dtype=bool)
        mask[idx_to_remove] = False
        pts = grid_4x4[mask]
        # Add small noise
        pts = pts + np.random.normal(0, 0.05, pts.shape)
        configs.append(pts.flatten())
        
    # 3. 3x5 Grid (15 points), remove 1
    grid_3x5 = np.array([(i, j) for i in range(3) for j in range(5)], dtype=float)
    grid_3x5 = grid_3x5 / 4.0
    for _ in range(10):
        idx_to_remove = np.random.choice(15, 1, replace=False)
        mask = np.ones(15, dtype=bool)
        mask[idx_to_remove] = False
        pts = grid_3x5[mask]
        pts = pts + np.random.normal(0, 0.05, pts.shape)
        configs.append(pts.flatten())
        
    # 4. Perturbed Regular Polygon
    for _ in range(10):
        pts = poly + np.random.normal(0, 0.2, poly.shape)
        configs.append(pts.flatten())
        
    # 5. Hexagonal Packing Approximation
    # Generate a larger hex grid and pick 14 closest to center
    hex_pts = []
    for q in range(-4, 5):
        for r in range(-4, 5):
            x = q + 0.5 * r
            y = np.sqrt(3) / 2 * r
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    center = np.mean(hex_pts, axis=0)
    dists = np.linalg.norm(hex_pts - center, axis=1)
    hex_pts = hex_pts[np.argsort(dists)]
    hex_pts = hex_pts[:14]
    # Normalize
    hex_pts = hex_pts - hex_pts.mean(axis=0)
    norm = np.max(np.abs(hex_pts))
    if norm > 0:
        hex_pts = hex_pts / norm
    configs.append(hex_pts.flatten())
    
    # 6. Random points
    for _ in range(10):
        pts = np.random.rand(N, 2)
        configs.append(pts.flatten())
        
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = 0.0
    best_points = np.zeros((N, 2))
    
    # Set random seed for reproducibility of the search process itself
    np.random.seed(42)
    
    initial_configs = _generate_initial_configs()
    
    # Shuffle configs to ensure diverse starting points are tried early
    np.random.shuffle(initial_configs)
    
    # Optimize each configuration
    for i, init in enumerate(initial_configs):
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                _neg_score, 
                init, 
                method='Nelder-Mead', 
                options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 1000}
            )
            
            score = -res.fun
            if score > best_score:
                best_score = score
                best_points = res.x.reshape(N, 2)
                
        except Exception:
            continue
            
    # If no valid point found (should not happen), return a valid regular polygon
    if best_score <= 0:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
# EVOLVE-BLOCK-END
