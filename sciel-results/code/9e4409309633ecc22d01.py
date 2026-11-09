import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

def get_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    points = np.asarray(points)
    if points.shape != (N, 2):
        return 0.0
    
    # Check for degeneracy
    if np.any(np.diff(np.sort(points, axis=0), axis=0) == 0):
        # This check is weak, better to rely on area calculation
        pass
        
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0
        
    # Vectorized triangle area calculation
    i_idx, j_idx, k_idx = np.indices((N, N, N))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    p1 = points[i_flat]
    p2 = points[j_flat]
    p3 = points[k_flat]
    
    v1 = p2 - p1
    v2 = p3 - p1
    # Cross product magnitude
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def neg_score(points_flat):
    """
    Objective function for minimization.
    """
    points = points_flat.reshape(N, 2)
    score = get_score(points)
    return -score

def generate_initials():
    """
    Generate several initial configurations.
    """
    configs = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Regular 12-gon + Center
    angles_12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    pts_12 = np.column_stack([np.cos(angles_12), np.sin(angles_12)])
    center = np.array([[0.0, 0.0]])
    configs.append(np.vstack([pts_12, center]))
    
    # 3. Regular 11-gon + 2 inner points (e.g., at radius 0.5, opposite)
    angles_11 = np.linspace(0, 2 * np.pi, 11, endpoint=False)
    pts_11 = np.column_stack([np.cos(angles_11), np.sin(angles_11)])
    inner_1 = np.array([[0.5, 0.0]])
    inner_2 = np.array([[-0.5, 0.0]])
    configs.append(np.vstack([pts_11, inner_1, inner_2]))
    
    # 4. Hexagonal Cluster: 1 center, 6 at d=1, 6 at d=sqrt(3) (corners)
    # Actually, a better hex cluster:
    # (0,0)
    # (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    # Then 6 more at distance 2? Or 12 at distance 1?
    # Let's try 1 center, 6 at d=1, 6 at d=2 (same angles as d=1? No, that's collinear)
    # Let's just use a random jittered grid as a 4th option
    rng = np.random.default_rng(42)
    grid = np.array([(i, j) for i in range(4) for j in range(4)])
    # Remove 3 points to get 13
    # Keep center-ish?
    # Just take first 13
    grid = grid[:13]
    # Jitter
    grid = grid + rng.normal(0, 0.1, grid.shape)
    configs.append(grid)
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    initials = generate_initials()
    best_score = -np.inf
    best_points = None
    
    for init_points in initials:
        if time.time() > deadline - 10.0:
            break
            
        # Flatten for optimizer
        x0 = init_points.flatten()
        
        # Run optimization
        try:
            res = minimize(
                neg_score, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if time.time() > deadline:
                break
                
            candidate_points = res.x.reshape(N, 2)
            score = get_score(candidate_points)
            
            if score > best_score:
                best_score = score
                best_points = candidate_points
                
        except Exception as e:
            pass
            
    if best_points is None:
        # Fallback to the first initial if optimization failed
        best_points = initials[0]
        
    return best_points

# EVOLVE-BLOCK-START
# The code above is the implementation.
# We need to wrap it in the EVOLVE-BLOCK structure as requested by the prompt format,
# or just provide the full program.
# The prompt says "Rewrite the program from scratch... Output the COMPLETE program".
# It also says "Keep the function names and signatures... of the seed program".
# Seed program had:
# N = 13
# def construct():
#    ...
#
# So I will provide the full script with N=13 and construct().
# EVOLVE-BLOCK-END
