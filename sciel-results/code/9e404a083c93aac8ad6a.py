import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

# EVOLVE-BLOCK-START
def construct():
    # Initial configuration: 4x4 grid minus 3 corners
    # Grid points (0,0) to (3,3)
    grid = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
    
    # Remove 3 corners to get 13 points.
    # Let's remove (0,0), (3,0), (0,3) to keep symmetry-ish or just pick 3.
    # Actually, removing corners usually creates larger minimum triangles.
    # Indices in grid:
    # (0,0) is 0
    # (3,0) is 3
    # (0,3) is 12
    # Let's remove indices 0, 3, 12.
    indices_to_remove = [0, 3, 12]
    keep_indices = [i for i in range(16) if i not in indices_to_remove]
    
    # Sort to ensure consistent ordering for optimization
    keep_indices.sort()
    initial_points = grid[keep_indices]
    
    # Normalize initial points to have reasonable scale
    # Center them
    initial_points -= np.mean(initial_points, axis=0)
    
    # Fix first two points to break affine symmetry
    # P0 = (0,0), P1 = (1,0)
    # We need to map initial_points[0] to (0,0) and initial_points[1] to (1,0)
    # This requires an affine transform.
    p0 = initial_points[0]
    p1 = initial_points[1]
    
    # Vector v = p1 - p0
    v = p1 - p0
    len_v = np.linalg.norm(v)
    
    # Transformation:
    # 1. Translate p0 to origin
    # 2. Rotate/Scale so p1 becomes (1,0)
    # Matrix M such that M * (p_i - p0) / len_v ... 
    # Simpler:
    # x' = (p - p0)
    # Rotate x' so that v is along x-axis.
    # Then scale so |v'| = 1.
    
    # Angle of v
    theta = np.arctan2(v[1], v[0])
    rot_matrix = np.array([
        [np.cos(-theta), -np.sin(-theta)],
        [np.sin(-theta), np.cos(-theta)]
    ])
    
    # Apply to all points
    shifted = initial_points - p0
    rotated = shifted @ rot_matrix.T
    scaled = rotated / len_v
    
    # Now scaled[0] is (0,0) and scaled[1] is (1,0)
    # Let's verify
    assert np.allclose(scaled[0], [0,0])
    assert np.allclose(scaled[1], [1,0])
    
    # Variables to optimize: coordinates of points 2 to 12 (11 points * 2 = 22 vars)
    x0 = scaled[2:].flatten()
    
    def objective(x):
        # Reconstruct points
        pts = np.zeros((13, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = x.reshape(11, 2)
        
        # Calculate min triangle area
        min_area = float('inf')
        for i in range(13):
            for j in range(i+1, 13):
                for k in range(j+1, 13):
                    p1, p2, p3 = pts[i], pts[j], pts[k]
                    # Area = 0.5 * |det(p2-p1, p3-p1)|
                    v1 = p2 - p1
                    v2 = p3 - p1
                    det = v1[0]*v2[1] - v1[1]*v2[0]
                    area = 0.5 * abs(det)
                    if area < min_area:
                        min_area = area
        
        if min_area == 0:
            return 0.0 # Degenerate, bad
            
        # Calculate hull area
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume # In 2D, volume is area
        except:
            return 0.0 # Degenerate hull
            
        if hull_area == 0:
            return 0.0
            
        ratio = min_area / hull_area
        return -ratio # Minimize negative ratio to maximize ratio
        
    # Use Powell method which doesn't require gradients and is robust
    # Nelder-Mead can also work but Powell is often better for this
    result = minimize(objective, x0, method='Powell', options={'maxiter': 200, 'xtol': 1e-5, 'disp': False})
    
    x_opt = result.x
    pts_opt = np.zeros((13, 2))
    pts_opt[0] = [0.0, 0.0]
    pts_opt[1] = [1.0, 0.0]
    pts_opt[2:] = x_opt.reshape(11, 2)
    
    # The score is invariant to affine transformations, so we can return pts_opt directly.
    # The evaluator will normalize by hull area.
    return pts_opt
# EVOLVE-BLOCK-END
