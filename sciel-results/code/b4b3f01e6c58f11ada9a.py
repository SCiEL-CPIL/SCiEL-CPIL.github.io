# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize
import scipy.spatial.distance

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    
    start_time = time.time()
    deadline = 570  # 30s safety margin
    
    def normalize(points):
        """Shift centroid to 0 and scale so max distance is 1."""
        points = points.copy()
        centroid = np.mean(points, axis=0)
        points -= centroid
        dists = scipy.spatial.distance.pdist(points)
        max_dist = np.max(dists)
        if max_dist > 1e-12:
            points /= max_dist
        return points

    def score(points):
        """Calculate the score R = (d_min / d_max)^2."""
        dists = scipy.spatial.distance.pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(x_flat):
        """Objective function for minimization: -score(normalized points)."""
        points = x_flat.reshape(n, d)
        points = normalize(points)
        s = score(points)
        return -s

    best_score = -1.0
    best_points = None
    
    # Generate multiple initial configurations
    inits = []
    
    # 1. Random points
    np.random.seed(42)
    for _ in range(5):
        p = np.random.randn(n, d)
        inits.append(p.flatten())
        
    # 2. Grid
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    xx, yy = np.meshgrid(x, y)
    p_grid = np.column_stack((xx.ravel(), yy.ravel()))
    inits.append(p_grid.flatten())
    
    # 3. Circle
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    p_circle = np.column_stack((np.cos(angles), np.sin(angles)))
    inits.append(p_circle.flatten())
    
    # 4. Two circles (inner/outer)
    angles_outer = np.linspace(0, 2*np.pi, 8, endpoint=False)
    p_outer = np.column_stack((np.cos(angles_outer), np.sin(angles_outer)))
    angles_inner = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
    p_inner = 0.5 * np.column_stack((np.cos(angles_inner), np.sin(angles_inner)))
    p_two_circ = np.vstack((p_outer, p_inner))
    inits.append(p_two_circ.flatten())

    for init in inits:
        if time.time() > deadline:
            break
            
        try:
            res = scipy.optimize.minimize(
                objective, 
                init, 
                method='Powell', 
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            opt_points = res.x.reshape(n, d)
            opt_points = normalize(opt_points)
            s = score(opt_points)
            
            if s > best_score:
                best_score = s
                best_points = opt_points
                
        except Exception:
            continue
            
    if best_points is None:
        # Fallback
        return np.random.randn(n, d)
        
    return best_points

# EVOLVE-BLOCK-END
