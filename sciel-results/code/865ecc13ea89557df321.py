import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    n = 16
    start_time = time.time()
    time_limit = 590.0  # Safety margin before 630s kill

    def objective(params):
        """
        Objective function for minimization.
        We minimize -log(R) = -2*(log(d_min) - log(d_max)) = 2*(log(d_max) - log(d_min)).
        This is equivalent to maximizing R.
        """
        points = params.reshape(n, 2)
        d = pdist(points)
        d_min = np.min(d)
        d_max = np.max(d)
        
        if d_min <= 1e-10 or d_max <= 1e-10:
            return 1e10  # Large penalty for invalid configs
        
        # We want to minimize log(d_max) - log(d_min)
        return 2.0 * (np.log(d_max) - np.log(d_min))

    def score(points):
        """Calculate the score R = (d_min / d_max)^2"""
        d = pdist(points)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def generate_hex_cluster():
        """Generate a hexagonal cluster of 16 points"""
        s3 = np.sqrt(3)
        p0 = [(0,0)]
        p1 = [(-0.5, s3/2), (0.5, s3/2), (1.5, s3/2)]
        p2 = [(-1, s3), (0, s3), (1, s3), (2, s3), (3, s3)]
        p3 = [(-1.5, 3*s3/2), (-0.5, 3*s3/2), (0.5, 3*s3/2), (1.5, 3*s3/2), (2.5, 3*s3/2), (3.5, 3*s3/2), (4.5, 3*s3/2)]
        
        all_pts = p0 + p1 + p2 + p3
        arr = np.array(all_pts)
        center = arr.mean(axis=0)
        arr -= center
        return arr

    def generate_square_grid():
        """Generate a 4x4 square grid"""
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        xx, yy = np.meshgrid(x, y)
        return np.column_stack([xx.ravel(), yy.ravel()])

    def generate_rings(n_inner, n_outer):
        """Generate ring configuration"""
        p_center = np.array([0.0, 0.0])
        r1 = 0.5
        angles1 = np.linspace(0, 2*np.pi, n_inner, endpoint=False)
        p_inner = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        
        r2 = 1.0
        angles2 = np.linspace(0, 2*np.pi, n_outer, endpoint=False)
        angles2 += np.pi / n_outer
        p_outer = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        
        return np.vstack([p_center, p_inner, p_outer])

    best_points = None
    best_score = -1.0

    # List of initial configurations
    inits = [
        generate_hex_cluster(),
        generate_square_grid(),
        generate_rings(6, 9),
        generate_rings(5, 10),
        generate_rings(7, 8),
        generate_rings(4, 11),
        generate_rings(8, 7),
    ]
    
    # Add random inits
    np.random.seed(42)
    for _ in range(10):
        rand_pts = np.random.rand(n, 2) * 2 - 1
        inits.append(rand_pts)
        
    # Add jittered versions of the best known (hex)
    hex_pts = generate_hex_cluster()
    for _ in range(5):
        jitter = np.random.normal(0, 0.1, (n, 2))
        inits.append(hex_pts + jitter)

    # Optimization using Powell's method (derivative-free, robust to non-smoothness)
    for i, init_pts in enumerate(inits):
        if time.time() - start_time > time_limit:
            break
            
        x0 = init_pts.flatten()
        
        try:
            res = minimize(
                objective, 
                x0, 
                method='Powell', 
                options={'maxiter': 1000, 'xtol': 1e-8, 'ftol': 1e-8}
            )
            
            # Check time limit inside the loop isn't possible directly in minimize, 
            # but Powell is fast enough per iteration. 
            # We rely on the outer loop break.
            
            new_points = res.x.reshape(n, 2)
            new_score = score(new_points)
            
            if new_score > best_score:
                best_score = new_score
                best_points = new_points
                
        except Exception:
            continue

    # If no valid points were found (should not happen), return a default
    if best_points is None:
        best_points = generate_hex_cluster()

    return best_points
