# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import scipy.spatial.distance
import time

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    time_limit = 580  # Safety margin
    
    n = 14
    d = 3
    
    def objective(points_flat):
        points = points_flat.reshape(n, d)
        if not np.all(np.isfinite(points)):
            return 0.0
        dists = scipy.spatial.distance.pdist(points)
        if np.any(np.isnan(dists)) or np.any(np.isinf(dists)):
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return -((d_min / d_max) ** 2)

    best_points = None
    best_score = -1.0
    
    def fibonacci_sphere(n):
        points = []
        phi = np.pi * (3. - np.sqrt(5.))
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2
            radius = np.sqrt(1 - y * y)
            theta = phi * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)
    
    # Generate initializations
    initializations = []
    
    # 1. Fibonacci sphere (very good start for sphere packing)
    fib_pts = fibonacci_sphere(n)
    initializations.append(fib_pts.flatten())
    
    # 2. Perturbed Fibonacci spheres
    np.random.seed(42)
    for i in range(20):
        noise = np.random.randn(n, d) * 0.1
        pts = fib_pts + noise
        # Re-center
        pts = pts - np.mean(pts, axis=0)
        initializations.append(pts.flatten())
        
    # 3. Random points in a sphere
    for i in range(30):
        np.random.seed(1000 + i)
        pts = np.random.randn(n, d)
        norms = np.linalg.norm(pts, axis=1)
        pts = pts / norms[:, np.newaxis]
        initializations.append(pts.flatten())
        
    # 4. Random points in a cube
    for i in range(20):
        np.random.seed(2000 + i)
        pts = np.random.rand(n, d) * 2 - 1
        initializations.append(pts.flatten())
        
    # Run optimization
    for init in initializations:
        if time.time() - start_time > time_limit:
            break
            
        try:
            # Use Nelder-Mead for non-smooth objective
            res = scipy.optimize.minimize(objective, init, method='Nelder-Mead', 
                                          options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            
            if res.success:
                score_val = -res.fun
                if score_val > best_score:
                    best_score = score_val
                    best_points = res.x.reshape(n, d)
        except Exception:
            pass
            
    if best_points is None:
        # Fallback
        return np.random.randn(n, d)
        
    return best_points

# EVOLVE-BLOCK-END
