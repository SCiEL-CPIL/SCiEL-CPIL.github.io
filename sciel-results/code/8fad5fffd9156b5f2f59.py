import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 300

def construct():
    t0 = time.time()
    # 150s limit. Keep a large safety margin (15s) to avoid timeout.
    DEADLINE = t0 + 130.0

    def elapsed():
        return time.time() - t0

    def score(points):
        """
        Calculate the maximum radius for the given centers.
        points: (N, 2) array.
        """
        if points.shape[0] < 2:
            return 0.0
        
        # Use cdist for faster pairwise distance calculation
        dist = cdist(points, points)
        
        # Set diagonal to infinity to ignore self-distance
        np.fill_diagonal(dist, np.inf)
        
        min_dist = np.min(dist)
        
        # Calculate maximum distance from origin
        norms = np.linalg.norm(points, axis=1)
        max_norm = np.max(norms)
        
        # Radius is limited by both pair distances and boundary
        r_pair = 0.5 * min_dist
        r_bound = 1.0 - max_norm
        
        return min(r_pair, r_bound)

    def gen_hex_seed():
        """
        Generate a hexagonal lattice seed.
        This is fast and provides a good initial packing.
        """
        # Estimate initial spacing
        # r ~ sqrt(0.9 / N)
        r_est = np.sqrt(0.9 / N)
        spacing = 2.0 * r_est
        
        k = int(1.0 / spacing) + 3
        pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 1.0:
                    pts.append([x, y])
        
        pts = np.array(pts)
        
        # If we have more than N points, select the ones closest to the center
        if len(pts) > N:
            radii = np.linalg.norm(pts, axis=1)
            idx = np.argsort(radii)
            pts = pts[idx[:N]]
        elif len(pts) < N:
            # Shrink spacing and retry
            while len(pts) < N:
                spacing *= 0.95
                k = int(1.0 / spacing) + 3
                pts = []
                for j in range(-k, k + 1):
                    for i in range(-k, k + 1):
                        x = (i + 0.5 * (j % 2)) * spacing
                        y = j * spacing * np.sqrt(3.0) / 2.0
                        if x*x + y*y <= 1.0:
                            pts.append([x, y])
                pts = np.array(pts)
                if len(pts) > N:
                    radii = np.linalg.norm(pts, axis=1)
                    idx = np.argsort(radii)
                    pts = pts[idx[:N]]
                elif len(pts) == N:
                    break
        
        return pts

    # Initial Seed
    # Use hex seed as it is fast and robust
    best_pts = gen_hex_seed()
    best_score = score(best_pts)
    
    # Optimization Loop
    # We use L-BFGS-B.
    # We perform a few short optimization runs to refine the packing.
    
    current_pts = best_pts.copy()
    
    # Limit iterations strictly to prevent long-running calls
    max_iter_per_step = 10
    
    # Number of restarts
    num_restarts = 5
    
    for _ in range(num_restarts):
        if elapsed() > DEADLINE - 10.0:
            break
            
        # Perturb the current solution for restart
        noise_scale = 0.005
        current_pts = current_pts + np.random.normal(0, noise_scale, current_pts.shape)
        
        # Ensure points are roughly inside
        current_pts = np.clip(current_pts, -1.0, 1.0)
        
        x0 = current_pts.flatten()
        
        try:
            # Run optimizer with tight limits
            res = minimize(
                lambda x: -score(x.reshape(N, 2)), 
                x0, 
                method='L-BFGS-B', 
                options={'maxiter': max_iter_per_step, 'ftol': 1e-8, 'gtol': 1e-4, 'maxls': 5}
            )
            
            if res.x is not None:
                cand_pts = res.x.reshape(N, 2)
                
                # Validate
                if np.all(np.isfinite(cand_pts)):
                    cand_score = score(cand_pts)
                    
                    if cand_score > best_score:
                        best_score = cand_score
                        best_pts = cand_pts
                        current_pts = best_pts.copy()
                        
        except Exception:
            pass

    return best_pts
