"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Initialization: Hexagonal Grid
    def init_hexagonal():
        # Generate hexagonal lattice points
        # Spacing d
        # We want to fill [-0.5, 0.5]^2
        # Let's estimate d. Area = 1. N=50. Area per circle ~ 0.02.
        # Hexagon area for radius r: sqrt(3)/2 * r^2.
        # d = 2r.
        # Let's just generate a grid and select 50 points.
        
        # Create a range of points
        # Hex grid:
        # x = i * d + (j % 2) * d/2
        # y = j * d * sqrt(3)/2
        
        # Let's try d = 0.14 (radius ~0.07)
        d = 0.14
        pts = []
        range_x = int(np.ceil(1.0 / d)) + 2
        range_y = int(np.ceil(1.0 / (d * np.sqrt(3)/2))) + 2
        
        for j in range(-range_y, range_y + 1):
            for i in range(-range_x, range_x + 1):
                x = i * d + (j % 2) * d / 2
                y = j * d * np.sqrt(3) / 2
                # Center the grid
                # Rough centering: shift by -0.5
                x -= 0.5
                y -= 0.5
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    pts.append((x, y))
        
        pts = np.array(pts)
        
        # If we have more than N points, select the N closest to center?
        # Or just take the first N?
        # Taking closest to center usually gives a more compact packing.
        dists_to_center = np.sum(pts**2, axis=1)
        idx = np.argsort(dists_to_center)
        pts = pts[idx[:N]]
        
        # Add small random jitter to break symmetry and help optimization
        rng = np.random.default_rng(42)
        pts += rng.normal(0, 0.005, pts.shape)
        
        # Clip to ensure they are inside (though optimization will fix)
        pts = np.clip(pts, -0.49, 0.49)
        
        return pts.flatten()

    def potential(x):
        centers = x.reshape(-1, 2)
        
        # Wall distances
        dx = 0.5 - np.abs(centers[:, 0])
        dy = 0.5 - np.abs(centers[:, 1])
        wall_dists = np.minimum(dx, dy)
        
        # Pairwise distances
        # Using cdist is efficient
        from scipy.spatial.distance import cdist
        dists = cdist(centers, centers)
        np.fill_diagonal(dists, 1e10)
        min_pair_dists = np.min(dists, axis=1)
        
        # Combine
        # We want to maximize the minimum of (wall_dists, min_pair_dists)
        # Minimizing sum of inverse powers is a good proxy
        # Use a large power to emphasize the minimum
        # Add epsilon to avoid division by zero
        eps = 1e-4
        
        # Clip distances to be at least eps to avoid inf
        wall_dists_safe = np.clip(wall_dists, eps, None)
        min_pair_dists_safe = np.clip(min_pair_dists, eps, None)
        
        # Potential
        V = np.sum(1.0 / (wall_dists_safe ** 4)) + np.sum(1.0 / (min_pair_dists_safe ** 4))
        
        return V

    def get_radius(centers):
        from scipy.spatial.distance import cdist
        dists = cdist(centers, centers)
        np.fill_diagonal(dists, 1e10)
        min_pair = np.min(dists)
        
        dx = 0.5 - np.abs(centers[:, 0])
        dy = 0.5 - np.abs(centers[:, 1])
        min_wall = np.min(np.minimum(dx, dy))
        
        r = 0.5 * np.min([min_pair, min_wall])
        return r

    best_x = None
    best_score = -1.0
    
    # Run optimization with a few restarts
    num_restarts = 5
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        x0 = init_hexagonal()
        # Add more jitter for subsequent restarts
        if i > 0:
            rng = np.random.default_rng(100 + i)
            x0 += rng.normal(0, 0.01 * i, x0.shape)
            x0 = np.clip(x0.reshape(-1, 2), -0.49, 0.49).flatten()
            
        try:
            res = minimize(potential, x0, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if res.success or res.fun < 1e10:
                x_opt = res.x
                centers = x_opt.reshape(-1, 2)
                
                # Ensure validity: clip to box
                centers = np.clip(centers, -0.5, 0.5)
                
                r = get_radius(centers)
                
                if r > best_score:
                    best_score = r
                    best_x = centers.copy()
                    
        except Exception as e:
            pass
            
    if best_x is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_x = grid[:N]
        
    return best_x
# EVOLVE-BLOCK-END
