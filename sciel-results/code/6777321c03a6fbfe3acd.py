import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution
import time

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 590  # 10s margin
    
    def get_score(points):
        # points: (14, 3)
        if not np.all(np.isfinite(points)):
            return 0.0
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        d_min = np.min(dists)
        return (d_min / d_max) ** 2

    def objective(points_flat):
        points = points_flat.reshape(n, d)
        return -get_score(points)
        
    # Initial guess 1: Icosahedron vertices + 2 face centers
    # Icosahedron coordinates (edge length 1)
    phi = (1 + np.sqrt(5)) / 2
    # Vertices
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1],
        [phi, 0, -1], [-phi, 0, -1]
    ])
    
    # Normalize ico_verts to unit sphere
    norms = np.linalg.norm(ico_verts, axis=1)
    ico_verts = ico_verts / norms[:, np.newaxis]
    
    # Face centers of icosahedron are along the 5-fold axes.
    # The 5-fold axes point to the centers of the faces.
    # We can approximate the face centers by averaging the vertices of a face.
    # Or simply add 2 points at (0,0,1) and (0,0,-1) if we rotate the ico?
    # Actually, the 5-fold axes are not aligned with the standard ico coords above.
    # Let's just add 2 points at the origin? No, that minimizes min distance.
    # Let's add 2 points at (0,0,1) and (0,0,-1) and let the optimizer sort it out.
    # Or better, use the known optimal structure: 12 ico + 2 poles on the 3-fold axis?
    # If we use the 3-fold axis, the ico already has vertices at the poles.
    # So adding 2 more points on the poles is bad.
    # Let's try adding 2 points on the 5-fold axes.
    # To find 5-fold axes directions, we can look at the symmetry.
    # For now, let's just use ico_verts + 2 random points on the sphere.
    
    init1 = np.vstack([ico_verts, np.array([[0, 0, 1], [0, 0, -1]])])
    # Normalize init1? No, let the optimizer handle scale, but it's good to start on sphere.
    init1_norms = np.linalg.norm(init1, axis=1)
    init1 = init1 / init1_norms[:, np.newaxis]
    
    best_score = -np.inf
    best_points = init1
    
    # Try a few initializations
    inits = [init1]
    # Random inits
    for i in range(5):
        rng = np.random.default_rng(i)
        rand_pts = rng.normal(size=(n, d))
        rand_pts /= np.linalg.norm(rand_pts, axis=1)[:, np.newaxis]
        inits.append(rand_pts)
        
    for init in inits:
        if time.time() > deadline:
            break
            
        try:
            # Use Powell or Nelder-Mead
            # Flatten for optimizer
            x0 = init.flatten()
            
            # Run Nelder-Mead
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.success or res.fun < -best_score:
                pts = res.x.reshape(n, d)
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = pts
                    
        except Exception:
            pass
            
    # If Nelder-Mead didn't do well, try Differential Evolution (slower but robust)
    # Only if time permits
    if time.time() < deadline - 100:
        try:
            bounds = [(-1, 1)] * (n * d)
            # DE is slow, use small pop
            res_de = differential_evolution(objective, bounds, maxiter=50, popsize=10, tol=1e-8, polish=True)
            if res_de.fun < -best_score:
                pts = res_de.x.reshape(n, d)
                score = -res_de.fun
                if score > best_score:
                    best_score = score
                    best_points = pts
        except Exception:
            pass

    # Final cleanup: Ensure finite and correct shape
    best_points = np.nan_to_num(best_points, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return best_points
