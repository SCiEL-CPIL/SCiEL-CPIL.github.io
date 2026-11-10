# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    n = 14
    d = 3
    
    start_time = time.time()
    deadline = start_time + 600.0  # Safety margin before 630s kill

    def metric(x):
        """
        Computes the negative squared ratio of min to max distance.
        We minimize this to maximize the ratio.
        """
        pts = x.reshape(n, d)
        dists = pdist(pts)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        d_min = np.min(dists)
        ratio = d_min / d_max
        return -(ratio ** 2)

    def generate_icosahedron():
        phi = (1 + np.sqrt(5)) / 2
        pts = [
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ]
        return np.array(pts)

    def generate_cuboctahedron():
        pts = [
            [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
            [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
            [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1]
        ]
        return np.array(pts)

    def generate_dodecahedron():
        phi = (1 + np.sqrt(5)) / 2
        inv_phi = 1 / phi
        pts = [
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
            [0, inv_phi, phi], [0, inv_phi, -phi], [0, -inv_phi, phi], [0, -inv_phi, -phi],
            [inv_phi, phi, 0], [inv_phi, -phi, 0], [-inv_phi, phi, 0], [-inv_phi, -phi, 0],
            [phi, 0, inv_phi], [phi, 0, -inv_phi], [-phi, 0, inv_phi], [-phi, 0, -inv_phi]
        ]
        return np.array(pts)

    best_score = -np.inf
    best_points = None
    rng = np.random.default_rng(42)

    # List of seed generators
    def seed_ico_plus_2():
        pts = generate_icosahedron()
        # Add 2 points at face centers (approx) or random
        # Face centers of icosahedron are not trivial to generate directly without geometry calcs.
        # Let's just add 2 random points near the surface.
        p1 = pts[0] + rng.normal(0, 0.1, 3)
        p2 = pts[1] + rng.normal(0, 0.1, 3)
        return np.vstack([pts, [p1, p2]])

    def seed_cubo_plus_2():
        pts = generate_cuboctahedron()
        p1 = pts[0] + rng.normal(0, 0.1, 3)
        p2 = pts[1] + rng.normal(0, 0.1, 3)
        return np.vstack([pts, [p1, p2]])

    def seed_dodeca_minus_6():
        pts = generate_dodecahedron()
        # Remove 6 points randomly
        idx = rng.choice(20, 6, replace=False)
        mask = np.ones(20, dtype=bool)
        mask[idx] = False
        return pts[mask]

    def seed_random_sphere():
        # Random points on unit sphere
        pts = rng.normal(0, 1, (14, 3))
        norms = np.linalg.norm(pts, axis=1)
        pts = pts / norms[:, np.newaxis]
        return pts

    def seed_lattice():
        # Points from a cubic lattice, scaled
        pts = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                for k in range(-2, 3):
                    if i*i + j*j + k*k <= 10: # Sphere of radius sqrt(10)
                        pts.append([i, j, k])
        pts = np.array(pts)
        # Select 14 points
        # Simple selection: take first 14? Or random 14?
        # Let's take 14 points that are "spread out"
        # Just random 14 for now
        idx = rng.choice(len(pts), 14, replace=False)
        return pts[idx]

    seeds = [
        seed_ico_plus_2,
        seed_cubo_plus_2,
        seed_dodeca_minus_6,
        seed_random_sphere,
        seed_lattice
    ]

    # Optimization loop
    # We will try each seed multiple times with small perturbations
    num_restarts_per_seed = 5
    
    for seed_func in seeds:
        for r in range(num_restarts_per_seed):
            if time.time() > deadline:
                break
            
            # Generate seed
            try:
                x0 = seed_func().flatten()
            except Exception:
                continue
            
            # Add small noise to avoid local minima identical to seed
            if r > 0:
                x0 = x0 + rng.normal(0, 0.05, x0.shape)
            
            # Run optimization
            # Nelder-Mead is good for non-smooth, low-to-mid dimension
            # Powell is also good
            # Let's try Nelder-Mead first
            try:
                res = minimize(metric, x0, method='Nelder-Mead', 
                               options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = res.x.reshape(n, d)
            except Exception:
                pass
            
            # If time allows, try Powell as well for the best found so far from this seed family?
            # Or just try Powell on the initial seed if Nelder-Mead didn't improve much?
            # Let's stick to Nelder-Mead for speed, it's usually fast enough.
            # We can increase maxiter if time permits.

    # Final check
    if best_points is None:
        # Fallback to random
        best_points = rng.normal(0, 1, (n, d))
    
    # Ensure finite
    if not np.all(np.isfinite(best_points)):
        best_points = np.zeros((n, d))
        
    return best_points

# EVOLVE-BLOCK-END
