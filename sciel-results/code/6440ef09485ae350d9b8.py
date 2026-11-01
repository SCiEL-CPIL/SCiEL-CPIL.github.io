import numpy as np
import time
from scipy.optimize import minimize

N = 14
D = 3

def get_pairwise_sq_dists(points):
    """
    Compute squared pairwise distances for N points in D dimensions.
    Returns a flat array of length N*(N-1)/2.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    """
    Calculate the actual score: (max_dist / min_dist)^2.
    """
    sq_dists = get_pairwise_sq_dists(points)
    min_d = np.min(sq_dists)
    if min_d < 1e-12:
        return np.inf
    return (np.max(sq_dists) / min_d)

def normalize_points(points):
    """
    Center and normalize points to unit mean squared norm.
    This removes translational and scale invariance.
    """
    points = points - np.mean(points, axis=0)
    mean_norm_sq = np.mean(np.sum(points**2, axis=1))
    if mean_norm_sq > 1e-12:
        points = points / np.sqrt(mean_norm_sq)
    return points

def objective_direct(x):
    """
    Direct objective: minimize (max_sq_dist / min_sq_dist) with a penalty for very small min.
    """
    points = x.reshape(N, D)
    points = normalize_points(points)
    
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-10)
    
    min_d = np.min(sq_dists)
    max_d = np.max(sq_dists)
    
    # The ratio we want to minimize
    ratio = max_d / min_d
    
    # Add a small penalty if min is too small (to avoid degenerate solutions)
    penalty = 0.0
    if min_d < 0.01:
        penalty = (0.01 - min_d) ** 2 * 100
    
    return ratio + penalty

def construct():
    start_time = time.time()
    timeout = 145.0 # Safety margin before 150s kill
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts = np.asarray(pts)
        if pts.shape != (N, D):
            return
        pts = normalize_points(pts)
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()

    rng = np.random.default_rng(42)
    
    # Generate initial candidates
    candidates = []
    
    # 1. Random points on sphere (more diverse)
    for _ in range(30):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        candidates.append(pts.flatten())
        
    # 2. Icosahedron (12 points) + 2 random points
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    for _ in range(10):
        p1 = rng.normal(size=D)
        p2 = rng.normal(size=D)
        p1 /= np.linalg.norm(p1)
        p2 /= np.linalg.norm(p2)
        pts = np.vstack([ico, p1.reshape(1, D), p2.reshape(1, D)])
        candidates.append(pts.flatten())

    # 3. Cuboctahedron vertices (12 points) + 2 random
    perms = np.array([
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0]
    ])
    cubo = perms / np.linalg.norm(perms, axis=1, keepdims=True)
    for _ in range(10):
        p1 = rng.normal(size=D)
        p2 = rng.normal(size=D)
        p1 /= np.linalg.norm(p1)
        p2 /= np.linalg.norm(p2)
        pts = np.vstack([cubo, p1.reshape(1, D), p2.reshape(1, D)])
        candidates.append(pts.flatten())
    
    # 4. Fibonacci sphere distribution (better initial spread)
    def fibonacci_sphere(n_points):
        """Generate points on a sphere using Fibonacci lattice."""
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n_points):
            z = 1 - 2 * i / (n_points - 1)
            radius = np.sqrt(1 - z**2)
            theta = golden_angle * i
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            points.append([x, y, z])
        return np.array(points)
    
    fib_pts = fibonacci_sphere(N)
    for _ in range(5):
        # Add small random perturbations
        perturbed = fib_pts + rng.normal(scale=0.1, size=(N, D))
        perturbed = perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True)
        candidates.append(perturbed.flatten())
    
    # Optimization Loop
    n_restarts = len(candidates)
    
    for i in range(n_restarts):
        if time.time() - start_time > timeout - 15:
            break
            
        x0 = candidates[i].copy()
        x0 = x0.reshape(N, D)
        x0 = normalize_points(x0)
        x0 = x0.flatten()
        
        try:
            # Use Nelder-Mead for non-smooth objective
            res = minimize(
                objective_direct, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if res.x is not None:
                pts_current = res.x.reshape(N, D)
                update_best(pts_current)
                
        except Exception:
            pass
    
    # Final polish with L-BFGS-B on the direct objective (smoother)
    if best_points is not None and time.time() - start_time < timeout - 5:
        try:
            x_polish = best_points.flatten()
            res = minimize(
                objective_direct, 
                x_polish, 
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.x is not None:
                pts_polish = res.x.reshape(N, D)
                update_best(pts_polish)
        except Exception:
            pass

    if best_points is None:
        best_points = normalize_points(candidates[0].reshape(N, D))
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
