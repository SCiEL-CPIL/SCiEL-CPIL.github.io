import numpy as np
from scipy.optimize import minimize
import time

N = 14
D = 3

def _pairwise_distances(pts):
    """Calculate pairwise distances between all pairs of points."""
    # pts is (N, D)
    # Use broadcasting to compute distances
    # dist[i, j] = ||pts[i] - pts[j]||
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    return dist

def _soft_max_min_ratio(dist, tau):
    """
    Calculate a smooth approximation of the log ratio of max to min distance.
    dist: (N, N) matrix of distances, diagonal is 0.
    We ignore the diagonal.
    """
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    d = dist[mask]
    
    # Soft max and min
    # Using log-sum-exp trick for stability
    # soft_max = tau * log(sum(exp(d/tau)))
    # soft_min = -tau * log(sum(exp(-d/tau)))
    
    # To avoid overflow in exp, we can shift, but d is usually small.
    # However, if d is large, exp(d/tau) can overflow.
    # Let's use a stable version.
    max_d = np.max(d)
    min_d = np.min(d)
    
    # If tau is small, we approach hard max/min.
    # If tau is large, we approach mean.
    
    # Stable soft max
    # log(sum(exp(d/tau))) = max_d/tau + log(sum(exp((d-max_d)/tau)))
    soft_max = tau * (max_d/tau + np.log(np.sum(np.exp((d - max_d)/tau))))
    
    # Stable soft min
    # log(sum(exp(-d/tau))) = -min_d/tau + log(sum(exp((min_d-d)/tau)))
    soft_min = -tau * (-min_d/tau + np.log(np.sum(np.exp((min_d - d)/tau))))
    
    # The objective is to minimize the ratio, which is equivalent to minimizing log(max) - log(min)
    # Or simply (max/min)^2.
    # Minimizing log(max) - log(min) is smoother.
    return soft_max - soft_min

def _objective(x, tau):
    """
    Objective function for scipy.optimize.minimize.
    x: flat array of size N*D
    """
    pts = x.reshape(N, D)
    
    # Center the points
    pts = pts - np.mean(pts, axis=0)
    
    # Normalize scale to prevent the optimizer from just scaling points to 0 or infinity
    # Fix the sum of squared norms to N
    norm_sq = np.sum(pts**2)
    if norm_sq < 1e-10:
        return 1e10
    pts = pts * np.sqrt(N / norm_sq)
    
    dist = _pairwise_distances(pts)
    
    # Check for coincident points
    min_d = np.min(dist[~np.eye(N, dtype=bool)])
    if min_d < 1e-4:
        return 1e10
    
    return _soft_max_min_ratio(dist, tau)

def _true_score(pts):
    """Calculate the actual score: (max_dist / min_dist)^2"""
    dist = _pairwise_distances(pts)
    mask = ~np.eye(N, dtype=bool)
    d = dist[mask]
    mn = np.min(d)
    mx = np.max(d)
    if mn < 1e-9:
        return np.inf
    return (mx / mn) ** 2

def _generate_icosahedron():
    """Generate vertices of a regular icosahedron."""
    phi = (1 + np.sqrt(5)) / 2
    # 12 vertices
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    return verts

def _generate_initial_guesses():
    """Generate a list of initial configurations."""
    guesses = []
    
    # 1. Icosahedron + Center + 1 point on Z-axis (symmetric extension)
    ico = _generate_icosahedron()
    # Normalize ico to have unit norm
    ico_norm = np.linalg.norm(ico, axis=1, keepdims=True)
    ico = ico / ico_norm
    
    # Add center (0,0,0)
    # Add 14th point. Maybe at (0, 0, 2) or similar?
    # Let's try placing the 14th point at (0,0,0) is taken.
    # How about 12 ico vertices + 2 points on Z axis?
    # Or 12 ico vertices + center + 1 point slightly perturbed?
    
    # Guess A: Icosahedron + Center + 1 point at (0,0,0.5)
    p14_a = np.vstack([ico, np.array([[0,0,0]]), np.array([[0,0,0.5]])])
    guesses.append(p14_a)
    
    # Guess B: Icosahedron + 2 points on Z axis (symmetric)
    # Replace center with two points at (0,0, h) and (0,0, -h)
    # This maintains inversion symmetry.
    p14_b = np.vstack([ico, np.array([[0,0,0.5]]), np.array([[0,0,-0.5]])])
    guesses.append(p14_b)
    
    # Guess C: Fibonacci Sphere
    n = N
    i = np.arange(n)
    phi = np.arccos(1 - 2*(i+0.5)/n)
    theta = np.pi * (1 + np.sqrt(5)) * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    fib_pts = np.column_stack((x, y, z))
    guesses.append(fib_pts)
    
    # Guess D: Random points in a sphere
    for _ in range(5):
        rand_pts = np.random.randn(N, D)
        # Center
        rand_pts = rand_pts - np.mean(rand_pts, axis=0)
        guesses.append(rand_pts)
        
    return guesses

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Keep 10s margin
    
    best_score = np.inf
    best_pts = None
    
    # Start with a reasonable tau, maybe anneal it?
    # For now, fixed tau. 0.1 seems reasonable for distances of order 1.
    tau = 0.1
    
    guesses = _generate_initial_guesses()
    
    for guess in guesses:
        if time.time() > deadline:
            break
            
        # Flatten for optimizer
        x0 = guess.flatten()
        
        try:
            res = minimize(
                _objective, 
                x0, 
                args=(tau,), 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
        except Exception:
            continue
            
        if res.success or res.fun < best_score:
            pts = res.x.reshape(N, D)
            # Center and normalize for final score calculation consistency
            pts = pts - np.mean(pts, axis=0)
            norm_sq = np.sum(pts**2)
            if norm_sq > 1e-10:
                pts = pts * np.sqrt(N / norm_sq)
                
            score = _true_score(pts)
            if score < best_score:
                best_score = score
                best_pts = pts
    
    # If no valid points found, fallback to grid (should not happen)
    if best_pts is None:
        side = int(np.ceil(N ** (1 / D)))
        import itertools
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_pts = np.array(grid, dtype=float)
        
    return best_pts
