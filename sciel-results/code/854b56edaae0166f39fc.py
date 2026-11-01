import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 14
D = 3

def _pairwise_sq_dists(pts):
    """Calculate squared pairwise distances between all pairs of points."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sum(diff**2, axis=-1)

def _true_score(pts):
    """Calculate the actual score: (max_dist / min_dist)^2"""
    sq_d = _pairwise_sq_dists(pts)
    mask = ~np.eye(N, dtype=bool)
    d_sq = sq_d[mask]
    mn_sq = np.min(d_sq)
    mx_sq = np.max(d_sq)
    if mn_sq < 1e-20:
        return np.inf
    return mx_sq / mn_sq

def _normalize_points(pts):
    """Center and normalize points to have sum of squared norms = N."""
    pts = pts - np.mean(pts, axis=0)
    norm_sq = np.sum(pts**2)
    if norm_sq < 1e-10:
        return pts
    return pts * np.sqrt(N / norm_sq)

def _objective_smooth(x, tau=0.05):
    """
    Smooth objective: soft approximation of log(max/min) ratio.
    x: flat array of size N*D
    """
    pts = x.reshape(N, D)
    pts = _normalize_points(pts)
    
    sq_d = _pairwise_sq_dists(pts)
    mask = ~np.eye(N, dtype=bool)
    d = np.sqrt(sq_d[mask])
    
    if np.min(d) < 1e-6:
        return 1e8
    
    # Stable soft max and min
    max_d = np.max(d)
    min_d = np.min(d)
    
    soft_max = tau * (max_d/tau + np.log(np.sum(np.exp((d - max_d)/tau))))
    soft_min = -tau * (-min_d/tau + np.log(np.sum(np.exp((min_d - d)/tau))))
    
    return soft_max - soft_min

def _objective_direct(x):
    """
    Direct objective: (max_dist / min_dist)^2 with penalty for close points.
    x: flat array of size N*D
    """
    pts = x.reshape(N, D)
    pts = _normalize_points(pts)
    
    sq_d = _pairwise_sq_dists(pts)
    mask = ~np.eye(N, dtype=bool)
    d_sq = sq_d[mask]
    
    mn_sq = np.min(d_sq)
    mx_sq = np.max(d_sq)
    
    if mn_sq < 1e-12:
        return 1e10
    
    ratio_sq = mx_sq / mn_sq
    
    # Add penalty if points are too close to encourage spread
    if mn_sq < 0.01:
        ratio_sq += 100.0 * (0.01 - mn_sq)
    
    return ratio_sq

def _generate_icosahedron():
    """Generate vertices of a regular icosahedron."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    return verts

def _generate_initial_guesses():
    """Generate a list of initial configurations."""
    guesses = []
    
    ico = _generate_icosahedron()
    ico_norm = np.linalg.norm(ico, axis=1, keepdims=True)
    ico = ico / ico_norm
    
    # Guess A: Icosahedron + Center + 1 point at (0,0,0.5)
    p14_a = np.vstack([ico, np.array([[0,0,0]]), np.array([[0,0,0.5]])])
    guesses.append(p14_a)
    
    # Guess B: Icosahedron + 2 points on Z axis (symmetric)
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
    
    # Guess D: Random points
    for _ in range(3):
        rand_pts = np.random.randn(N, D)
        rand_pts = rand_pts - np.mean(rand_pts, axis=0)
        guesses.append(rand_pts)
        
    return guesses

def _local_optimize(x0, maxiter=300):
    """Run local optimization from a starting point."""
    try:
        res = minimize(
            _objective_direct,
            x0,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if res.success or res.fun < 1e8:
            pts = res.x.reshape(N, D)
            pts = _normalize_points(pts)
            score = _true_score(pts)
            if score < np.inf:
                return pts, score
    except Exception:
        pass
    return None, np.inf

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Keep 15s margin for safety
    
    best_score = np.inf
    best_pts = None
    
    def update_best(pts, score):
        nonlocal best_score, best_pts
        if score < best_score:
            best_score = score
            best_pts = pts
            return True
        return False
    
    # Phase 1: Local optimization from multiple starting points
    guesses = _generate_initial_guesses()
    
    for guess in guesses:
        if time.time() > deadline - 60:
            break
        x0 = guess.flatten()
        pts, score = _local_optimize(x0, maxiter=300)
        if pts is not None:
            update_best(pts, score)
    
    # Phase 2: Differential evolution for global search
    if time.time() < deadline - 30:
        try:
            # Bounds for each coordinate (after normalization, points are roughly in [-2, 2])
            bounds = [(-3, 3)] * (N * D)
            
            de_res = differential_evolution(
                _objective_direct,
                bounds,
                maxiter=50,
                popsize=15,
                tol=1e-8,
                mutation=(0.3, 1.0),
                recombination=0.8,
                seed=42,
                polish=False,
                update_interval=5,
                callback=lambda x, convergence: time.time() > deadline - 30
            )
            
            if de_res.fun < 1e8:
                pts = de_res.x.reshape(N, D)
                pts = _normalize_points(pts)
                score = _true_score(pts)
                if score < np.inf:
                    update_best(pts, score)
                    
                    # Polish the DE result with local optimization
                    if time.time() < deadline - 10:
                        pts2, score2 = _local_optimize(de_res.x, maxiter=500)
                        if pts2 is not None:
                            update_best(pts2, score2)
        except Exception:
            pass
    
    # Phase 3: More local optimizations from perturbed best solution
    if best_pts is not None and time.time() < deadline - 5:
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            # Perturb best solution
            noise = np.random.randn(N, D) * 0.1
            perturbed = best_pts + noise
            x0 = perturbed.flatten()
            pts, score = _local_optimize(x0, maxiter=200)
            if pts is not None:
                update_best(pts, score)
    
    # Fallback
    if best_pts is None:
        side = int(np.ceil(N ** (1 / D)))
        import itertools
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_pts = np.array(grid, dtype=float)
        
    return best_pts
