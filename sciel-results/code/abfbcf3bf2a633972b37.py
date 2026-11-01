"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import itertools

N = 14
D = 3

# EVOLVE-BLOCK-START
def _pairwise_distances(points):
    """Compute all pairwise squared distances."""
    # points: (n, d)
    # squared distances: (n, n)
    sq = np.sum(points ** 2, axis=1)
    sqd = sq[:, None] + sq[None, :] - 2 * points @ points.T
    np.maximum(sqd, 0, out=sqd)
    return sqd

def _score(points):
    """Compute the squared ratio of max to min pairwise distance."""
    sqd = _pairwise_distances(points)
    # Get upper triangle indices
    iu = np.triu_indices(N, k=1)
    d_sq = sqd[iu]
    d_min = np.min(d_sq)
    d_max = np.max(d_sq)
    if d_min <= 0:
        return np.inf
    return (d_max / d_min) ** 2

def _objective(x):
    """Objective function for optimization: squared ratio of max to min distance."""
    points = x.reshape(N, D)
    return _score(points)

def _objective_log(x):
    """Alternative objective using log for numerical stability."""
    points = x.reshape(N, D)
    sqd = _pairwise_distances(points)
    iu = np.triu_indices(N, k=1)
    d_sq = sqd[iu]
    d_min = np.min(d_sq)
    d_max = np.max(d_sq)
    if d_min <= 1e-12:
        return 1e10
    return np.log(d_max / d_min)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_points = None
    best_score = np.inf
    
    # Strategy 1: Use known good initial configurations
    # Start with a few heuristic initial points
    
    # Initial guess: points on a sphere or grid
    # Use a Fibonacci sphere for initial distribution
    def fibonacci_sphere(n, radius=1.0):
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n):
            y = 1 - (i / (n - 1)) * 2
            r = np.sqrt(1 - y ** 2)
            theta = golden_angle * i
            x = r * np.cos(theta)
            z = r * np.sin(theta)
            points.append([x * radius, y * radius, z * radius])
        return np.array(points)
    
    # Generate several initial configurations
    initial_configs = []
    
    # 1. Fibonacci sphere
    initial_configs.append(fibonacci_sphere(N, radius=1.0))
    
    # 2. Random points in a cube
    np.random.seed(42)
    for seed in range(5):
        np.random.seed(seed)
        pts = np.random.randn(N, D)
        # Normalize to unit sphere-ish
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        initial_configs.append(pts)
    
    # 3. Grid-based
    side = int(np.ceil(N ** (1 / D)))
    grid = list(itertools.product(range(side), repeat=D))[:N]
    grid_pts = np.array(grid, dtype=float)
    grid_pts = grid_pts - grid_pts.mean(axis=0)
    initial_configs.append(grid_pts)
    
    # Evaluate initial configurations
    for pts in initial_configs:
        if time.time() > deadline:
            break
        s = _score(pts)
        if s < best_score:
            best_score = s
            best_points = pts.copy()
    
    # Strategy 2: Local optimization using L-BFGS-B
    for pts in initial_configs:
        if time.time() > deadline:
            break
        try:
            result = minimize(
                _objective,
                pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            if time.time() > deadline:
                break
            if result.fun < best_score:
                best_score = result.fun
                best_points = result.x.reshape(N, D)
        except Exception:
            pass
    
    # Strategy 3: Differential evolution for global search
    if time.time() < deadline - 30:
        # Use differential evolution with a reasonable number of iterations
        bounds = [(-5, 5)] * (N * D)
        try:
            de_result = differential_evolution(
                _objective,
                bounds,
                maxiter=50,
                popsize=10,
                tol=1e-8,
                seed=42,
                polish=False
            )
            if time.time() > deadline:
                break
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_points = de_result.x.reshape(N, D)
        except Exception:
            pass
    
    # Strategy 4: More local refinement from the best point found
    if best_points is not None and time.time() < deadline - 10:
        try:
            result = minimize(
                _objective,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_points = result.x.reshape(N, D)
        except Exception:
            pass
    
    # Final refinement with COBYLA
    if best_points is not None and time.time() < deadline - 5:
        try:
            result = minimize(
                _objective,
                best_points.flatten(),
                method='COBYLA',
                options={'maxiter': 200, 'rhobeg': 0.1}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_points = result.x.reshape(N, D)
        except Exception:
            pass
    
    if best_points is None:
        # Fallback to grid
        side = int(np.ceil(N ** (1 / D)))
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_points = np.array(grid, dtype=float)
    
    return best_points
# EVOLVE-BLOCK-END
