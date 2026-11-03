"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time


def _score(points):
    """Compute the packing score for given points."""
    n = len(points)
    # Distance between all pairs
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    
    # Distance from origin
    radii = np.sqrt(np.sum(points ** 2, axis=-1))
    min_boundary = 1.0 - np.max(radii)
    
    if min_boundary < 0:
        return -1.0  # Invalid: circle outside container
    
    r = min(min_pair_dist / 2.0, min_boundary)
    return r


def _objective(params):
    """Objective function to minimize (negative score)."""
    points = params.reshape(-1, 2)
    score = _score(points)
    return -score


def _hexagonal_initial(n_circles, seed=0):
    """Generate a hexagonal lattice initial guess."""
    rng = np.random.RandomState(seed)
    
    # Try different spacings
    best_points = None
    best_score = -np.inf
    
    for spacing in np.linspace(0.15, 0.35, 20):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        
        if len(points) < n_circles:
            continue
        
        # Pick the n_circles points closest to center
        points = points[np.argsort(radius, kind="stable")[:n_circles]]
        score = _score(points)
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    if best_points is None:
        # Fallback: random points
        angles = rng.uniform(0, 2 * np.pi, n_circles)
        radii = rng.uniform(0, 0.8, n_circles)
        best_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    return best_points


def _random_initial(n_circles, seed=0):
    """Generate random initial points inside the unit circle."""
    rng = np.random.RandomState(seed)
    angles = rng.uniform(0, 2 * np.pi, n_circles)
    radii = np.sqrt(rng.uniform(0, 0.81, n_circles))  # sqrt for uniform distribution
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 10s before 150s limit
    
    # Start with hexagonal lattice
    points = _hexagonal_initial(N, seed=42)
    best_points = points.copy()
    best_score = _score(points)
    
    # Differential Evolution for global search
    bounds = [(-0.95, 0.95)] * (2 * N)
    
    # Use DE with reasonable parameters
    de_result = differential_evolution(
        _objective,
        bounds,
        maxiter=300,
        popsize=15,
        tol=1e-10,
        seed=42,
        polish=False,  # We'll polish separately
        init='sobol'
    )
    
    if de_result.fun < -best_score:
        de_points = de_result.x.reshape(-1, 2)
        de_score = _score(de_points)
        if de_score > best_score:
            best_score = de_score
            best_points = de_points.copy()
    
    # L-BFGS polish from multiple starting points
    start_candidates = [best_points.copy()]
    
    # Add some perturbed versions
    for seed in range(5):
        if time.time() > deadline - 20:
            break
        perturbed = best_points + 0.01 * np.random.RandomState(seed).randn(2 * N).reshape(-1, 2)
        # Ensure points stay within bounds
        radii = np.sqrt(np.sum(perturbed ** 2, axis=-1))
        mask = radii > 0.95
        if np.any(mask):
            perturbed[mask] = perturbed[mask] * 0.95 / radii[mask]
        start_candidates.append(perturbed)
    
    # Also try random starts
    for seed in range(3):
        if time.time() > deadline - 15:
            break
        start_candidates.append(_random_initial(N, seed=100 + seed))
    
    for i, start in enumerate(start_candidates):
        if time.time() > deadline - 10:
            break
        
        res = minimize(
            _objective,
            start.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15}
        )
        
        if res.fun < -best_score:
            res_points = res.x.reshape(-1, 2)
            res_score = _score(res_points)
            if res_score > best_score:
                best_score = res_score
                best_points = res_points.copy()
    
    # Final validation
    final_score = _score(best_points)
    if final_score < 0:
        # Fallback to hexagonal
        best_points = _hexagonal_initial(N, seed=0)
    
    return best_points
# EVOLVE-BLOCK-END
