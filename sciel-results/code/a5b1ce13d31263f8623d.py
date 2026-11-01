"""Rewritten program: random init + Nelder-Mead optimization for 50-circle packing."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 120.0  # seconds, with safety margin before 150s kill


def _score(points):
    """Compute the packing radius: min of half-pair-distances and boundary distances."""
    # Pairwise distances
    n = len(points)
    # Compute all pairwise distances efficiently
    dx = points[:, 0:1] - points[0:1, 0]  # not efficient, use broadcasting
    # Better: use scipy or manual
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    
    # Boundary distances
    radii = np.sqrt(np.sum(points ** 2, axis=1))
    min_boundary = 1.0 - np.max(radii)
    
    return min(min_pair, min_boundary)


def _objective(points_flat):
    """Objective to minimize: negative of the packing radius."""
    points = points_flat.reshape(N, 2)
    # If any point is outside the unit circle, penalize heavily
    radii = np.sqrt(np.sum(points ** 2, axis=1))
    if np.any(radii > 1.0):
        return 100.0 + np.max(radii) - 1.0
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    
    min_boundary = 1.0 - np.max(radii)
    
    return -min(min_pair, min_boundary)


def _random_init():
    """Generate N random points inside the unit disk."""
    while True:
        points = np.random.randn(N, 2)
        radii = np.sqrt(np.sum(points ** 2, axis=1))
        if np.all(radii <= 1.0):
            # Scale to be well inside
            scale = 0.8 / np.max(radii) if np.max(radii) > 0 else 1.0
            return points * scale


def construct():
    start_time = time.time()
    
    best_points = None
    best_score = -np.inf
    
    # Try multiple random restarts with Nelder-Mead
    n_restarts = 10
    for restart in range(n_restarts):
        # Check time
        if time.time() - start_time > DEADLINE - 30:
            break
        
        # Random initialization
        x0 = _random_init()
        
        # Run Nelder-Mead
        try:
            result = minimize(
                _objective,
                x0.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
        except Exception:
            continue
        
        # Evaluate result
        points = result.x.reshape(N, 2)
        score = _score(points)
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
        
        # Check if we've hit the deadline
        if time.time() - start_time > DEADLINE - 20:
            break
    
    # If no valid result, fall back to a simple grid
    if best_points is None or best_score <= 0:
        # Simple hexagonal lattice fallback
        spacing = 0.25
        points = []
        for j in range(-5, 6):
            for i in range(-5, 6):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.sqrt(x ** 2 + y ** 2) <= 1.0 - spacing / 2:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= N:
            # Select N closest to center
            radii = np.sqrt(np.sum(points ** 2, axis=1))
            indices = np.argsort(radii)[:N]
            best_points = points[indices]
        else:
            # Just take N random points
            best_points = _random_init()
    
    # Final polish: run a few more Nelder-Mead iterations from the best point
    if time.time() - start_time < DEADLINE - 10:
        try:
            result = minimize(
                _objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True
                }
            )
            points = result.x.reshape(N, 2)
            score = _score(points)
            if score > best_score:
                best_points = points
        except Exception:
            pass
    
    return best_points
