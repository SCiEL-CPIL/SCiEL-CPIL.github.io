"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 300
_DEADLINE = 150.0  # seconds
_START_TIME = time.time()


def _remaining_time():
    return _DEADLINE - (time.time() - _START_TIME) - 2.0  # 2s safety margin


def _hex_lattice_init(n):
    """Generate initial hexagonal lattice points inside unit disk."""
    # Estimate spacing for n circles
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    
    # Generate hexagonal lattice points
    points = []
    for k in range(-20, 21):
        for j in range(-20, 21):
            x = (k + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r < 0.95:  # Leave margin
                points.append([x, y])
    
    points = np.array(points)
    
    # If we have more than n, select the n closest to origin
    if len(points) >= n:
        dists = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(dists, kind='stable')[:n]
        points = points[idx]
    else:
        # Pad with random points if needed
        while len(points) < n:
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, 0.9)
            points = np.vstack([points, [r * np.cos(angle), r * np.sin(angle)]])
    
    return points


def _compute_min_dist(points):
    """Compute the minimum pairwise distance and minimum distance to boundary."""
    n = len(points)
    if n < 2:
        return 0.0, 0.0
    
    # Compute all pairwise distances
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    
    # Get minimum pairwise distance (excluding diagonal)
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists)
    
    # Minimum distance to boundary
    radii = np.hypot(points[:, 0], points[:, 1])
    min_boundary = 1.0 - np.max(radii)
    
    return min_pairwise, min_boundary


def _score(points):
    """Compute the packing score (minimum of half min-pairwise and min-boundary)."""
    min_pairwise, min_boundary = _compute_min_dist(points)
    return min(min_pairwise / 2.0, min_boundary)


def _gradient_ascent_step(points, step_size, max_iter=100):
    """Perform gradient ascent on the minimum distance objective."""
    n = len(points)
    current_points = points.copy()
    current_score = _score(current_points)
    
    for _ in range(max_iter):
        if _remaining_time() < 5.0:
            break
        
        # Compute distances
        diffs = current_points[:, np.newaxis, :] - current_points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Find minimum distance pair
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx
        min_dist = dists[i, j]
        
        # Compute gradient for the minimum distance
        grad = np.zeros_like(current_points)
        if min_dist > 1e-10:
            diff = current_points[i] - current_points[j]
            unit_diff = diff / min_dist
            grad[i] += unit_diff
            grad[j] -= unit_diff
        
        # Also consider boundary constraints
        radii = np.hypot(current_points[:, 0], current_points[:, 1])
        max_radius = np.max(radii)
        if max_radius > 0.9:
            max_idx = np.argmax(radii)
            if radii[max_idx] > 1e-10:
                grad[max_idx] -= current_points[max_idx] / radii[max_idx]
        
        # Update points
        new_points = current_points + step_size * grad
        
        # Check if score improved
        new_score = _score(new_points)
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
        else:
            break
    
    return current_points, current_score


def construct():
    # Generate initial hexagonal lattice
    points = _hex_lattice_init(N)
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Adaptive step size gradient ascent
    step_size = 0.01
    total_iter = 0
    max_total_iter = 5000
    
    while _remaining_time() > 10.0 and total_iter < max_total_iter:
        # Perform gradient ascent
        points, score = _gradient_ascent_step(best_points, step_size, max_iter=50)
        total_iter += 50
        
        if score > best_score:
            best_points = points
            best_score = score
            step_size = max(0.001, step_size * 0.9)  # Reduce step size
        else:
            # Try smaller step size
            step_size *= 0.5
            if step_size < 0.0001:
                break
    
    # Final polish with very small steps
    if _remaining_time() > 5.0:
        points, score = _gradient_ascent_step(best_points, 0.001, max_iter=100)
        if score > best_score:
            best_points = points
            best_score = score
    
    # Ensure all points are strictly inside the unit circle
    radii = np.hypot(best_points[:, 0], best_points[:, 1])
    mask = radii < 1.0
    if not np.all(mask):
        best_points = best_points[mask]
        # If we lost points, we need to add them back
        while len(best_points) < N:
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, 0.8)
            best_points = np.vstack([best_points, [r * np.cos(angle), r * np.sin(angle)]])
    
    return best_points
# EVOLVE-BLOCK-END
