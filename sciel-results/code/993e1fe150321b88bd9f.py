"""Improved packing: multi-start local optimization with repulsion energy."""
import numpy as np
import time
from scipy.spatial.distance import cdist

N = 50
START_TIME = time.time()
TIME_LIMIT = 115.0  # safety margin before 120s kill


def _score(points):
    """Compute the score: largest common radius."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # Check all points inside unit circle
    norms = np.hypot(points[:, 0], points[:, 1])
    if np.any(norms > 1.0):
        return 0.0
    # Min distance from any point to boundary
    boundary_min = 1.0 - np.max(norms)
    # Min pairwise distance / 2
    if N > 1:
        d = cdist(points, points)
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d) / 2.0
    else:
        min_pair = boundary_min
    return min(min_pair, boundary_min)


def _energy(points, r_target):
    """Repulsion energy: penalize overlaps and boundary violations."""
    # Pairwise repulsion
    d = cdist(points, points)
    np.fill_diagonal(d, np.inf)
    # Penalize pairs closer than 2*r_target
    overlap_penalty = np.sum(np.maximum(0, 2 * r_target - d) ** 2)
    # Boundary penalty
    norms = np.hypot(points[:, 0], points[:, 1])
    boundary_penalty = np.sum(np.maximum(0, norms - (1.0 - r_target)) ** 2)
    return overlap_penalty + 10.0 * boundary_penalty


def _gradient(points, r_target):
    """Compute gradient of energy w.r.t. point positions."""
    n = len(points)
    grad = np.zeros_like(points)
    # Pairwise forces
    d = cdist(points, points)
    np.fill_diagonal(d, np.inf)
    # For overlapping pairs, push apart
    for i in range(n):
        for j in range(i + 1, n):
            diff = points[i] - points[j]
            dist = d[i, j]
            if dist < 2 * r_target and dist > 1e-12:
                force = 2 * (2 * r_target - dist) * diff / dist
                grad[i] += force
                grad[j] -= force
    # Boundary force
    norms = np.hypot(points[:, 0], points[:, 1])
    for i in range(n):
        if norms[i] > 1.0 - r_target and norms[i] > 1e-12:
            force = 20.0 * (norms[i] - (1.0 - r_target)) * points[i] / norms[i]
            grad[i] -= force
    return grad


def _local_optimize(points, r_target, max_iters=500, lr=0.005):
    """Local optimization using gradient descent with line search."""
    points = points.copy()
    best_score = _score(points)
    current_energy = _energy(points, r_target)
    
    for it in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        grad = _gradient(points, r_target)
        # Gradient descent step
        new_points = points - lr * grad
        
        # Line search: accept if energy decreases
        new_energy = _energy(new_points, r_target)
        if new_energy < current_energy:
            points = new_points
            current_energy = new_energy
            # Adaptive learning rate
            if it % 50 == 0 and it > 0:
                lr *= 0.9
        else:
            lr *= 0.8
            if lr < 1e-6:
                break
        
        # Update best score
        s = _score(points)
        if s > best_score:
            best_score = s
    
    return points, best_score


def _hexagonal_initial(spacing):
    """Generate hexagonal lattice points inside unit disk."""
    points = []
    k = int(1.0 / spacing) + 2
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x * x + y * y <= (1.0 - spacing / 2.0) ** 2:
                points.append([x, y])
    points = np.array(points)
    if len(points) < N:
        return None
    # Select N closest to origin
    norms = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(norms, kind='stable')[:N]
    return points[idx]


def _random_initial():
    """Generate random initial configuration."""
    points = []
    for _ in range(N):
        while True:
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            if x * x + y * y <= 0.81:  # radius 0.9
                points.append([x, y])
                break
    return np.array(points)


def _ring_initial():
    """Generate ring-based initial configuration."""
    points = []
    # Center
    points.append([0.0, 0.0])
    # Rings
    radii = [0.3, 0.55, 0.75, 0.9]
    counts = [6, 12, 18, 13]  # total = 1+6+12+18+13 = 50
    for r, n in zip(radii, counts):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        for a in angles:
            points.append([r * np.cos(a), r * np.sin(a)])
    return np.array(points[:N])


def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_points = None
    best_score = 0.0
    
    # Try multiple initial configurations
    initializations = []
    
    # Hexagonal with various spacings
    for spacing in [0.22, 0.24, 0.26, 0.28]:
        pts = _hexagonal_initial(spacing)
        if pts is not None:
            initializations.append(pts)
    
    # Ring-based
    initializations.append(_ring_initial())
    
    # Random
    np.random.seed(42)
    for _ in range(3):
        initializations.append(_random_initial())
    
    for init_points in initializations:
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        # Start with a target radius slightly below the known best
        # Known best: 0.1258254895304
        # Start from lower and increase
        r_target = 0.10
        
        # Iteratively increase target radius
        for _ in range(8):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            
            optimized, score = _local_optimize(init_points, r_target, max_iters=300, lr=0.003)
            if score > best_score:
                best_score = score
                best_points = optimized.copy()
            
            # If we achieved the target, try increasing
            if score >= r_target * 0.95:
                r_target += 0.005
            else:
                # Failed to achieve, adjust
                r_target = max(0.08, score * 1.05)
        
        # Final polish with the achieved radius
        if best_points is not None:
            final_r = best_score
            optimized, score = _local_optimize(best_points, final_r, max_iters=400, lr=0.001)
            if score > best_score:
                best_score = score
                best_points = optimized.copy()
    
    # Final verification and cleanup
    if best_points is None:
        # Fallback: simple hexagonal
        pts = _hexagonal_initial(0.25)
        if pts is not None:
            best_points = pts
        else:
            best_points = np.zeros((N, 2))
    
    # Ensure validity: clip to unit disk
    norms = np.hypot(best_points[:, 0], best_points[:, 1])
    mask = norms > 1.0
    if np.any(mask):
        best_points[mask] = best_points[mask] / norms[mask, np.newaxis] * 0.999
    
    return best_points
