"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial.distance import cdist

N = 300


def _compute_score(points):
    """Compute the actual packing score (radius) for given points."""
    n = len(points)
    if n < 2:
        return 0.0
    
    # Pairwise distances
    d = cdist(points, points)
    np.fill_diagonal(d, np.inf)
    min_dist = np.min(d)
    
    # Boundary constraint
    norms = np.hypot(points[:, 0], points[:, 1])
    max_norm = np.max(norms)
    boundary_r = 1.0 - max_norm
    
    return min(min_dist / 2.0, boundary_r)


def _hex_lattice_init(n_points):
    """Generate a hexagonal lattice packing inside the unit disk."""
    best_points = None
    best_score = -1
    
    # Search over different lattice spacings
    for scale in np.linspace(0.09, 0.14, 60):
        dx = scale
        dy = scale * np.sqrt(3) / 2
        
        points = []
        max_r = 1.0
        n_rows = int(2 * max_r / dy) + 2
        n_cols = int(2 * max_r / dx) + 2
        
        for i in range(-n_cols, n_cols + 1):
            for j in range(-n_rows, n_rows + 1):
                x = i * dx + (0.5 * dx if j % 2 != 0 else 0)
                y = j * dy
                if x * x + y * y <= 1.0:
                    points.append([x, y])
        
        points = np.array(points)
        
        if len(points) < n_points:
            continue
        
        # Sort by distance from center, take closest n_points
        radii = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(radii, kind='stable')[:n_points]
        pts = points[idx]
        
        est_r = _compute_score(pts)
        
        if est_r > best_score:
            best_score = est_r
            best_points = pts
    
    if best_points is None:
        best_points = np.random.randn(n_points, 2) * 0.5
        best_score = 0
    
    return best_points, best_score


def _vectorized_gradient(pts, target_r):
    """
    Compute gradient of a smooth objective that maximizes minimum distance
    while keeping circles inside the unit disk.
    
    Uses inverse-square repulsion for close pairs and radial confinement.
    """
    n = len(pts)
    # Pairwise differences: diff[i,j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 2)
    dist = np.linalg.norm(diff, axis=2)  # (n, n)
    np.fill_diagonal(dist, np.inf)
    
    # Repulsion: for pairs closer than 2*target_r, push apart
    # Force magnitude ~ 1/dist^2 for close pairs, decays for far pairs
    # Use a smooth cutoff: force = max(0, (2*target_r - dist))^2 / (dist^2 + eps)
    eps = 1e-12
    overlap = np.maximum(0, 2.0 * target_r - dist)
    force_mag = overlap ** 2 / (dist ** 2 + eps)
    np.fill_diagonal(force_mag, 0)
    
    # Force vector: F_i = sum_j force_mag[i,j] * (pts[i] - pts[j]) / dist[i,j]
    # = sum_j force_mag[i,j] * diff[i,j] / dist[i,j]
    unit_diff = diff / (dist[:, :, np.newaxis] + eps)  # (n, n, 2)
    repulsion = np.sum(force_mag[:, :, np.newaxis] * unit_diff, axis=1)  # (n, 2)
    
    # Boundary: push points inward if they're too close to the edge
    norms = np.linalg.norm(pts, axis=1)
    boundary_margin = 1.0 - target_r
    boundary_violation = np.maximum(0, norms - boundary_margin)
    # Radial force: push inward proportional to violation
    radial_force = np.zeros_like(pts)
    mask = boundary_violation > 0
    radial_force[mask] = 2.0 * boundary_violation[mask, np.newaxis] * pts[mask] / (norms[mask, np.newaxis] + eps)
    
    # Total gradient (we want to minimize overlap + boundary violation)
    grad = repulsion + radial_force
    
    return grad


def _gradient_descent(pts, target_r, deadline, lr=0.01, momentum=0.9):
    """
    Perform gradient descent to improve packing with target radius target_r.
    Returns improved points and achieved radius.
    """
    start = time.time()
    n = len(pts)
    v = np.zeros_like(pts)  # velocity for momentum
    
    best_pts = pts.copy()
    best_r = _compute_score(pts)
    
    for step in range(10000):
        elapsed = time.time() - start
        if elapsed > deadline - 5:  # safety margin
            break
        
        grad = _vectorized_gradient(pts, target_r)
        
        # Momentum update
        v = momentum * v - lr * grad
        pts = pts + v
        
        # Project back to unit disk (hard constraint)
        norms = np.linalg.norm(pts, axis=1)
        max_norm = 1.0 - 0.01  # small margin
        mask = norms > max_norm
        if np.any(mask):
            pts[mask] = pts[mask] * (max_norm / norms[mask, np.newaxis])
        
        # Compute current score
        current_r = _compute_score(pts)
        
        if current_r > best_r:
            best_r = current_r
            best_pts = pts.copy()
        
        # Adaptive learning rate: reduce if not improving
        if step % 100 == 0 and step > 0:
            lr = max(0.001, lr * 0.95)
    
    return best_pts, best_r


def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s limit
    
    # Get initial hex lattice configuration
    points, init_r = _hex_lattice_init(N)
    
    best_pts = points.copy()
    best_r = init_r
    
    # Strategy: Try multiple target radii with gradient descent
    # Start from current best and try to improve
    
    # First, try to improve with target slightly above current
    target_r = best_r * 1.01
    remaining = deadline - (time.time() - start_time)
    
    if remaining > 10:
        improved_pts, improved_r = _gradient_descent(best_pts, target_r, remaining)
        if improved_r > best_r:
            best_pts = improved_pts
            best_r = improved_r
    
    # Try a few more target radii
    for multiplier in [1.02, 1.03, 1.05]:
        target_r = init_r * multiplier
        remaining = deadline - (time.time() - start_time)
        if remaining < 10:
            break
        
        improved_pts, improved_r = _gradient_descent(best_pts, target_r, remaining)
        if improved_r > best_r:
            best_pts = improved_pts
            best_r = improved_r
    
    # Final polish: fine-tune with very small steps
    remaining = deadline - (time.time() - start_time)
    if remaining > 10:
        # Use a slightly lower target to refine
        target_r = best_r * 0.999
        improved_pts, improved_r = _gradient_descent(best_pts, target_r, remaining, lr=0.005, momentum=0.95)
        if improved_r > best_r:
            best_pts = improved_pts
            best_r = improved_r
    
    # Final check: ensure all points are within bounds and valid
    final_pts = best_pts.copy()
    
    # Clamp points to be within unit circle (with small margin for radius)
    norms = np.hypot(final_pts[:, 0], final_pts[:, 1])
    max_allowed_norm = 1.0 - best_r
    mask = norms > max_allowed_norm
    if np.any(mask):
        final_pts[mask] = final_pts[mask] * (max_allowed_norm / norms[mask, np.newaxis])
    
    return final_pts


# Test
if __name__ == "__main__":
    start = time.time()
    points = construct()
    elapsed = time.time() - start
    print(f"Shape: {points.shape}")
    print(f"First few points: {points[:5]}")
    
    # Compute score
    r = _compute_score(points)
    print(f"Radius: {r}")
    print(f"Score: {r / 0.0531542991469}")
    print(f"Time: {elapsed:.2f}s")
