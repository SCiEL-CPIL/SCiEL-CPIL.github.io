"""Physics-based circle packing: repulsive forces + random restarts."""
import time
import numpy as np
from scipy.spatial import cKDTree

N = 300
_START_TIME = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill


def _score(points):
    """Compute the packing radius: min of inter-circle half-distances and boundary distances."""
    if len(points) < 2:
        return 0.0
    tree = cKDTree(points)
    pairs = tree.query_pairs(1.0)
    if len(pairs) == 0:
        min_dist = 1.0
    else:
        min_dist = np.inf
        for i, j in pairs:
            d = np.linalg.norm(points[i] - points[j])
            if d < min_dist:
                min_dist = d
    boundary = 1.0 - np.max(np.linalg.norm(points, axis=1))
    return min(min_dist / 2.0, boundary)


def _initial_hex(n_points):
    """Generate a hexagonal lattice point cloud, then trim to n_points."""
    # Aim for roughly a hexagonal region of area pi (unit circle)
    # With 300 points, spacing ~ sqrt(pi/300 * 2/sqrt(3))
    spacing = 0.058
    pts = []
    k = int(1.0 / spacing) + 2
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x * x + y * y <= 0.95:
                pts.append((x, y))
    pts = np.array(pts)
    if len(pts) > n_points:
        # Keep closest to center
        r = np.linalg.norm(pts, axis=1)
        idx = np.argsort(r, kind="stable")[:n_points]
        pts = pts[idx]
    elif len(pts) < n_points:
        # Add random points
        extra = np.random.randn(n_points - len(pts), 2) * 0.5
        pts = np.vstack([pts, extra])
    return pts


def _repulsion_step(points, alpha, k_rep, k_bound, max_iter=200):
    """
    Simple repulsive force simulation.
    - Repel between all pairs (approximate with KD-tree for speed)
    - Pull toward center if outside boundary
    """
    n = len(points)
    points = points.copy()
    
    for it in range(max_iter):
        if time.time() > _DEADLINE:
            break
        
        # Compute forces
        forces = np.zeros_like(points)
        
        # Pairwise repulsion using KD-tree
        tree = cKDTree(points)
        pairs = tree.query_pairs(0.3)  # only nearby pairs matter
        
        for i, j in pairs:
            d = points[i] - points[j]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                d = np.random.randn(2) * 1e-6
                dist = np.linalg.norm(d)
            # Repulsive force ~ 1/dist^2, capped
            f = k_rep / (dist * dist)
            f = min(f, 10.0)
            d_hat = d / dist
            forces[i] += f * d_hat
            forces[j] -= f * d_hat
        
        # Boundary constraint: push inward if outside
        r = np.linalg.norm(points, axis=1)
        outside = r > 0.99
        if np.any(outside):
            for idx in np.where(outside)[0]:
                pull = k_bound * (r[idx] - 0.99)
                if r[idx] > 1e-12:
                    forces[idx] -= pull * points[idx] / r[idx]
                else:
                    forces[idx] += pull * np.random.randn(2)
        
        # Small centering force to keep cluster centered
        forces -= 0.01 * points
        
        # Update positions with damping
        points += alpha * forces
        alpha *= 0.995  # cool down
        
        # Hard clip to boundary
        r = np.linalg.norm(points, axis=1)
        mask = r > 0.999
        if np.any(mask):
            points[mask] *= 0.999 / r[mask]
    
    return points


def construct():
    best_points = None
    best_score = -1.0
    
    # Try multiple random restarts
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > _DEADLINE - 10.0:
            break
        
        # Initialize
        if restart == 0:
            points = _initial_hex(N)
        else:
            # Perturb the best so far, or start fresh
            if best_points is not None and restart < 3:
                points = best_points + np.random.randn(N, 2) * 0.005
            else:
                points = _initial_hex(N) + np.random.randn(N, 2) * 0.02
        
        # Ensure valid start (inside circle)
        r = np.linalg.norm(points, axis=1)
        points = points * np.minimum(1.0, 0.99 / np.maximum(r, 1e-12))[:, None]
        
        # Run repulsion simulation
        points = _repulsion_step(points, alpha=0.01, k_rep=0.5, k_bound=5.0, max_iter=300)
        
        # Evaluate
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
        
        # If we have time, do a second pass with finer tuning
        if time.time() < _DEADLINE - 20.0:
            points = _repulsion_step(points, alpha=0.003, k_rep=0.3, k_bound=8.0, max_iter=200)
            s = _score(points)
            if s > best_score:
                best_score = s
                best_points = points.copy()
    
    if best_points is None:
        # Fallback: hexagonal lattice
        best_points = _initial_hex(N)
    
    return best_points
