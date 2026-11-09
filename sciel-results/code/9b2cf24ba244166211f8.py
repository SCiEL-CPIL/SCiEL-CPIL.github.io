"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import time

_DEADLINE = time.time() + 130  # safety margin before 150s kill


def _score(pts):
    """Compute the packing radius for a set of points in [-0.5, 0.5]^2."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Check inside square
    if np.any(pts < -0.5 + 1e-12) or np.any(pts > 0.5 - 1e-12):
        return 0.0
    # Distance to boundary
    dist_boundary = np.min(np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5), axis=1)
    dist_boundary = np.minimum(dist_boundary, np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5))
    r_boundary = np.min(dist_boundary)
    # Nearest neighbor distances
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    r_nn = np.min(dists[:, 1]) / 2.0
    return min(r_boundary, r_nn)


def _initial_grid():
    """Create initial grid placement."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].astype(float)


def _repulsion_step(pts, tree, lr, n_iters=50):
    """Perform repulsion-based optimization to improve packing."""
    best_pts = pts.copy()
    best_score = _score(pts)
    
    for it in range(n_iters):
        if time.time() > _DEADLINE - 5:
            break
        # Compute repulsion forces
        forces = np.zeros_like(pts)
        for i in range(N):
            # Repulsion from other circles
            dists, idxs = tree.query(pts[i], k=5)
            for j in range(1, len(idxs)):
                if idxs[j] == i:
                    continue
                d = dists[j]
                if d < 1e-10:
                    # Random perturbation for coincident points
                    forces[i] += np.random.randn(2) * 0.01
                else:
                    # Repulsive force inversely proportional to distance
                    f = 1.0 / (d * d)
                    forces[i] += (pts[i] - pts[idxs[j]]) / d * f
            # Attraction toward center to keep points in bounds
            forces[i] -= pts[i] * 0.01
        
        # Update positions with learning rate
        new_pts = pts + lr * forces
        # Clamp to square
        new_pts = np.clip(new_pts, -0.5 + 1e-6, 0.5 - 1e-6)
        
        # Check if improvement
        new_score = _score(new_pts)
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            pts = new_pts
            tree = cKDTree(pts)
        else:
            # Reduce learning rate if no improvement
            lr *= 0.95
            # Occasionally try a random perturbation
            if np.random.random() < 0.1:
                idx = np.random.randint(N)
                pts[idx] += np.random.randn(2) * 0.005
                pts[idx] = np.clip(pts[idx], -0.5 + 1e-6, 0.5 - 1e-6)
                tree = cKDTree(pts)
    
    return best_pts, best_score


def _hill_climbing(pts, max_time=100):
    """Perform hill climbing optimization."""
    best_pts = pts.copy()
    best_score = _score(pts)
    tree = cKDTree(pts)
    lr = 0.005
    start_time = time.time()
    
    while time.time() - start_time < max_time and time.time() < _DEADLINE - 5:
        # Try moving a random subset of points
        n_move = np.random.randint(1, 10)
        idxs = np.random.choice(N, size=n_move, replace=False)
        
        # Compute forces for these points
        forces = np.zeros((n_move, 2))
        for i, idx in enumerate(idxs):
            dists, neighbors = tree.query(pts[idx], k=5)
            for j in range(1, len(neighbors)):
                if neighbors[j] == idx:
                    continue
                d = dists[j]
                if d < 1e-10:
                    forces[i] += np.random.randn(2) * 0.01
                else:
                    f = 1.0 / (d * d)
                    forces[i] += (pts[idx] - pts[neighbors[j]]) / d * f
            forces[i] -= pts[idx] * 0.01
        
        # Propose new positions
        new_pts = pts.copy()
        new_pts[idxs] += lr * forces
        new_pts = np.clip(new_pts, -0.5 + 1e-6, 0.5 - 1e-6)
        
        new_score = _score(new_pts)
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            pts = new_pts
            tree = cKDTree(pts)
            lr = min(lr * 1.05, 0.01)
        else:
            lr *= 0.98
    
    return best_pts, best_score


def construct():
    """Construct optimal packing of N circles in unit square."""
    start_time = time.time()
    
    # Start with grid
    pts = _initial_grid()
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Phase 1: Global repulsion optimization
    tree = cKDTree(pts)
    pts, score = _repulsion_step(pts, tree, lr=0.002, n_iters=100)
    if score > best_score:
        best_score = score
        best_pts = pts.copy()
    
    # Phase 2: Hill climbing
    pts, score = _hill_climbing(pts, max_time=100)
    if score > best_score:
        best_score = score
        best_pts = pts.copy()
    
    # Final validation
    if _score(best_pts) < best_score * 0.999:
        # Fallback to grid if something went wrong
        return _initial_grid()
    
    return best_pts
# EVOLVE-BLOCK-END
