"""Rewritten packing solver: multi-start force-directed + gradient polishing."""
import numpy as np
from scipy.spatial import cKDTree
import time

N = 50

# EVOLVE-BLOCK-START

def _score(points):
    """Compute the common radius for a set of points."""
    # Distance to boundary
    boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(boundary)
    # Pairwise distances
    tree = cKDTree(points)
    pairs = tree.query_pairs(2.0)  # generous upper bound
    if len(pairs) == 0:
        return min_boundary
    min_pair = np.inf
    for i, j in pairs:
        d = np.linalg.norm(points[i] - points[j])
        if d < min_pair:
            min_pair = d
    return min(min_pair / 2.0, min_boundary)


def _init_hexagonal():
    """Generate points from a hexagonal lattice inside the unit disk."""
    # Try to fit N points in a hex lattice
    # Optimal hex packing density is pi/(2*sqrt(3)) ~ 0.9069
    # Area of unit disk = pi, so max points ~ pi / (pi/(2*sqrt(3))) * N... 
    # Let's just generate a dense hex grid and pick N closest to center
    spacing = 0.2  # start with some spacing
    pts = []
    r = 1.0
    # Generate hex grid points
    dx = spacing
    dy = spacing * np.sqrt(3) / 2
    for j in range(-10, 11):
        for i in range(-10, 11):
            x = i * dx + (j % 2) * dx / 2
            y = j * dy
            if x*x + y*y < (r - spacing/2)**2:
                pts.append([x, y])
    pts = np.array(pts)
    if len(pts) < N:
        # Reduce spacing
        spacing = 0.15
        pts = []
        dx = spacing
        dy = spacing * np.sqrt(3) / 2
        for j in range(-15, 16):
            for i in range(-15, 16):
                x = i * dx + (j % 2) * dx / 2
                y = j * dy
                if x*x + y*y < (r - spacing/2)**2:
                    pts.append([x, y])
        pts = np.array(pts)
    # Sort by distance from center and take N
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:N]
    return pts[idx]


def _init_rings():
    """Generate points on concentric rings."""
    # Ring 0: center
    # Ring 1: 6 points
    # Ring 2: 12 points
    # Ring 3: 18 points
    # Ring 4: 24 points -> total 61, take 50
    rings = [0, 6, 12, 18, 24]
    pts = [[0.0, 0.0]]
    for ring_idx, count in enumerate(rings[1:]):
        radius = (ring_idx + 1) * 0.22
        angles = np.linspace(0, 2*np.pi, count, endpoint=False)
        for a in angles:
            pts.append([radius * np.cos(a), radius * np.sin(a)])
    pts = np.array(pts[:N])
    return pts


def _init_random(seed):
    """Random points inside unit disk."""
    rng = np.random.RandomState(seed)
    pts = []
    while len(pts) < N:
        r = np.sqrt(rng.rand())
        theta = 2 * np.pi * rng.rand()
        pts.append([r * np.cos(theta), r * np.sin(theta)])
    return np.array(pts)


def _force_directed_optimize(pts, max_iter=2000, learning_rate=0.01, time_limit=None):
    """
    Force-directed optimization: repulsive forces between circles, attractive to boundary.
    """
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    best_score = _score(pts)
    best_pts = pts.copy()
    
    if time_limit is None:
        time_limit = 12.0
    start_time = time.time()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Compute forces
        forces = np.zeros_like(pts)
        
        # Repulsive forces between pairs
        for i in range(n):
            for j in range(i+1, n):
                d = pts[i] - pts[j]
                dist = np.linalg.norm(d)
                if dist < 1e-12:
                    # Perturb randomly
                    d = np.random.randn(2) * 1e-6
                    dist = np.linalg.norm(d)
                # Force proportional to 1/dist^2, pushing apart
                f = d / (dist**3)
                forces[i] += f
                forces[j] -= f
        
        # Attractive force to boundary (push outward)
        for i in range(n):
            r = np.linalg.norm(pts[i])
            if r > 1e-12:
                # Push toward boundary
                forces[i] += 0.5 * pts[i] / r
        
        # Update positions
        pts += learning_rate * forces
        
        # Clamp to unit disk
        norms = np.linalg.norm(pts, axis=1)
        for i in range(n):
            if norms[i] > 0.99:
                pts[i] *= 0.99 / norms[i]
        
        # Check improvement
        score = _score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
        
        # Decay learning rate
        learning_rate *= 0.999
    
    return best_pts, best_score


def _gradient_polish(pts, max_iter=5000, time_limit=None):
    """
    Gradient-based polishing using the min of pairwise distances and boundary distances.
    """
    if time_limit is None:
        time_limit = 15.0
    start_time = time.time()
    
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    best_score = _score(pts)
    best_pts = pts.copy()
    
    lr = 0.001
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Compute gradient of the score function
        # Score = min(min_pair_dist/2, min_boundary)
        # We want to maximize this.
        
        # Compute all pairwise distances and boundary distances
        pair_dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                pair_dists[i, j] = d
                pair_dists[j, i] = d
        
        boundary_dists = 1.0 - np.linalg.norm(pts, axis=1)
        
        # Find the minimum value and which constraints are active
        min_pair = np.min(pair_dists[np.triu_indices(n, k=1)])
        min_boundary = np.min(boundary_dists)
        
        current_score = min(min_pair / 2.0, min_boundary)
        
        # Compute gradient: we want to increase the minimum constraint
        grad = np.zeros_like(pts)
        
        # If boundary is the limiting factor
        if min_boundary <= min_pair / 2.0:
            # Active boundary constraints
            for i in range(n):
                if boundary_dists[i] < min_boundary + 1e-6:
                    r = np.linalg.norm(pts[i])
                    if r > 1e-12:
                        # Gradient of (1 - |p_i|) wrt p_i = -p_i / |p_i|
                        # We want to maximize, so move in direction +p_i/|p_i|
                        grad[i] += pts[i] / r
        
        # If pair distance is the limiting factor
        if min_pair / 2.0 <= min_boundary + 1e-6:
            for i in range(n):
                for j in range(i+1, n):
                    if pair_dists[i, j] < min_pair + 1e-6:
                        d = pts[i] - pts[j]
                        dist = np.linalg.norm(d)
                        if dist > 1e-12:
                            # Gradient of |p_i - p_j| wrt p_i = d/|d|
                            # wrt p_j = -d/|d|
                            # We want to increase distance, so move i in +d/|d| and j in -d/|d|
                            grad[i] += d / dist
                            grad[j] -= d / dist
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad /= grad_norm
        
        # Update
        pts += lr * grad
        
        # Clamp to unit disk
        norms = np.linalg.norm(pts, axis=1)
        for i in range(n):
            if norms[i] > 0.999:
                pts[i] *= 0.999 / norms[i]
        
        # Check improvement
        score = _score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
        
        # Decay learning rate
        lr *= 0.9995
    
    return best_pts, best_score


def construct():
    deadline = time.time() + 115.0  # Safety margin
    
    best_pts = None
    best_score = 0.0
    
    # Try multiple initializations
    inits = [
        ("hex", _init_hexagonal()),
        ("rings", _init_rings()),
    ]
    
    # Add some random inits
    for seed in range(5):
        inits.append((f"rand{seed}", _init_random(seed)))
    
    for name, init_pts in inits:
        if time.time() > deadline - 20.0:
            break
        
        # Force-directed optimization
        fd_pts, fd_score = _force_directed_optimize(init_pts, max_iter=1500, time_limit=10.0)
        if fd_score > best_score:
            best_score = fd_score
            best_pts = fd_pts.copy()
        
        # Gradient polish
        if time.time() > deadline - 10.0:
            break
        gp_pts, gp_score = _gradient_polish(fd_pts, max_iter=3000, time_limit=8.0)
        if gp_score > best_score:
            best_score = gp_score
            best_pts = gp_pts.copy()
    
    # Final verification and cleanup
    if best_pts is None:
        # Fallback: hexagonal
        best_pts = _init_hexagonal()
    
    # Ensure all points are strictly inside the unit circle
    norms = np.linalg.norm(best_pts, axis=1)
    for i in range(len(best_pts)):
        if norms[i] >= 1.0:
            best_pts[i] *= 0.999 / norms[i]
    
    return best_pts

# EVOLVE-BLOCK-END
