"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _score(pts):
    """Compute the packing radius: min of half min pairwise dist and min distance to boundary."""
    n = len(pts)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # Distance to boundary
    norms = np.sqrt(np.sum(pts ** 2, axis=1))
    min_boundary = np.min(1.0 - norms)
    return min(min_pair, min_boundary)


def _soft_score(pts, beta=100.0):
    """Smooth surrogate: soft-min of constraints."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    # soft-min of pairwise distances / 2
    # Use log-sum-exp trick for soft-min: -logsumexp(-beta * x) / beta
    upper_tri = np.triu_indices(n, k=1)
    pd = dists[upper_tri]
    # soft min of pd / 2
    logsumexp_neg = np.log(np.sum(np.exp(-beta * pd / 2.0)))
    soft_pair = -logsumexp_neg / beta
    # Boundary
    norms = np.sqrt(np.sum(pts ** 2, axis=1))
    boundary = 1.0 - norms
    logsumexp_neg_b = np.log(np.sum(np.exp(-beta * boundary)))
    soft_boundary = -logsumexp_neg_b / beta
    # soft min of both
    logsumexp_both = np.log(np.exp(-beta * soft_pair) + np.exp(-beta * soft_boundary))
    return -logsumexp_both / beta


def _hex_lattice_init(n):
    """Initialize with hexagonal lattice points inside unit disk."""
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0 - spacing / 2.0]
        if len(points) >= n:
            # Sort by radius and take n closest to center
            order = np.argsort(radius[radius <= 1.0 - spacing / 2.0], kind="stable")
            return points[order[:n]]
        spacing *= 0.98


def _ring_init(n):
    """Initialize with concentric rings."""
    # Try different ring structures
    best_pts = None
    best_score = -1
    for num_rings in range(1, 10):
        if num_rings > n:
            break
        # Distribute n points among rings
        # First ring has 1 point (center), rest distributed
        remaining = n - 1
        if remaining < 0:
            continue
        # Points per ring
        pts_per_ring = [1]
        rem = remaining
        for r in range(1, num_rings):
            # Increase points per ring
            pts_per_ring.append(min(rem, 6 * r))
            rem -= pts_per_ring[-1]
            if rem <= 0:
                break
        # Adjust to exactly n
        while sum(pts_per_ring) < n:
            pts_per_ring[-1] += 1
        while sum(pts_per_ring) > n:
            pts_per_ring[-1] -= 1
            if pts_per_ring[-1] <= 0:
                pts_per_ring.pop()
        
        if len(pts_per_ring) == 0:
            continue
            
        total_pts = sum(pts_per_ring)
        if total_pts != n:
            continue
            
        # Create points
        points = []
        max_r = 1.0
        for ring_idx, count in enumerate(pts_per_ring):
            if ring_idx == 0:
                points.append([0.0, 0.0])
            else:
                r = max_r * ring_idx / num_rings
                for i in range(count):
                    angle = 2 * np.pi * i / count + np.pi / count  # offset
                    points.append([r * np.cos(angle), r * np.sin(angle)])
        
        if len(points) != n:
            continue
            
        pts = np.array(points)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts
    
    if best_pts is not None:
        return best_pts
    return _hex_lattice_init(n)


def _random_init(n, seed=42):
    """Random initialization inside unit disk."""
    rng = np.random.RandomState(seed)
    while True:
        pts = rng.uniform(-1, 1, size=(n, 2))
        norms = np.sqrt(np.sum(pts ** 2, axis=1))
        if np.all(norms < 1.0):
            return pts


def _force_directed(pts, iterations=200, learning_rate=0.01):
    """Simple force-directed relaxation."""
    pts = pts.copy()
    n = len(pts)
    for it in range(iterations):
        forces = np.zeros_like(pts)
        # Repulsion between all pairs
        for i in range(n):
            for j in range(i + 1, n):
                diff = pts[i] - pts[j]
                dist = np.sqrt(np.sum(diff ** 2))
                if dist < 1e-10:
                    diff = np.random.randn(2) * 1e-5
                    dist = np.sqrt(np.sum(diff ** 2))
                force = diff / dist ** 3
                forces[i] += force
                forces[j] -= force
        # Attraction to keep inside
        for i in range(n):
            norm = np.sqrt(np.sum(pts[i] ** 2))
            if norm > 0.9:
                forces[i] -= pts[i] / norm * 0.5
        pts += learning_rate * forces
        # Project inside unit disk
        norms = np.sqrt(np.sum(pts ** 2, axis=1))
        mask = norms > 0.999
        pts[mask] *= 0.999 / norms[mask, np.newaxis]
    return pts


def _lbfgs_refine(pts, beta=100.0, maxiter=500):
    """Refine using L-BFGS-B on soft score."""
    def objective(x):
        p = x.reshape(-1, 2)
        # Penalty for going outside
        norms = np.sqrt(np.sum(p ** 2, axis=1))
        penalty = np.sum(np.maximum(0, norms - 0.9999) ** 2)
        return -_soft_score(p, beta) + penalty
    
    x0 = pts.flatten()
    result = minimize(objective, x0, method='L-BFGS-B', maxiter=maxiter,
                      options={'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
    pts_refined = result.x.reshape(-1, 2)
    # Project inside
    norms = np.sqrt(np.sum(pts_refined ** 2, axis=1))
    mask = norms > 0.9999
    pts_refined[mask] *= 0.9999 / norms[mask, np.newaxis]
    return pts_refined


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Multiple initialization strategies
    inits = []
    
    # Hex lattice
    pts_hex = _hex_lattice_init(N)
    inits.append(pts_hex)
    
    # Ring structures
    pts_ring = _ring_init(N)
    inits.append(pts_ring)
    
    # Random with different seeds
    for seed in [42, 123, 777]:
        if time.time() > deadline - 30:
            break
        pts_rand = _random_init(N, seed=seed)
        inits.append(pts_rand)
    
    best_pts = None
    best_score = -1
    
    for pts in inits:
        if time.time() > deadline - 20:
            break
        
        # Force-directed relaxation
        pts_fd = _force_directed(pts, iterations=100, learning_rate=0.005)
        s_fd = _score(pts_fd)
        if s_fd > best_score:
            best_score = s_fd
            best_pts = pts_fd.copy()
        
        # L-BFGS refinement with adaptive beta
        if time.time() > deadline - 15:
            break
        
        for beta in [50.0, 100.0, 200.0]:
            if time.time() > deadline - 10:
                break
            pts_ref = _lbfgs_refine(pts_fd, beta=beta, maxiter=300)
            s_ref = _score(pts_ref)
            if s_ref > best_score:
                best_score = s_ref
                best_pts = pts_ref.copy()
    
    # Final polish: more iterations with higher beta
    if best_pts is not None and time.time() < deadline - 5:
        for beta in [200.0, 500.0]:
            if time.time() > deadline - 3:
                break
            pts_final = _lbfgs_refine(best_pts, beta=beta, maxiter=500)
            s_final = _score(pts_final)
            if s_final > best_score:
                best_score = s_final
                best_pts = pts_final.copy()
    
    if best_pts is None:
        best_pts = pts_hex
    
    # Final validity check
    norms = np.sqrt(np.sum(best_pts ** 2, axis=1))
    mask = norms > 0.99999
    best_pts[mask] *= 0.99999 / norms[mask, np.newaxis]
    
    return best_pts
# EVOLVE-BLOCK-END
