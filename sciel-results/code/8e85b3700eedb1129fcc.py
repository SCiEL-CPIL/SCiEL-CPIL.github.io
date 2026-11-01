"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_score(pts):
    """Compute the packing score (max radius)."""
    if len(pts) != N:
        return -1.0
    if not np.all(np.isfinite(pts)):
        return -1.0
    dists = np.linalg.norm(pts, axis=1)
    if np.any(dists > 1.0):
        return -1.0
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    sq_dists = np.sum(diff * diff, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    min_dist = np.sqrt(np.min(sq_dists))
    r = min(min_dist / 2.0, 1.0 - np.max(dists))
    return r

def _hex_init():
    """Generate hexagonal lattice initialization."""
    rng = np.random.default_rng(42)
    for spacing in [0.06, 0.058, 0.056, 0.055, 0.054, 0.053]:
        pts = []
        k = int(1.0 / spacing) + 2
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing * 0.55:
                    pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= N:
            r = np.linalg.norm(pts, axis=1)
            pts = pts[np.argsort(r, kind='stable')[:N]]
            return pts
    return None

def _ring_init():
    """Generate concentric ring initialization."""
    # Place points on concentric rings
    pts = [[0.0, 0.0]]  # center
    remaining = N - 1
    ring_idx = 1
    while remaining > 0:
        # Number of points on this ring
        n_ring = min(6 * ring_idx, remaining)
        radius = 0.55 * ring_idx / np.sqrt(N) * 2.0
        if radius > 0.95:
            radius = 0.95
        angles = np.linspace(0, 2 * np.pi, n_ring, endpoint=False) + (ring_idx % 2) * np.pi / n_ring
        for a in angles:
            pts.append([radius * np.cos(a), radius * np.sin(a)])
        remaining -= n_ring
        ring_idx += 1
    return np.array(pts[:N])

def _random_init(seed=0):
    """Generate random initialization with rejection sampling."""
    rng = np.random.default_rng(seed)
    pts = []
    # Start with a minimum radius estimate
    min_r = 0.04
    attempts = 0
    while len(pts) < N and attempts < 500000:
        x = rng.uniform(-1, 1)
        y = rng.uniform(-1, 1)
        if x*x + y*y > 1.0:
            continue
        # Check distance to all existing points
        if len(pts) > 0:
            dists = np.sqrt(np.sum((np.array(pts) - [x, y])**2, axis=1))
            if np.min(dists) < min_r:
                continue
        pts.append([x, y])
        attempts += 1
    if len(pts) < N:
        return None
    return np.array(pts[:N])

def _repulsion_optimize(pts, deadline, max_iters=200):
    """Iterative closest-pair repulsion optimization.
    
    At each step, find the closest pair (or point closest to boundary),
    and push them apart. This is a simple but effective local search.
    """
    pts = pts.copy()
    n = len(pts)
    score = _compute_score(pts)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute all pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff * diff, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        
        # Find closest pair
        min_sq = np.min(sq_dists)
        min_dist = np.sqrt(min_sq)
        
        # Find closest to boundary
        radii = np.linalg.norm(pts, axis=1)
        max_radius = np.max(radii)
        boundary_margin = 1.0 - max_radius
        
        # Determine which constraint is tightest
        # Pair constraint: min_dist / 2
        # Boundary constraint: boundary_margin
        pair_constraint = min_dist / 2.0
        boundary_constraint = boundary_margin
        
        if pair_constraint < boundary_constraint:
            # Closest pair is the bottleneck
            i, j = np.unravel_index(np.argmin(sq_dists), (n, n))
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist > 1e-12:
                direction = d / dist
                # Push apart
                push = 0.001 + 0.0005 * (it / max_iters)
                pts[i] += direction * push
                pts[j] -= direction * push
        else:
            # Boundary is the bottleneck
            idx = np.argmax(radii)
            r = radii[idx]
            if r > 1e-12:
                direction = pts[idx] / r
                # Push inward
                push = 0.001 + 0.0005 * (it / max_iters)
                pts[idx] -= direction * push
        
        # Project all points back inside the unit circle
        radii = np.linalg.norm(pts, axis=1)
        for i in range(n):
            if radii[i] > 0.999:
                pts[i] *= 0.999 / radii[i]
        
        # Compute new score
        new_score = _compute_score(pts)
        if new_score > score:
            score = new_score
        else:
            # If score didn't improve, try smaller steps
            pass
    
    return pts, score

def _multi_repulsion(pts, deadline, n_restarts=5):
    """Run repulsion optimization from multiple perturbed starting points."""
    best_pts = pts.copy()
    best_score = _compute_score(pts)
    
    rng = np.random.default_rng(123)
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Perturb the starting point
        perturbation = 0.002 * (restart + 1)
        perturbed = pts + rng.normal(0, perturbation, size=pts.shape)
        
        # Project inside
        radii = np.linalg.norm(perturbed, axis=1)
        for i in range(len(perturbed)):
            if radii[i] > 0.999:
                perturbed[i] *= 0.999 / radii[i]
        
        # Run repulsion
        optimized, score = _repulsion_optimize(perturbed, deadline, max_iters=100)
        
        if score > best_score:
            best_score = score
            best_pts = optimized.copy()
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Try multiple initializations
    best_points = None
    best_score = -1.0
    
    # 1. Hexagonal lattice
    hex_pts = _hex_init()
    if hex_pts is not None:
        score = _compute_score(hex_pts)
        if score > best_score:
            best_score = score
            best_points = hex_pts.copy()
    
    # 2. Ring initialization
    ring_pts = _ring_init()
    score = _compute_score(ring_pts)
    if score > best_score:
        best_score = score
        best_points = ring_pts.copy()
    
    # 3. Random initialization
    for seed in [0, 1, 2]:
        if time.time() > deadline - 30:
            break
        rand_pts = _random_init(seed)
        if rand_pts is not None:
            score = _compute_score(rand_pts)
            if score > best_score:
                best_score = score
                best_points = rand_pts.copy()
    
    if best_points is None:
        # Fallback: simple grid
        best_points = np.zeros((N, 2))
        best_score = 0.0
    
    # Now optimize using repulsion from the best initialization
    if time.time() < deadline - 10:
        optimized, score = _multi_repulsion(best_points, deadline, n_restarts=8)
        if score > best_score:
            best_score = score
            best_points = optimized.copy()
    
    # Final validation
    final_score = _compute_score(best_points)
    if final_score <= 0:
        # Return a valid but simple configuration
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)])
    
    return best_points
# EVOLVE-BLOCK-END
