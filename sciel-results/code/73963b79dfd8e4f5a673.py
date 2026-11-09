"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _score_fast(pts):
    """Compute the radius (minimum of half pairwise distances and half min distance to boundary)."""
    # pts: (N, 3)
    # distance to boundary: min over axes of (0.5 - |coord|)
    # For each point, the distance to the nearest face is min(0.5 - x, 0.5 + x) = 0.5 - |x|
    # But we need the minimum over all points of this value
    # Actually, the constraint is that for each point i and each coordinate k,
    # the distance from c_ik to the nearest boundary face must be >= r.
    # The nearest face for coordinate k is at -0.5 or 0.5.
    # Distance to -0.5 face: c_ik - (-0.5) = c_ik + 0.5
    # Distance to 0.5 face: 0.5 - c_ik
    # So min distance for point i is min over k of min(c_ik + 0.5, 0.5 - c_ik) = min over k of (0.5 - |c_ik|)
    
    # Pairwise distances
    # Compute all pairwise distances efficiently
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    # Set diagonal to inf
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists)
    
    # Boundary constraint
    # For each point, compute min distance to any face
    # dist_to_face[i, k] = 0.5 - |pts[i, k]|
    dist_to_face = 0.5 - np.abs(pts)  # shape (N, 3)
    min_boundary = np.min(dist_to_face)
    
    return min(min_pairwise / 2.0, min_boundary)


def _score_fast_v2(pts):
    """Faster score computation using vectorized operations."""
    # Pairwise distances using broadcasting
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_pairwise_sq = np.min(dists_sq)
    min_pairwise = np.sqrt(min_pairwise_sq)
    
    # Boundary
    dist_to_face = 0.5 - np.abs(pts)
    min_boundary = np.min(dist_to_face)
    
    return min(min_pairwise / 2.0, min_boundary)


def _random_initial(n=50, seed=None):
    """Generate random initial positions in [-0.5, 0.5]^3."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    return rng.uniform(-0.5, 0.5, size=(n, 3))


def _simple_cubic_grid(n=50):
    """Simple cubic grid initialization."""
    m = int(np.ceil(n ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]


def _fbb_placement(n=50, seed=None):
    """Furthest point sampling / best-packing style initialization."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    # Start with a random point
    pts = [rng.uniform(-0.5, 0.5, size=3)]
    
    for _ in range(1, n):
        # Generate many candidates and pick the one that maximizes min distance to existing points and boundary
        candidates = rng.uniform(-0.5, 0.5, size=(2000, 3))
        
        best_score = -1
        best_candidate = None
        
        for c in candidates:
            # Distance to existing points
            if len(pts) > 0:
                dists_to_pts = np.sqrt(np.sum((np.array(pts) - c) ** 2, axis=1))
                min_dist_to_pts = np.min(dists_to_pts)
            else:
                min_dist_to_pts = np.inf
            
            # Distance to boundary
            dist_to_boundary = np.min(0.5 - np.abs(c))
            
            score = min(min_dist_to_pts, dist_to_boundary)
            
            if score > best_score:
                best_score = score
                best_candidate = c
        
        pts.append(best_candidate)
    
    return np.array(pts)


def _simulated_annealing(pts, max_iter=50000, start_temp=0.05, end_temp=0.0001, 
                         step_scale=0.1, deadline=None):
    """Simulated annealing to optimize sphere packing."""
    n = len(pts)
    best_pts = pts.copy()
    best_score = _score_fast_v2(pts)
    
    current_pts = pts.copy()
    current_score = best_score
    
    # Precompute indices
    all_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            all_pairs.append((i, j))
    
    rng = np.random.RandomState(42)
    
    start_time = time.time()
    for it in range(max_iter):
        # Check deadline
        if deadline is not None:
            if time.time() > deadline:
                break
        
        # Compute temperature
        t = start_temp * (end_temp / start_temp) ** (it / max_iter)
        
        # Pick a random point to move
        i = rng.randint(n)
        
        # Generate a perturbation
        step_size = step_scale * t
        delta = rng.normal(0, step_size, size=3)
        
        new_pts = current_pts.copy()
        new_pts[i] = current_pts[i] + delta
        
        # Clamp to cube
        new_pts[i] = np.clip(new_pts[i], -0.5, 0.5)
        
        # Compute new score
        new_score = _score_fast_v2(new_pts)
        
        # Accept or reject
        if new_score >= current_score:
            current_pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
        else:
            # Metropolis criterion
            delta_e = current_score - new_score
            if delta_e < 0:
                current_pts = new_pts
                current_score = new_score
            else:
                accept_prob = np.exp(-delta_e / max(t, 1e-10))
                if rng.random() < accept_prob:
                    current_pts = new_pts
                    current_score = new_score
    
    return best_pts, best_score


def _local_optimization(pts, max_iter=10000, deadline=None):
    """Local optimization using coordinate descent and pairwise repulsion."""
    n = len(pts)
    best_pts = pts.copy()
    best_score = _score_fast_v2(pts)
    
    current_pts = pts.copy()
    current_score = best_score
    
    rng = np.random.RandomState(123)
    
    start_time = time.time()
    
    # Use a simple approach: repeatedly try to move points apart
    for it in range(max_iter):
        if deadline is not None:
            if time.time() > deadline:
                break
        
        # Try to improve by moving the most constrained point
        # Find the point with the smallest margin
        dist_to_face = 0.5 - np.abs(current_pts)
        min_boundary_per_point = np.min(dist_to_face, axis=1)
        
        # Also consider pairwise constraints
        diff = current_pts[:, np.newaxis, :] - current_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pairwise_per_point = np.min(dists, axis=1)
        
        # The limiting constraint for each point
        limiting = np.minimum(min_boundary_per_point, min_pairwise_per_point / 2)
        
        # Pick the most constrained point
        i = np.argmin(limiting)
        
        # Try several random perturbations
        best_local_score = current_score
        best_local_pts = current_pts.copy()
        
        for trial in range(20):
            step = 0.01 * (1 + rng.random())
            delta = rng.normal(0, step, size=3)
            new_pts = current_pts.copy()
            new_pts[i] = current_pts[i] + delta
            new_pts[i] = np.clip(new_pts[i], -0.5, 0.5)
            
            new_score = _score_fast_v2(new_pts)
            
            if new_score > best_local_score:
                best_local_score = new_score
                best_local_pts = new_pts.copy()
        
        if best_local_score > current_score:
            current_pts = best_local_pts
            current_score = best_local_score
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
    
    return best_pts, best_score


def construct():
    """Construct the sphere packing configuration."""
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Start with a good initial configuration
    # Use a combination of grid and random initialization
    
    # First, try the simple grid as a baseline
    grid_pts = _simple_cubic_grid(N)
    grid_score = _score_fast_v2(grid_pts)
    
    best_pts = grid_pts.copy()
    best_score = grid_score
    
    # Try simulated annealing from the grid
    if time.time() < deadline:
        sa_pts, sa_score = _simulated_annealing(
            grid_pts, 
            max_iter=30000, 
            start_temp=0.05, 
            end_temp=0.0001,
            step_scale=0.1,
            deadline=deadline
        )
        if sa_score > best_score:
            best_pts = sa_pts
            best_score = sa_score
    
    # Try local optimization
    if time.time() < deadline:
        lo_pts, lo_score = _local_optimization(
            best_pts,
            max_iter=5000,
            deadline=deadline
        )
        if lo_score > best_score:
            best_pts = lo_pts
            best_score = lo_score
    
    # Try a few random restarts with SA if time permits
    for restart in range(5):
        if time.time() > deadline - 10:
            break
        
        random_pts = _random_initial(N, seed=100 + restart)
        # Quick SA from random start
        sa_pts, sa_score = _simulated_annealing(
            random_pts,
            max_iter=10000,
            start_temp=0.03,
            end_temp=0.0001,
            step_scale=0.08,
            deadline=deadline
        )
        if sa_score > best_score:
            best_pts = sa_pts
            best_score = sa_score
    
    # Final local optimization
    if time.time() < deadline:
        lo_pts, lo_score = _local_optimization(
            best_pts,
            max_iter=3000,
            deadline=deadline
        )
        if lo_score > best_score:
            best_pts = lo_pts
            best_score = lo_score
    
    return best_pts
# EVOLVE-BLOCK-END
