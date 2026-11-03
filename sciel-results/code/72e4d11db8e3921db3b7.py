"""Packing 300 equal circles in the unit disk.
Strategy:
1. Generate initial positions using hexagonal lattice.
2. Refine with gradient ascent on soft-min radius (log-sum-exp).
3. Local refinement of most constrained points.
4. Final high-precision polish pass.
5. Deadline guard to ensure termination within 120s.
"""
import numpy as np
import time
from scipy.spatial import cKDTree

N = 300

# EVOLVE-BLOCK-START
def _hex_init(n, spacing=None):
    """Generate initial hexagonal lattice positions inside unit disk."""
    if spacing is None:
        spacing = 2.0 / (np.sqrt(n) + 1.0)
    while True:
        k = int(1.0 / spacing) + 3
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points, radius = points[mask], radius[mask]
        if len(points) >= n:
            return points[np.argsort(radius, kind="stable")[:n]]
        spacing *= 0.97


def _compute_radius(pts):
    """Compute the minimum radius for a given set of points."""
    n = len(pts)
    if n < 2:
        return float(np.min(1.0 - np.linalg.norm(pts, axis=1)))
    
    norms = np.linalg.norm(pts, axis=1)
    min_boundary = float(np.min(1.0 - norms))
    
    # Use KD-tree for efficient nearest neighbor distances
    tree = cKDTree(pts)
    # Query k=2 to get nearest neighbor (excluding self)
    dists, _ = tree.query(pts, k=2)
    min_pair_dist = float(np.min(dists[:, 1]))
    
    return min(min_pair_dist / 2.0, min_boundary)


def _soft_min_logsumexp(values, k):
    """Compute soft min using log-sum-exp."""
    m = float(np.min(values))
    shifted = values - m
    log_sum_exp = np.log(np.sum(np.exp(-k * shifted)))
    soft = m - np.log_sum_exp / k
    weights = np.exp(-k * shifted) / np.sum(np.exp(-k * shifted))
    return soft, weights


def _objective_and_grad(pts, k, threshold=0.2):
    """
    Compute soft-min radius and its gradient.
    Returns (soft_value, gradient_array).
    """
    n = len(pts)
    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms
    
    # Soft min of boundary constraints
    soft_boundary, weights_b = _soft_min_logsumexp(boundary, k)
    
    # Gradient of soft_boundary w.r.t. pts
    grad_boundary = np.zeros_like(pts)
    for i in range(n):
        if norms[i] > 1e-10:
            grad_boundary[i] = -weights_b[i] * pts[i] / norms[i]
    
    # Pairwise distances using KD-tree
    tree = cKDTree(pts)
    # Get pairs within threshold
    pairs = tree.query_pairs(r=threshold)
    
    if len(pairs) == 0:
        return soft_boundary, grad_boundary
    
    # Compute distances for these pairs
    pair_i = np.array([p[0] for p in pairs])
    pair_j = np.array([p[1] for p in pairs])
    diff = pts[pair_i] - pts[pair_j]
    dists = np.linalg.norm(diff, axis=1)
    
    # Filter out zero distances
    valid = dists > 1e-10
    if not np.any(valid):
        return soft_boundary, grad_boundary
    
    pair_i = pair_i[valid]
    pair_j = pair_j[valid]
    diff = diff[valid]
    dists = dists[valid]
    pair_radii = dists / 2.0
    
    # Soft min of pair radii
    soft_pair, weights_p = _soft_min_logsumexp(pair_radii, k)
    
    # Gradient of soft_pair w.r.t. pts
    grad_pair = np.zeros_like(pts)
    for idx in range(len(pair_i)):
        w = weights_p[idx]
        if w < 1e-12:
            continue
        i, j = pair_i[idx], pair_j[idx]
        dist_ij = dists[idx]
        unit = diff[idx] / dist_ij
        grad_pair[i] += w * unit / 2.0
        grad_pair[j] -= w * unit / 2.0
    
    # Combine: take the tighter constraint
    if soft_boundary < soft_pair:
        return soft_boundary, grad_boundary
    else:
        return soft_pair, grad_pair


def _objective_only(pts, k, threshold=0.2):
    """Compute only the soft-min radius value."""
    n = len(pts)
    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms
    soft_boundary, _ = _soft_min_logsumexp(boundary, k)
    
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=threshold)
    
    if len(pairs) == 0:
        return soft_boundary
    
    pair_i = np.array([p[0] for p in pairs])
    pair_j = np.array([p[1] for p in pairs])
    diff = pts[pair_i] - pts[pair_j]
    dists = np.linalg.norm(diff, axis=1)
    
    valid = dists > 1e-10
    if not np.any(valid):
        return soft_boundary
    
    pair_radii = dists[valid] / 2.0
    soft_pair, _ = _soft_min_logsumexp(pair_radii, k)
    
    return min(soft_boundary, soft_pair)


def _grad_only(pts, k, threshold=0.2):
    """Compute only the gradient of soft-min radius."""
    n = len(pts)
    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms
    soft_boundary, weights_b = _soft_min_logsumexp(boundary, k)
    
    grad_boundary = np.zeros_like(pts)
    for i in range(n):
        if norms[i] > 1e-10:
            grad_boundary[i] = -weights_b[i] * pts[i] / norms[i]
    
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=threshold)
    
    if len(pairs) == 0:
        return grad_boundary
    
    pair_i = np.array([p[0] for p in pairs])
    pair_j = np.array([p[1] for p in pairs])
    diff = pts[pair_i] - pts[pair_j]
    dists = np.linalg.norm(diff, axis=1)
    
    valid = dists > 1e-10
    if not np.any(valid):
        return grad_boundary
    
    pair_i = pair_i[valid]
    pair_j = pair_j[valid]
    diff = diff[valid]
    dists = dists[valid]
    pair_radii = dists / 2.0
    
    soft_pair, weights_p = _soft_min_logsumexp(pair_radii, k)
    
    grad_pair = np.zeros_like(pts)
    for idx in range(len(pair_i)):
        w = weights_p[idx]
        if w < 1e-12:
            continue
        i, j = pair_i[idx], pair_j[idx]
        dist_ij = dists[idx]
        unit = diff[idx] / dist_ij
        grad_pair[i] += w * unit / 2.0
        grad_pair[j] -= w * unit / 2.0
    
    if soft_boundary < soft_pair:
        return grad_boundary
    else:
        return grad_pair


def _tighten_min_constraint(pts, time_budget, k=50.0, lr=0.002):
    """
    Focus optimization on the most constrained points (smallest cells).
    Identify points with small Voronoi-like cells and optimize them more aggressively.
    """
    pts = pts.copy()
    best_pts = pts.copy()
    best_score = _compute_radius(pts)
    
    t0 = time.time()
    step = 0
    patience = 0
    max_patience = 200
    
    while time.time() - t0 < time_budget:
        # Compute current radius
        current_score = _compute_radius(pts)
        if current_score > best_score:
            best_score = current_score
            best_pts = pts.copy()
            patience = 0
        else:
            patience += 1
        
        if patience > max_patience:
            break
        
        # Compute gradient
        grad = _grad_only(pts, k, threshold=0.15)
        
        # Gradient ascent
        pts = pts + lr * grad
        
        # Keep points inside the unit disk
        norms = np.linalg.norm(pts, axis=1)
        too_far = norms > 0.999
        if np.any(too_far):
            pts[too_far] = pts[too_far] * (0.999 / norms[too_far])
        
        step += 1
        if step % 30 == 0:
            lr = max(0.0003, lr * 0.92)
            k = min(300.0, k * 1.04)
    
    return best_pts, best_score


def _local_refine(pts, time_budget, n_local=30):
    """
    Local refinement: identify the most constrained points and optimize them
    with higher learning rate while keeping others fixed.
    """
    pts = pts.copy()
    best_pts = pts.copy()
    best_score = _compute_radius(pts)
    
    t0 = time.time()
    step = 0
    patience = 0
    max_patience = 150
    
    while time.time() - t0 < time_budget:
        # Compute current radius
        current_score = _compute_radius(pts)
        if current_score > best_score:
            best_score = current_score
            best_pts = pts.copy()
            patience = 0
        else:
            patience += 1
        
        if patience > max_patience:
            break
        
        # Identify most constrained points: those with smallest distance to neighbors or boundary
        norms = np.linalg.norm(pts, axis=1)
        boundary_dist = 1.0 - norms
        
        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=2)
        neighbor_dist = dists[:, 1]
        
        # Combined constraint score
        constraint_score = np.minimum(boundary_dist, neighbor_dist / 2.0)
        
        # Select most constrained points
        local_idx = np.argsort(constraint_score)[:n_local]
        
        # Compute full gradient but only update local points
        grad = _grad_only(pts, 80.0, threshold=0.12)
        
        # Update only local points with higher lr
        lr_local = 0.003
        pts[local_idx] += lr_local * grad[local_idx]
        
        # Keep points inside the unit disk
        norms = np.linalg.norm(pts, axis=1)
        too_far = norms > 0.999
        if np.any(too_far):
            pts[too_far] = pts[too_far] * (0.999 / norms[too_far])
        
        step += 1
        if step % 20 == 0:
            lr_local = max(0.0005, lr_local * 0.93)
    
    return best_pts, best_score


def _polish(pts, time_budget):
    """Final high-precision polish with tighter parameters, focusing on radius."""
    pts = pts.copy()
    best_pts = pts.copy()
    best_score = _compute_radius(pts)
    
    k = 100.0
    lr = 0.001
    patience = 0
    max_patience = 300
    
    t0 = time.time()
    step = 0
    while time.time() - t0 < time_budget:
        # Use only radius gradient for polish
        soft_val, grad = _objective_and_grad(pts, k, threshold=0.10)
        
        pts = pts + lr * grad
        
        norms = np.linalg.norm(pts, axis=1)
        too_far = norms > 0.999
        if np.any(too_far):
            pts[too_far] = pts[too_far] * (0.999 / norms[too_far])
        
        current_score = _compute_radius(pts)
        if current_score > best_score:
            best_score = current_score
            best_pts = pts.copy()
            patience = 0
        else:
            patience += 1
        
        step += 1
        if step % 50 == 0:
            lr = max(0.0001, lr * 0.9)
            k = min(500.0, k * 1.03)
        
        if patience > max_patience:
            break
    
    return best_pts, best_score


def _optimize(pts, time_budget, initial_lr=0.005, initial_k=30.0):
    """Optimize packing using gradient ascent on soft-min radius."""
    pts = pts.copy()
    best_pts = pts.copy()
    best_score = _compute_radius(pts)
    
    k = initial_k
    lr = initial_lr
    patience = 0
    max_patience = 500
    
    t0 = time.time()
    step = 0
    while time.time() - t0 < time_budget:
        combined_val, grad = _objective_and_grad(pts, k, threshold=0.15)
        
        # Gradient ascent
        pts = pts + lr * grad
        
        # Keep points inside the unit disk
        norms = np.linalg.norm(pts, axis=1)
        too_far = norms > 0.999
        if np.any(too_far):
            pts[too_far] = pts[too_far] * (0.999 / norms[too_far])
        
        # Compute actual score
        current_score = _compute_radius(pts)
        if current_score > best_score:
            best_score = current_score
            best_pts = pts.copy()
            patience = 0
        else:
            patience += 1
        
        # Adaptive learning rate and sharpness
        step += 1
        if step % 100 == 0:
            lr = max(0.0005, lr * 0.95)
            k = min(200.0, k * 1.05)
        
        if patience > max_patience:
            break
    
    return best_pts, best_score


def _validate_and_fix(pos):
    """Ensure positions are valid: finite, inside unit disk, no coincident points."""
    pos = np.array(pos, dtype=np.float64)
    if pos.shape != (N, 2):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = np.linspace(0.1, 0.9, N)
        pos = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    if not np.all(np.isfinite(pos)):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = np.linspace(0.1, 0.9, N)
        pos = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    norms = np.linalg.norm(pos, axis=1)
    mask = norms >= 1.0
    if np.any(mask):
        pos[mask] = pos[mask] * (0.999 / norms[mask])
    
    # Check for coincident points
    tree = cKDTree(pos)
    min_dist, _ = tree.query(pos, k=2)
    min_dist = min_dist[:, 1]
    if np.min(min_dist) < 1e-4:
        rng = np.random.default_rng(42)
        for i in range(N):
            if min_dist[i] < 1e-4:
                pos[i] += rng.standard_normal(2) * 1e-3
                n_i = np.linalg.norm(pos[i])
                if n_i > 0.999:
                    pos[i] = pos[i] * (0.999 / n_i)
    
    return pos


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 130s budget, leaving 20s margin
    
    # Step 1: Generate initial positions
    positions = _hex_init(N)
    
    # Step 2: Main optimization
    time_remaining = deadline - time.time()
    if time_remaining > 10.0:
        positions, _ = _optimize(positions, time_budget=time_remaining - 30.0)
    
    # Step 3: Tighten min constraint
    time_remaining = deadline - time.time()
    if time_remaining > 10.0:
        positions, _ = _tighten_min_constraint(positions, time_budget=time_remaining - 20.0)
    
    # Step 4: Local refinement
    time_remaining = deadline - time.time()
    if time_remaining > 8.0:
        positions, _ = _local_refine(positions, time_budget=time_remaining - 12.0)
    
    # Step 5: Polish pass
    time_remaining = deadline - time.time()
    if time_remaining > 5.0:
        positions, _ = _polish(positions, time_budget=time_remaining - 5.0)
    
    # Final validation
    positions = _validate_and_fix(positions)
    
    return positions
# EVOLVE-BLOCK-END
