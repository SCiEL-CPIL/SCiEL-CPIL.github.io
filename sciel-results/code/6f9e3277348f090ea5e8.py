"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import time

N = 300

# EVOLVE-BLOCK-START

def _initial_hex():
    """Generate a hexagonal lattice packing inside the unit disk."""
    # Start with a reasonable spacing
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    
    while True:
        k = int(1.0 / spacing) + 2
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        # Keep points that can fit a circle of radius spacing/2
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        radius = radius[mask]
        
        if len(points) >= N:
            # Take the N closest to center for symmetry
            idx = np.argsort(radius, kind='stable')[:N]
            return points[idx]
        spacing *= 0.98


def _compute_score(pts):
    """Compute the packing score: min of pairwise distances/2 and distance to boundary."""
    n = len(pts)
    # Distance to boundary
    dists_boundary = 1.0 - np.linalg.norm(pts, axis=1)
    min_boundary = np.min(dists_boundary)
    
    # Pairwise distances using KDTree for efficiency
    tree = cKDTree(pts)
    # Query k nearest neighbors (including self, so k=2)
    dists, _ = tree.query(pts, k=2)
    # dists[:, 0] is 0 (self), dists[:, 1] is nearest neighbor
    min_pair = np.min(dists[:, 1]) / 2.0
    
    return min(min_boundary, min_pair)


def _soft_min_score(pts, beta=100.0):
    """Smooth surrogate using log-sum-exp for soft-min."""
    n = len(pts)
    
    # Boundary constraint
    dists_boundary = 1.0 - np.linalg.norm(pts, axis=1)
    
    # Pairwise distances
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    pair_dists = dists[:, 1] / 2.0
    
    # Combine all constraints
    all_dists = np.concatenate([dists_boundary, pair_dists])
    
    # Soft-min via log-sum-exp
    m = np.min(all_dists)
    soft_min = m + np.log(np.mean(np.exp(all_dists - m))) / beta
    
    return -soft_min  # We maximize, so negate


def _soft_min_grad(pts, beta=100.0):
    """Gradient of the soft-min score."""
    n = len(pts)
    
    # Boundary distances
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    # Avoid division by zero
    norms_safe = np.where(norms == 0, 1e-10, norms)
    dists_boundary = 1.0 - norms[:, 0]
    
    # Pairwise: nearest neighbor distances
    tree = cKDTree(pts)
    dists, idx = tree.query(pts, k=2)
    pair_dists = dists[:, 1] / 2.0
    nearest_idx = idx[:, 1]
    
    all_dists = np.concatenate([dists_boundary, pair_dists])
    m = np.min(all_dists)
    
    # Weights for soft-min
    exp_vals = np.exp(all_dists - m)
    weights = exp_vals / (np.sum(exp_vals) * beta)
    
    grad = np.zeros_like(pts)
    
    # Boundary gradient contribution
    for i in range(n):
        w = weights[i]
        # d/dx_i (1 - |x_i|) = -x_i / |x_i|
        if norms[i, 0] > 1e-10:
            grad[i] -= w * pts[i] / norms[i, 0]
    
    # Pairwise gradient contribution
    for i in range(n):
        w = weights[n + i]
        j = nearest_idx[i]
        diff = pts[i] - pts[j]
        dist = np.linalg.norm(diff)
        if dist > 1e-10:
            # d/dx_i (|x_i - x_j| / 2) = (x_i - x_j) / (2 * |x_i - x_j|)
            grad[i] -= w * diff / (2.0 * dist)
            # d/dx_j (|x_i - x_j| / 2) = -(x_i - x_j) / (2 * |x_i - x_j|)
            grad[j] += w * diff / (2.0 * dist)
    
    return -grad  # Negate because we're maximizing


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Generate initial hexagonal packing
    pts = _initial_hex()
    current_score = _compute_score(pts)
    best_pts = pts.copy()
    best_score = current_score
    
    # L-BFGS-B optimization with soft-min surrogate
    beta = 50.0
    
    # First pass: coarser soft-min for broader exploration
    result = minimize(
        _soft_min_score,
        pts,
        args=(beta,),
        method='L-BFGS-B',
        options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
    )
    
    if time.time() > deadline:
        return best_pts
    
    if result.fun < _soft_min_score(pts, beta):
        candidate = np.clip(result.x, -1.0, 1.0)
        # Ensure inside unit circle
        norms = np.linalg.norm(candidate, axis=1)
        mask = norms <= 1.0
        if np.all(mask):
            cand_score = _compute_score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_pts = candidate.copy()
                pts = candidate.copy()
    
    # Second pass: finer soft-min for polishing
    beta2 = 200.0
    if time.time() < deadline - 10.0:
        result2 = minimize(
            _soft_min_score,
            pts,
            args=(beta2,),
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        
        if result2.fun < _soft_min_score(pts, beta2):
            candidate = np.clip(result2.x, -1.0, 1.0)
            norms = np.linalg.norm(candidate, axis=1)
            mask = norms <= 1.0
            if np.all(mask):
                cand_score = _compute_score(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = candidate.copy()
                    pts = candidate.copy()
    
    # Final polish: direct local search on the nonsmooth objective
    if time.time() < deadline - 5.0:
        # Simple iterative local refinement
        step_size = 0.001
        for _ in range(50):
            if time.time() > deadline - 2.0:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 2.0:
                    break
                for dx in [-step_size, 0, step_size]:
                    for dy in [-step_size, 0, step_size]:
                        if dx == 0 and dy == 0:
                            continue
                        new_pts = pts.copy()
                        new_pts[i, 0] += dx
                        new_pts[i, 1] += dy
                        
                        # Check if inside unit circle
                        if np.linalg.norm(new_pts[i]) > 1.0:
                            continue
                        
                        new_score = _compute_score(new_pts)
                        if new_score > best_score + 1e-10:
                            best_score = new_score
                            best_pts = new_pts.copy()
                            pts = new_pts.copy()
                            improved = True
                            break
                    if improved:
                        break
            if not improved:
                step_size *= 0.5
                if step_size < 1e-6:
                    break
    
    # Final validation
    final_score = _compute_score(best_pts)
    if final_score > 0:
        return best_pts
    else:
        return _initial_hex()

# EVOLVE-BLOCK-END
