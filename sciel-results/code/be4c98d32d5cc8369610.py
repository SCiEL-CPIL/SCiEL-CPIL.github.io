"""Improved packing: active-set projection + local refinement for 50 equal circles in unit disk."""
import numpy as np
import time

N = 50
DEADLINE = 115.0  # seconds; leave margin before 120s limit


def _score(pts):
    """Compute the packing radius r = min(min pairwise dist/2, min (1 - |c_i|))."""
    if len(pts) != N:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # Distance to boundary
    norms = np.sqrt(np.sum(pts * pts, axis=1))
    min_boundary = np.min(1.0 - norms)
    return min(min_pair, min_boundary)


def _project_and_separate(pts, r, max_iter=500, tol=1e-10):
    """
    Active-set style: iteratively project points onto the unit disk (with margin r)
    and separate overlapping pairs.
    Returns improved points and achieved radius.
    """
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    
    for it in range(max_iter):
        moved = False
        # Separate overlapping pairs
        for i in range(n):
            for j in range(i + 1, n):
                d = pts[j] - pts[i]
                dist = np.linalg.norm(d)
                if dist < 2.0 * r:
                    if dist < 1e-15:
                        # Perturb randomly
                        d = np.array([1.0, 0.0])
                        dist = 1.0
                    # Move both apart by half the deficit
                    deficit = (2.0 * r - dist) / 2.0
                    direction = d / dist
                    pts[i] -= direction * deficit
                    pts[j] += direction * deficit
                    moved = True
        
        # Project onto disk of radius (1 - r)
        for i in range(n):
            norm = np.linalg.norm(pts[i])
            max_r = 1.0 - r
            if norm > max_r:
                pts[i] = pts[i] * (max_r / norm)
                moved = True
        
        if not moved:
            break
    
    return pts


def _local_refine(pts, r, max_iter=2000, tol=1e-12):
    """
    Local refinement using a simple gradient-like approach on the min objective.
    Uses subgradient of the min function.
    """
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    lr = 0.01
    
    for it in range(max_iter):
        # Compute current radius
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        min_pair_idx = np.argmin(dists)
        i, j = np.unravel_index(min_pair_idx, (n, n))
        
        norms = np.sqrt(np.sum(pts * pts, axis=1))
        min_boundary = np.min(1.0 - norms)
        min_boundary_idx = np.argmin(1.0 - norms)
        
        if min_pair_dist / 2.0 < min_boundary:
            # Pairwise constraint is active
            active_i, active_j = i, j
            d = pts[j] - pts[i]
            dist = np.linalg.norm(d)
            if dist < 1e-15:
                continue
            direction = d / dist
            # Subgradient: push i away from j, j away from i
            grad_i = -direction
            grad_j = direction
            # Move
            pts[active_i] += lr * grad_i
            pts[active_j] += lr * grad_j
        else:
            # Boundary constraint is active
            active_i = min_boundary_idx
            norm = np.linalg.norm(pts[active_i])
            if norm < 1e-15:
                continue
            direction = pts[active_i] / norm
            # Push inward
            grad_i = -direction
            pts[active_i] += lr * grad_i
        
        # Re-project onto disk
        for i in range(n):
            norm = np.linalg.norm(pts[i])
            max_r = 1.0 - r
            if norm > max_r:
                pts[i] = pts[i] * (max_r / norm)
        
        # Check convergence
        if it % 100 == 0:
            cur_r = _score(pts)
            if cur_r <= r + tol:
                break
            lr *= 0.95
    
    return pts


def _hexagonal_init(r):
    """Generate a hexagonal lattice packing for initial guess."""
    spacing = 2.0 * r
    pts = []
    # Generate points in a hexagonal pattern
    max_k = int(1.0 / spacing) + 3
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.sqrt(x*x + y*y) <= 1.0 - r - 1e-9:
                pts.append([x, y])
    
    pts = np.array(pts)
    if len(pts) < N:
        # If not enough, reduce spacing
        return None
    # Sort by distance from center, take N closest
    dists = np.sqrt(np.sum(pts * pts, axis=1))
    order = np.argsort(dists)
    return pts[order[:N]]


def construct():
    start_time = time.time()
    
    # Start with a reasonable initial radius
    # Known best is ~0.1258, start slightly lower
    r_init = 0.11
    
    # Try hexagonal initialization
    pts = _hexagonal_init(r_init)
    if pts is None:
        # Fallback: random initialization
        pts = np.random.randn(N, 2) * 0.5
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True) * 0.5
    
    # Project initial points
    for i in range(N):
        norm = np.linalg.norm(pts[i])
        if norm > 1.0 - r_init:
            pts[i] = pts[i] * ((1.0 - r_init) / norm)
    
    # Active-set separation
    pts = _project_and_separate(pts, r_init, max_iter=200)
    
    # Local refinement
    pts = _local_refine(pts, r_init, max_iter=500)
    
    best_r = _score(pts)
    best_pts = pts.copy()
    
    # Try to increase radius iteratively
    r = r_init
    for trial in range(20):
        if time.time() - start_time > DEADLINE:
            break
        
        # Try a slightly larger radius
        r_new = r + 0.001
        pts_trial = best_pts.copy()
        pts_trial = _project_and_separate(pts_trial, r_new, max_iter=300)
        pts_trial = _local_refine(pts_trial, r_new, max_iter=300)
        r_achieved = _score(pts_trial)
        
        if r_achieved > best_r:
            best_r = r_achieved
            best_pts = pts_trial.copy()
            r = r_new
        else:
            # Try smaller step
            r_new = r + 0.0005
            pts_trial = best_pts.copy()
            pts_trial = _project_and_separate(pts_trial, r_new, max_iter=300)
            pts_trial = _local_refine(pts_trial, r_new, max_iter=300)
            r_achieved = _score(pts_trial)
            if r_achieved > best_r:
                best_r = r_achieved
                best_pts = pts_trial.copy()
                r = r_new
            else:
                break
    
    # Final polish: try to maximize radius from current configuration
    if time.time() - start_time < DEADLINE - 5:
        # Binary search for maximum feasible radius
        lo, hi = best_r, 0.14
        for _ in range(10):
            if time.time() - start_time > DEADLINE - 2:
                break
            mid = (lo + hi) / 2
            pts_trial = best_pts.copy()
            pts_trial = _project_and_separate(pts_trial, mid, max_iter=500)
            r_achieved = _score(pts_trial)
            if r_achieved >= mid - 1e-9:
                lo = mid
                best_pts = pts_trial.copy()
                best_r = max(best_r, r_achieved)
            else:
                hi = mid
    
    # Final validation and projection
    for i in range(N):
        norm = np.linalg.norm(best_pts[i])
        if norm > 1.0 - best_r:
            best_pts[i] = best_pts[i] * ((1.0 - best_r) / norm)
    
    return best_pts
