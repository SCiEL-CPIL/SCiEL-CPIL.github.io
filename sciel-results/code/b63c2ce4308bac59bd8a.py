"""
Packing 50 equal circles in a unit disk.
Combines ring-structured initialization with multi-stage smooth optimization
using soft-min log-sum-exp objective and a final binary-search-style polishing phase.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 135.0  # safety margin before 150s kill


def _pairwise_distances(pts):
    """Compute pairwise distances between all points."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def _score(pts):
    """Compute the packing score: minimum of half pairwise distances and boundary distances."""
    pts = np.asarray(pts)
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    min_boundary = np.min(1.0 - np.sqrt(np.sum(pts ** 2, axis=1)))
    return max(min(min_pair, min_boundary), 0.0)


def _softmin_logsumexp(values, beta):
    """Soft-min via log-sum-exp for numerical stability."""
    if len(values) == 0:
        return 0.0
    neg = -values * beta
    neg_max = np.max(neg)
    if not np.isfinite(neg_max):
        return 0.0
    exp_sum = np.sum(np.exp(neg - neg_max))
    if exp_sum == 0:
        return 0.0
    return -(neg_max + np.log(exp_sum)) / beta


def _smooth_objective(x, beta=50.0):
    """
    Smooth objective: minimize -softmin of all constraint margins.
    The softmin approximates the minimum, so minimizing -softmin maximizes the minimum.
    """
    pts = x.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    boundary = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
    pair_half = d / 2.0
    
    # Get all finite values
    all_vals = np.concatenate([pair_half.ravel(), boundary])
    all_vals = all_vals[np.isfinite(all_vals)]
    
    if len(all_vals) == 0:
        return 1e10
    
    soft_min = _softmin_logsumexp(all_vals, beta)
    return -soft_min


def _ring_initialization(ring_radii, ring_counts, angles_offset=None):
    """Generate points from a ring structure."""
    centers = []
    for idx, (count, radius) in enumerate(zip(ring_counts, ring_radii)):
        if count == 1 and abs(radius) < 1e-10:
            centers.append(np.array([0.0, 0.0]))
        else:
            if angles_offset is not None and idx < len(angles_offset):
                offset = angles_offset[idx]
            else:
                offset = 0.0
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + offset
            xs = radius * np.cos(angles)
            ys = radius * np.sin(angles)
            for i in range(count):
                centers.append(np.array([xs[i], ys[i]]))
    
    pts = np.array(centers[:N])
    # Project inside unit circle
    norms = np.sqrt(np.sum(pts ** 2, axis=1))
    for i in range(len(pts)):
        if norms[i] > 0.999:
            pts[i] *= 0.999 / norms[i]
    return pts


def _hexagonal_init(n):
    """
    Initialize with a hexagonal lattice structure.
    This is a strong starting point for circle packing.
    """
    a = 1.0
    sqrt3 = np.sqrt(3)
    
    def hex_to_cart(q, r):
        x = a * 1.5 * q
        y = a * sqrt3 * (r + q / 2.0)
        return np.array([x, y])
    
    max_ring = 6
    candidates = []
    for q in range(-max_ring, max_ring + 1):
        for r in range(-max_ring, max_ring + 1):
            candidates.append((q, r))
            
    cart_points = []
    for q, r in candidates:
        p = hex_to_cart(q, r)
        d = np.linalg.norm(p)
        cart_points.append((d, p))
        
    cart_points.sort(key=lambda x: x[0])
    selected_points = [p for d, p in cart_points[:n]]
    points = np.array(selected_points)
    
    max_r = np.max(np.linalg.norm(points, axis=1))
    if max_r > 0:
        scale = 0.85 / max_r
        points = points * scale
        
    return points


def _optimize_from_init(pts, deadline, max_beta_stages=5):
    """Multi-stage annealing optimization from initial points."""
    current_pts = pts.copy()
    current_score = _score(current_pts)
    
    # Progressive beta values for smooth optimization
    betas = [5.0, 15.0, 40.0, 100.0, 250.0]
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
        x0 = current_pts.ravel()
        try:
            result = minimize(
                _smooth_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-11, 'maxfun': 5000}
            )
            new_pts = result.x.reshape(N, 2)
            if not np.all(np.isfinite(new_pts)):
                continue
            new_score = _score(new_pts)
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
        except Exception:
            pass
    
    # High-precision polish with larger betas
    for beta in [500.0, 1000.0, 2000.0, 5000.0]:
        if time.time() > deadline - 5:
            break
        x0 = current_pts.ravel()
        try:
            result = minimize(
                _smooth_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxfun': 20000}
            )
            new_pts = result.x.reshape(N, 2)
            if not np.all(np.isfinite(new_pts)):
                continue
            new_score = _score(new_pts)
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
        except Exception:
            pass
    
    return current_pts, current_score


def _binary_search_polish(pts, r_target, deadline, max_iters=500):
    """
    Polish by iteratively pushing points to achieve target radius r_target.
    This is a direct constraint satisfaction approach.
    """
    pts = pts.copy()
    
    for _ in range(max_iters):
        if time.time() > deadline - 3:
            break
        
        # Compute current margins
        d = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        pair_margins = d / 2.0
        boundary_margins = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
        
        # Find the most violated constraint
        min_pair_margin = np.min(pair_margins)
        min_boundary_margin = np.min(boundary_margins)
        
        if min_pair_margin < min_boundary_margin - 1e-14:
            # Pair constraint is binding
            i, j = np.unravel_index(np.argmin(pair_margins), pair_margins.shape)
            if i == j:
                break
            di = pts[i] - pts[j]
            dist = np.linalg.norm(di)
            if dist < 1e-15:
                break
            di /= dist
            push = (2 * r_target - dist) / 2.0
            if push > 1e-15:
                pts[i] += push * di
                pts[j] -= push * di
        else:
            # Boundary constraint is binding
            i = int(np.argmin(boundary_margins))
            pi = pts[i]
            dist = np.linalg.norm(pi)
            if dist < 1e-15:
                pts[i] += 1e-4 * np.array([1.0, 0.0])
            else:
                di = pi / dist
                push = (1.0 - r_target - dist)
                if push > 1e-15:
                    pts[i] += push * di
        
        # Check convergence
        d = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        r = min(np.min(d) / 2.0, np.min(1.0 - np.sqrt(np.sum(pts ** 2, axis=1))))
        if r >= r_target - 1e-12:
            break
    
    return pts


def _de_ring_optimize(deadline):
    """
    Differential evolution over ring parameters to find good ring configurations.
    """
    ring_counts_options = [
        [1, 6, 12, 18, 13],
        [1, 6, 12, 14, 17],
        [1, 6, 12, 16, 15],
        [1, 6, 12, 15, 16],
        [1, 6, 10, 14, 19],
        [1, 6, 10, 12, 21],
    ]
    
    best_score = 0.0
    best_params = None
    best_counts = None
    
    for counts in ring_counts_options:
        if time.time() > deadline - 20:
            break
        
        n_rings = len(counts)
        
        # Define parameter space: radii and angle offsets
        # Radii: 0 to 0.9, Angles: 0 to 2*pi
        bounds = []
        for i in range(n_rings):
            if i == 0:
                bounds.append((0.0, 0.0))  # center
            else:
                bounds.append((0.0, 0.9))
        for i in range(n_rings):
            bounds.append((0.0, 2 * np.pi))
        
        # Use scipy's differential evolution
        def neg_score(params):
            radii = params[:n_rings]
            angles = params[n_rings:]
            pts = _ring_initialization(radii, counts, angles)
            return -_score(pts)
        
        x0 = np.zeros(len(bounds))
        # Initialize with reasonable values
        for i in range(1, n_rings):
            x0[i] = 0.3 + 0.5 * i / n_rings
        
        try:
            result = minimize(
                neg_score,
                x0,
                bounds=bounds,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            params_opt = result.x
            score_opt = -result.fun
            if score_opt > best_score:
                best_score = score_opt
                best_params = params_opt
                best_counts = counts
        except Exception:
            pass
    
    if best_params is not None and best_counts is not None:
        n_rings = len(best_counts)
        radii = best_params[:n_rings]
        angles = best_params[n_rings:]
        return _ring_initialization(radii, best_counts, angles), best_score
    
    return None, 0.0


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE
    
    # Start with hexagonal initialization
    P_best = _hexagonal_init(N)
    score_best = _score(P_best)
    
    # Try differential evolution over ring structures
    P_ring, score_ring = _de_ring_optimize(deadline - 10)
    if P_ring is not None and score_ring > score_best:
        P_best = P_ring
        score_best = score_ring
    
    # Multi-stage optimization from best initial point
    P_best, score_best = _optimize_from_init(P_best, deadline)
    
    # Binary search polish to push toward higher radius
    r_current = score_best
    r_target = r_current * 1.02
    P_polished = _binary_search_polish(P_best, r_target, deadline - 5)
    score_polished = _score(P_polished)
    if score_polished > score_best:
        P_best = P_polished
        score_best = score_polished
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        P_final, score_final = _optimize_from_init(P_best, deadline - 2)
        if score_final > score_best:
            P_best = P_final
            score_best = score_final
    
    # Ensure validity: clamp points that might be slightly outside
    r_final = _score(P_best)
    if r_final > 0:
        r2 = np.hypot(P_best[:, 0], P_best[:, 1])
        for i in range(N):
            if r2[i] > 1 - r_final + 1e-10:
                scale = (1 - r_final) / r2[i]
                P_best[i] *= scale
    
    return P_best.reshape(N, 2)
