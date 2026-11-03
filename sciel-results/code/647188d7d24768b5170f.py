"""Pack 50 equal circles in unit disk — ring seed + smooth-min L-BFGS-B refinement."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
T_START = time.time()
TIME_LIMIT = 145.0  # safety margin before 150s kill

def _time_left():
    return TIME_LIMIT - (time.time() - T_START)

def _score(pts):
    """Return the achievable common radius for a placement."""
    if not np.all(np.isfinite(pts)):
        return -1.0
    # boundary constraint
    norms = np.linalg.norm(pts, axis=1)
    r_bound = 1.0 - norms
    if np.any(r_bound < -1e-12):
        return -1.0
    # pairwise
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    r_pair = dist.min(axis=1) / 2.0
    return float(min(r_bound.min(), r_pair.min()))

def _build_ring_seed():
    """Build the known best-known packing structure: center + concentric rings."""
    # Common good structure: 1, 6, 12, 18, 13 = 50
    rings = [(6, 0.22), (12, 0.44), (18, 0.66), (13, 0.87)]
    pts = [np.array([0.0, 0.0])]
    for count, radius in rings:
        for i in range(count):
            angle = 2.0 * np.pi * i / count
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            pts.append(np.array([x, y]))
    return np.array(pts)

def _softmin_logsumexp(values, beta):
    """Soft-min via log-sum-exp."""
    neg = -values * beta
    neg_max = np.max(neg)
    exp_sum = np.sum(np.exp(neg - neg_max))
    return -(neg_max + np.log(exp_sum)) / beta

def _smooth_objective(x, beta=50.0):
    """Smooth objective: minimize -softmin of all constraint margins."""
    pts = x.reshape(N, 2)
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    boundary = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
    pair_half = dist / 2.0
    all_vals = np.concatenate([pair_half.ravel(), boundary])
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals) == 0:
        return 0.0
    soft_min = _softmin_logsumexp(all_vals, beta)
    return -soft_min

def _optimize_smooth(pts, deadline):
    """Multi-stage annealing optimization using smooth soft-min objective."""
    current_pts = pts.copy()
    current_score = _score(current_pts)
    
    betas = [5.0, 15.0, 40.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 5000.0]
    
    for beta in betas:
        if _time_left() < 5.0:
            break
        x0 = current_pts.ravel()
        try:
            result = minimize(
                _smooth_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 10000}
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

def _gradient_polish(pts, iterations=500):
    """Gradient-based local optimization using subgradient of min distance."""
    pts = pts.copy()
    n = len(pts)
    best_score = _score(pts)
    best_pts = pts.copy()
    
    lr = 0.001
    for it in range(iterations):
        if _time_left() < 3.0:
            break
        
        # Compute all pairwise distances
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Find the minimum distance pair
        min_idx = np.unravel_index(np.argmin(dist), dist.shape)
        i, j = min_idx
        min_dist = dist[i, j]
        
        # Also check boundary
        norms = np.linalg.norm(pts, axis=1)
        min_norm = norms.min()
        boundary_idx = np.argmin(norms)
        
        # Determine which constraint is binding
        r_candidate = min(min_dist / 2.0, 1.0 - min_norm)
        
        if min_dist / 2.0 <= 1.0 - min_norm:
            # Pairwise constraint binding: push i and j apart
            if min_dist < 1e-10:
                direction = np.array([1.0, 0.0])
            else:
                direction = (pts[i] - pts[j]) / min_dist
            grad = np.zeros_like(pts)
            grad[i] = direction * lr
            grad[j] = -direction * lr
        else:
            # Boundary constraint binding: push inward
            if min_norm < 1e-10:
                direction = np.array([1.0, 0.0])
            else:
                direction = pts[boundary_idx] / min_norm
            grad = np.zeros_like(pts)
            grad[boundary_idx] = -direction * lr
        
        pts = pts + grad
        
        # Project back: ensure inside disk
        norms = np.linalg.norm(pts, axis=1)
        for k in range(n):
            if norms[k] > 0.999:
                pts[k] = pts[k] * (0.999 / norms[k])
        
        new_score = _score(pts)
        if new_score > best_score:
            best_score = new_score
            best_pts = pts.copy()
        
        # Decay learning rate
        lr *= 0.999
    
    return best_pts, best_score

def construct():
    """Construct 50 circle centers in unit disk maximizing common radius."""
    # Build initial seed from ring structure
    pts = _build_ring_seed()
    
    # Verify initial validity
    init_score = _score(pts)
    if init_score <= 0:
        # Fallback: simple ring placement with safer radii
        pts = np.zeros((N, 2))
        ring_configs = [(6, 0.20), (12, 0.40), (18, 0.60), (13, 0.80)]
        idx = 1
        for count, radius in ring_configs:
            for i in range(count):
                angle = 2.0 * np.pi * i / count
                pts[idx, 0] = radius * np.cos(angle)
                pts[idx, 1] = radius * np.sin(angle)
                idx += 1
        init_score = _score(pts)
    
    best_pts = pts.copy()
    best_score = init_score
    
    # Phase 1: Smooth-min L-BFGS-B optimization (main refinement)
    if _time_left() > 10.0:
        smooth_pts, smooth_score = _optimize_smooth(pts, TIME_LIMIT)
        if smooth_score > best_score:
            best_pts = smooth_pts
            best_score = smooth_score
    
    # Phase 2: Gradient polish for final tightening
    if _time_left() > 5.0:
        for _ in range(3):
            if _time_left() < 3.0:
                break
            pol_pts, pol_score = _gradient_polish(best_pts, iterations=300)
            if pol_score > best_score:
                best_pts = pol_pts
                best_score = pol_score
    
    # Final validation
    final_score = _score(best_pts)
    if final_score <= 0:
        # Absolute fallback: very conservative hexagonal packing
        spacing = 0.2
        points = []
        for j in range(-5, 6):
            for i in range(-5, 6):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3) / 2
                if x*x + y*y <= (1 - spacing/2)**2:
                    points.append([x, y])
        points = np.array(points[:N])
        if len(points) < N:
            extra = N - len(points)
            extra_pts = np.zeros((extra, 2))
            for k in range(extra):
                extra_pts[k] = [0.01 * (k+1), 0.0]
            points = np.vstack([points, extra_pts])
        return points
    
    return best_pts
