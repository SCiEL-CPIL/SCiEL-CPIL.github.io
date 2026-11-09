import numpy as np
import time
from scipy.optimize import minimize

def get_points(deltas):
    """Generate 20 points on unit circle with cyclic order preserved.
    deltas: 19 values added to angles 1..19 (angle 0 is fixed at 0)."""
    n = 20
    base_angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    thetas = base_angles.copy()
    if len(deltas) > 0:
        thetas[1:] += deltas
    # Wrap to [0, 2*pi) and sort to maintain cyclic order
    thetas = thetas % (2*np.pi)
    thetas = np.sort(thetas)
    x = np.cos(thetas)
    y = np.sin(thetas)
    return np.column_stack((x, y))

def normalize_polygon(points):
    """Normalize: subtract centroid, divide by mean distance to centroid."""
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    dists = np.linalg.norm(centered, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return None
    return centered / mean_dist

def check_convexity_fast(points):
    """Fast check if points form a strictly convex polygon (vectorized)."""
    n = len(points)
    if n < 3:
        return False
    # Vectorized cross products
    p0 = points
    p1 = np.roll(points, -1, axis=0)
    p2 = np.roll(points, -2, axis=0)
    v1 = p1 - p0
    v2 = p2 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    
    # All cross products must be nonzero and same sign
    if np.any(np.abs(cross) < 1e-12):
        return False
    if len(np.unique(np.sign(cross))) != 1:
        return False
    # Check no repeated points (use min pairwise distance)
    from scipy.spatial.distance import pdist
    min_dist = pdist(points, metric='euclidean').min()
    if min_dist < 1e-10:
        return False
    return True

def vertex_score(v_idx, points):
    """Compute the score for a single vertex."""
    dists = np.linalg.norm(points - points[v_idx], axis=1)
    dists[v_idx] = np.inf
    sorted_dists = np.sort(dists)
    
    if sorted_dists[0] < 0.05:
        return 0.0
    
    gaps = sorted_dists[1:] - sorted_dists[:-1]
    k = np.argmin(gaps)
    D = (sorted_dists[k] + sorted_dists[k+1]) / 2.0
    
    if D < 0.1:
        return 0.0
    
    if k == 0:
        d_sel = sorted_dists[0:4]
    elif k == 17:
        d_sel = sorted_dists[15:19]
    else:
        d_sel = sorted_dists[k-1:k+3]
    
    ratios = np.maximum(d_sel / D, D / d_sel)
    raw = np.mean(ratios)
    score = min(1.0, 1.0 / raw)
    return score

def compute_all_scores(points):
    """Compute all 20 vertex scores."""
    n = len(points)
    scores = np.array([vertex_score(i, points) for i in range(n)])
    return scores

def soft_min(scores, beta=50.0):
    """Smooth approximation of min using log-sum-exp."""
    # soft_min ≈ -log(sum(exp(-beta*s)))/beta
    shifted = -beta * scores
    max_shifted = np.max(shifted)
    result = -max_shifted/beta - np.log(np.sum(np.exp(shifted - max_shifted)))/beta
    return result

def smooth_objective(deltas, beta=50.0):
    """Smooth objective: negative soft-min of vertex scores."""
    points = get_points(deltas)
    normalized = normalize_polygon(points)
    if normalized is None:
        return 0.0
    if not check_convexity_fast(normalized):
        return 0.0
    scores = compute_all_scores(normalized)
    if np.any(scores == 0):
        return 0.0
    return -soft_min(scores, beta)

def exact_objective(deltas):
    """Exact objective: negative of the minimum vertex score."""
    points = get_points(deltas)
    normalized = normalize_polygon(points)
    if normalized is None:
        return 0.0
    if not check_convexity_fast(normalized):
        return 0.0
    scores = compute_all_scores(normalized)
    return -np.min(scores)

def reconstruct_deltas(points):
    """Reconstruct angle deltas from point coordinates."""
    n = 20
    thetas = np.arctan2(points[:, 1], points[:, 0])
    thetas = thetas % (2*np.pi)
    thetas = np.sort(thetas)
    base_angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    # Match each theta to its nearest base angle
    deltas = np.zeros(19)
    for i in range(1, n):
        # Find the base angle closest to thetas[i]
        diffs = (thetas[i] - base_angles) % (2*np.pi)
        diffs = np.where(diffs > np.pi, diffs - 2*np.pi, diffs)
        best_base = np.argmin(np.abs(diffs))
        deltas[i-1] = (thetas[i] - base_angles[best_base]) % (2*np.pi)
        if deltas[i-1] > np.pi:
            deltas[i-1] -= 2*np.pi
    return deltas

def equidistant_points_polygon20() -> np.ndarray:
    """Main function: optimize the 20-gon vertex positions."""
    start_time = time.time()
    deadline = start_time + 615  # safety margin before 630s limit
    
    n = 20
    x0 = np.zeros(19)
    best_result = get_points(x0)
    best_score = -exact_objective(x0)
    
    rng = np.random.default_rng(12345)
    
    # Phase 1: Coarse search with smooth objective (lower beta for smoother landscape)
    beta_coarse = 20.0
    for _ in range(8):
        if time.time() > deadline:
            break
        res = minimize(
            lambda d: smooth_objective(d, beta_coarse),
            x0,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-7, 'maxfun': 30000}
        )
        if res.x is not None:
            s = -exact_objective(res.x)
            if s > best_score:
                best_score = s
                best_result = get_points(res.x)
        x0 = res.x
    
    # Phase 2: Medium refinement with higher beta
    beta_med = 50.0
    for _ in range(6):
        if time.time() > deadline:
            break
        current_deltas = reconstruct_deltas(best_result)
        res = minimize(
            lambda d: smooth_objective(d, beta_med),
            current_deltas,
            method='L-BFGS-B',
            options={'maxiter': 3000, 'ftol': 1e-11, 'gtol': 1e-8, 'maxfun': 40000}
        )
        if res.x is not None:
            s = -exact_objective(res.x)
            if s > best_score:
                best_score = s
                best_result = get_points(res.x)
        current_deltas = res.x
    
    # Phase 3: Random restarts with smooth objective
    scales = [0.3, 0.2, 0.1, 0.05]
    for scale in scales:
        if time.time() > deadline:
            break
        for _ in range(4):
            if time.time() > deadline:
                break
            x_init = rng.normal(0, scale, size=19)
            try:
                res = minimize(
                    lambda d: smooth_objective(d, 30.0),
                    x_init,
                    method='L-BFGS-B',
                    options={'maxiter': 1500, 'ftol': 1e-10, 'gtol': 1e-7, 'maxfun': 25000}
                )
                if res.x is not None:
                    s = -exact_objective(res.x)
                    if s > best_score:
                        best_score = s
                        best_result = get_points(res.x)
            except Exception:
                pass
    
    # Phase 4: Final high-precision polish with exact objective
    if time.time() < deadline - 30:
        try:
            current_deltas = reconstruct_deltas(best_result)
            res = minimize(
                exact_objective,
                current_deltas,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-13, 'gtol': 1e-10, 'maxfun': 60000}
            )
            if res.x is not None:
                s = -exact_objective(res.x)
                if s > best_score:
                    best_score = s
                    best_result = get_points(res.x)
        except Exception:
            pass
    
    # Phase 5: One more polish with even tighter tolerance
    if time.time() < deadline - 15:
        try:
            current_deltas = reconstruct_deltas(best_result)
            res = minimize(
                exact_objective,
                current_deltas,
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 40000}
            )
            if res.x is not None:
                s = -exact_objective(res.x)
                if s > best_score:
                    best_score = s
                    best_result = get_points(res.x)
        except Exception:
            pass
    
    # Ensure the result is normalized and convex
    normalized = normalize_polygon(best_result)
    if normalized is not None and check_convexity_fast(normalized):
        return normalized
    else:
        # Fallback: return the regular polygon
        regular = np.column_stack((np.cos(np.linspace(0, 2*np.pi, n, endpoint=False)),
                                   np.sin(np.linspace(0, 2*np.pi, n, endpoint=False))))
        return normalize_polygon(regular)
