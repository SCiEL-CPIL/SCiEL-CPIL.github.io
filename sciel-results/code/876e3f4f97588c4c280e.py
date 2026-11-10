# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 585  # wall-clock deadline with safety margin
_BEST_POINTS = None
_BEST_SCORE = 0.0  # 1/(1+C), higher is better (fewer convex 7-gons)


def _check_valid(points: np.ndarray) -> bool:
    """Validate according to the task contract."""
    if points.shape != (33, 2):
        return False
    if not np.all(np.isfinite(points)):
        return False
    if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
        return False
    n = 33
    diff = points[:, None, :] - points[None, :, :]
    sq_dists = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(sq_dists, 1.0)
    if np.any(sq_dists < 1e-12):
        return False
    # Collinearity check
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            for k in range(j + 1, n):
                xk, yk = points[k]
                cross = xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj)
                if abs(cross) < 1e-9:
                    return False
    return True


def _count_convex_7(points: np.ndarray) -> int:
    """
    Count number of 7-point subsets in convex position using DP over convex chains.
    Optimized with vectorized cross product computation.
    """
    n = 33
    order = np.lexsort((points[:, 1], points[:, 0]))
    pts = points[order]
    
    total = 0
    
    for i in range(n):
        later = list(range(i + 1, n))
        if len(later) < 6:
            continue
        
        pi = pts[i]
        
        # Compute angles and sort
        angles = np.arctan2(pts[later, 1] - pi[1], pts[later, 0] - pi[0])
        later_sorted = [later[j] for j in np.argsort(angles)]
        m = len(later_sorted)
        
        # Extract sorted points
        sorted_pts = pts[later_sorted]
        
        # Cross with pivot: (b-a) x (c-b) where a=pi, b=sorted_pts[j], c=sorted_pts[k]
        dp2 = np.zeros((m, m), dtype=np.int64)
        if m >= 2:
            js = np.arange(m)[:, None]
            ks = np.arange(m)[None, :]
            mask = js < ks
            
            bx = sorted_pts[js, 0]
            by = sorted_pts[js, 1]
            cx = sorted_pts[ks, 0]
            cy = sorted_pts[ks, 1]
            
            cross_vals = (bx - pi[0]) * (cy - by) - (by - pi[1]) * (cx - bx)
            dp2 = np.where(mask & (cross_vals > 1e-9), 1, 0).astype(np.int64)
        
        dp = [None, None, dp2]
        
        for l in range(3, 7):
            dpl = np.zeros((m, m), dtype=np.int64)
            prev_dp = dp[l - 1]
            
            if m < l:
                dp.append(dpl)
                continue
            
            for j in range(l - 2, m):
                for k in range(j + 1, m):
                    if j == 0:
                        dpl[j][k] = 0
                        continue
                    ixs = sorted_pts[:j, 0]
                    iys = sorted_pts[:j, 1]
                    jx, jy = sorted_pts[j]
                    kx, ky = sorted_pts[k]
                    
                    cross_vals = (jx - ixs) * (ky - jy) - (jy - iys) * (kx - jx)
                    valid_i = cross_vals > 1e-9
                    s = np.sum(prev_dp[:j, j] * valid_i.astype(np.int64))
                    dpl[j][k] = s
            
            dp.append(dpl)
        
        dp6 = dp[6]
        
        for j in range(4, m):
            for k in range(j + 1, m):
                if dp6[j][k] == 0:
                    continue
                a = sorted_pts[j]
                b = sorted_pts[k]
                c = pi
                cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
                if cross > 1e-9:
                    total += dp6[j][k]
    
    return total


def _evaluate(points: np.ndarray):
    """Return (valid, score) where score = 1/(1+C)."""
    if not _check_valid(points):
        return False, 0.0
    C = _count_convex_7(points)
    return True, 1.0 / (1.0 + C)


def _generate_tight_clusters(seed: int) -> np.ndarray:
    """Generate points in very tight clusters to minimize convex 7-gons."""
    rng = np.random.RandomState(seed)
    n_clusters = rng.randint(3, 5)
    centers = rng.rand(n_clusters, 2) * 0.7 + 0.15
    spread = 0.015
    
    pts = []
    for _ in range(33):
        c = centers[rng.randint(n_clusters)]
        p = c + rng.randn(2) * spread
        pts.append(p)
    pts = np.array(pts)
    pts = np.clip(pts, 0.001, 0.999)
    
    for _ in range(100):
        diff = pts[:, None, :] - pts[None, :, :]
        sq_dists = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(sq_dists, 1.0)
        if np.any(sq_dists < 1e-12):
            idx = np.unravel_index(np.argmin(sq_dists), sq_dists.shape)
            if idx[0] != idx[1]:
                pts[idx[1]] += rng.randn(2) * 0.005
                pts = np.clip(pts, 0.001, 0.999)
        else:
            break
    
    return pts


def _generate_near_collinear(seed: int) -> np.ndarray:
    """Generate points near a line with small perturbations."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0.05, 0.95, 33)
    x = t + rng.randn(33) * 0.01
    y = t * 0.3 + 0.35 + rng.randn(33) * 0.01
    pts = np.column_stack([x, y])
    pts = np.clip(pts, 0.001, 0.999)
    return pts


def _generate_clustered(seed: int) -> np.ndarray:
    """Generate points in tight clusters to minimize convex 7-gons."""
    rng = np.random.RandomState(seed)
    n_clusters = rng.randint(4, 6)
    centers = rng.rand(n_clusters, 2) * 0.8 + 0.1
    spread = 0.04
    
    pts = []
    for _ in range(33):
        c = centers[rng.randint(n_clusters)]
        p = c + rng.randn(2) * spread
        pts.append(p)
    pts = np.array(pts)
    pts = np.clip(pts, 0.0, 1.0)
    return pts


def _generate_arc(seed: int) -> np.ndarray:
    """Generate points on/near a convex arc (fewer convex polygons)."""
    rng = np.random.RandomState(seed)
    # Points along a circular arc
    theta = np.linspace(0, np.pi * 0.7, 33) + rng.randn(33) * 0.02
    r = 0.4 + rng.randn(33) * 0.05
    x = 0.5 + r * np.cos(theta)
    y = 0.5 + r * np.sin(theta)
    pts = np.column_stack([x, y])
    pts = np.clip(pts, 0.0, 1.0)
    return pts


def _generate_concentric(seed: int) -> np.ndarray:
    """Generate points in concentric rings."""
    rng = np.random.RandomState(seed)
    n_rings = 4
    pts = []
    for ring in range(n_rings):
        r = 0.1 + ring * 0.15
        n_in_ring = 33 // n_rings if ring < n_rings - 1 else 33 - (33 // n_rings) * (n_rings - 1)
        angles = np.linspace(0, 2 * np.pi, n_in_ring, endpoint=False) + rng.randn(n_in_ring) * 0.1
        x = 0.5 + r * np.cos(angles)
        y = 0.5 + r * np.sin(angles)
        pts.append(np.column_stack([x, y]))
    pts = np.vstack(pts)
    pts = np.clip(pts, 0.0, 1.0)
    return pts


def _generate_grid_jittered(seed: int) -> np.ndarray:
    """Generate points on a jittered grid."""
    rng = np.random.RandomState(seed)
    n_pts = 6
    x = np.linspace(0.05, 0.95, n_pts)
    y = np.linspace(0.05, 0.95, n_pts)
    X, Y = np.meshgrid(x, y)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    idx = rng.permutation(len(pts))[:33]
    pts = pts[idx]
    pts += rng.randn(33, 2) * 0.02
    pts = np.clip(pts, 0.0, 1.0)
    return pts


def _generate_random(seed: int) -> np.ndarray:
    """Generate random points in unit square."""
    rng = np.random.RandomState(seed)
    return rng.rand(33, 2)


def _perturb(points: np.ndarray, scale: float, rng: np.random.RandomState) -> np.ndarray:
    """Add small random perturbation and clip to [0,1]."""
    perturbed = points + rng.randn(33, 2) * scale
    perturbed = np.clip(perturbed, 0.0, 1.0)
    return perturbed


def erdos_szekeres_problem7() -> np.ndarray:
    global _BEST_POINTS, _BEST_SCORE, _DEADLINE
    
    start_time = time.time()
    
    # Phase 1: Try a variety of initial constructions
    candidates = []
    
    # Tight clusters (most promising)
    for seed in range(40):
        candidates.append(_generate_tight_clusters(seed))
    
    # Near-collinear
    for seed in range(20):
        candidates.append(_generate_near_collinear(seed))
    
    # Clustered
    for seed in range(20):
        candidates.append(_generate_clustered(seed))
    
    # Arc
    for seed in range(15):
        candidates.append(_generate_arc(seed))
    
    # Concentric
    for seed in range(15):
        candidates.append(_generate_concentric(seed))
    
    # Grid jittered
    for seed in range(15):
        candidates.append(_generate_grid_jittered(seed))
    
    # Random
    for seed in range(10):
        candidates.append(_generate_random(seed))
    
    # Evaluate all candidates
    for pts in candidates:
        if time.time() > _DEADLINE:
            break
        valid, score = _evaluate(pts)
        if valid and score > _BEST_SCORE:
            _BEST_SCORE = score
            _BEST_POINTS = pts.copy()
    
    # Phase 2: Local search (perturbation) around the best found
    if _BEST_POINTS is not None:
        current_best = _BEST_POINTS.copy()
        current_score = _BEST_SCORE
        rng = np.random.RandomState(12345)
        
        # Adaptive scales: start larger, refine
        scales = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
        
        for scale in scales:
            if time.time() > _DEADLINE:
                break
            n_trials = 200
            for _ in range(n_trials):
                if time.time() > _DEADLINE:
                    break
                candidate = _perturb(current_best, scale, rng)
                valid, score = _evaluate(candidate)
                if valid and score > current_score:
                    current_score = score
                    current_best = candidate
                    _BEST_SCORE = score
                    _BEST_POINTS = candidate.copy()
        
        # Additional refinement with very small perturbations
        if time.time() < _DEADLINE - 30:
            for _ in range(300):
                if time.time() > _DEADLINE:
                    break
                candidate = _perturb(current_best, 0.0003, rng)
                valid, score = _evaluate(candidate)
                if valid and score > current_score:
                    current_score = score
                    current_best = candidate
                    _BEST_SCORE = score
                    _BEST_POINTS = candidate.copy()
    
    # Return the best found, or a default if nothing valid
    if _BEST_POINTS is not None:
        return _BEST_POINTS
    
    # Fallback: random points
    np.random.seed(0)
    return np.random.rand(33, 2)


# EVOLVE-BLOCK-END
