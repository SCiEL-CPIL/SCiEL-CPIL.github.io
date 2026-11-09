# EVOLVE-BLOCK-START
import numpy as np
import time

def _normalize(pts):
    centroid = pts.mean(axis=0)
    p = pts - centroid
    d = np.linalg.norm(p, axis=1)
    mean_d = d.mean()
    if mean_d <= 0.01:
        return None, None, None
    p = p / mean_d
    return p, d, mean_d

def _check_strictly_convex(p):
    n = len(p)
    if n < 3:
        return False
    # no repeated points
    for i in range(n):
        if np.linalg.norm(p[i] - p[(i + 1) % n]) < 1e-9:
            return False
    # compute cross products of consecutive edges
    signs = []
    for i in range(n):
        p0 = p[i]
        p1 = p[(i + 1) % n]
        p2 = p[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) < 1e-9:
            return False
        signs.append(1 if cross > 0 else -1)
    # all same sign
    if signs.count(signs[0]) != n:
        return False
    # total turning = +-2*pi
    angles = []
    for i in range(n):
        p0 = p[i]
        p1 = p[(i + 1) % n]
        p2 = p[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_a = np.clip(cos_a, -1, 1)
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        sin_a = cross / (np.linalg.norm(v1) * np.linalg.norm(v2))
        ang = np.arctan2(sin_a, cos_a)
        angles.append(ang)
    total_turn = sum(angles)
    expected = 2 * np.pi if signs[0] > 0 else -2 * np.pi
    if abs(abs(total_turn) - 2 * np.pi) > 1e-4:
        return False
    return True

def _vertex_score_from_dist(d_sorted):
    """
    Compute vertex score given the sorted distances to other vertices.
    d_sorted must be of length 19.
    """
    if np.any(d_sorted < 0.05):
        return 0.0
    
    gaps = d_sorted[1:] - d_sorted[:-1]
    k = int(np.argmin(gaps))  # 0..17
    D = (d_sorted[k] + d_sorted[k + 1]) / 2
    if D < 0.1:
        return 0.0
    
    # select four distances
    if k == 0:
        selected = d_sorted[0:4]
    elif k == 17:
        selected = d_sorted[15:19]
    else:
        selected = d_sorted[k - 1:k + 3]
        
    if len(selected) < 4:
        return 0.0
        
    # raw = mean over these four d of max(d/D, D/d)
    ratios = np.maximum(selected / D, D / selected)
    raw = np.mean(ratios)
    
    if raw < 1e-12:
        return 0.0
        
    score = min(1.0, 1.0 / raw)
    return score

def _compute_all_vertex_scores(p):
    """
    Vectorized computation of all vertex scores.
    p: (n, 2) array of normalized points.
    Returns: (n,) array of scores.
    """
    n = len(p)
    # Compute distance matrix: (n, n)
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    scores = np.zeros(n)
    for i in range(n):
        # Get distances to all other vertices
        d_i = dist[i].copy()
        d_i[i] = np.inf # Exclude self
        
        # Sort distances
        d_sorted = np.sort(d_i)
        
        # Compute score
        scores[i] = _vertex_score_from_dist(d_sorted)
        
    return scores

def _score_polygon(p):
    """Score a polygon. Returns min vertex score (0 if invalid)."""
    pn, _, _ = _normalize(p)
    if pn is None:
        return 0.0
    if not _check_strictly_convex(pn):
        return 0.0
    
    scores = _compute_all_vertex_scores(pn)
    return np.min(scores)

def _smooth_score(p):
    """
    Smoothed surrogate for the min-vertex-score.
    Uses soft-min to make the objective continuous for optimization.
    """
    pn, _, _ = _normalize(p)
    if pn is None:
        return -100.0
        
    if not _check_strictly_convex(pn):
        return -100.0
        
    scores = _compute_all_vertex_scores(pn)
    
    # Soft min
    beta = 50.0
    # Avoid log(0)
    scores_safe = np.maximum(scores, 1e-12)
    soft_min = -np.log(np.sum(np.exp(-beta * scores_safe)) / len(scores_safe)) / beta
    return soft_min

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0  # 50s safety margin
    
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    def radii_to_pts(r):
        return np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    
    best_score = 0.0
    best_radii = np.ones(n)
    
    def try_candidate(r):
        nonlocal best_score, best_radii
        pts = radii_to_pts(r)
        s = _score_polygon(pts)
        if s > best_score:
            best_score = s
            best_radii = r.copy()
            return True
        return False

    # Initial seeds
    # 1. Regular
    r_current = np.ones(n)
    try_candidate(r_current)
    
    # 2. Alternating
    r_alt = 1.0 + 0.1 * np.array([1, -1] * 10)
    try_candidate(r_alt)
    
    # 3. Random
    rng = np.random.default_rng(42)
    for _ in range(50):
        r_rand = 1.0 + 0.3 * rng.normal(0, 1, n)
        r_rand = np.clip(r_rand, 0.5, 2.0)
        try_candidate(r_rand)

    # Phase 1: Global Exploration with Coordinate Ascent on Smooth Surrogate
    # We use a simple hill-climbing/coordinate-ascent on the smooth score
    # to get a good starting point for the precise discrete search.
    
    current_r = best_radii.copy()
    current_smooth = _smooth_score(radii_to_pts(current_r))
    
    # Coordinate Ascent
    step_size = 0.05
    for _ in range(200): # Limit iterations
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            # Try increasing
            r_test = current_r.copy()
            r_test[i] += step_size
            if r_test[i] > 2.5 or r_test[i] < 0.5:
                continue
            s_test = _smooth_score(radii_to_pts(r_test))
            if s_test > current_smooth:
                current_r = r_test
                current_smooth = s_test
                improved = True
            # Try decreasing
            r_test = current_r.copy()
            r_test[i] -= step_size
            if r_test[i] > 2.5 or r_test[i] < 0.5:
                continue
            s_test = _smooth_score(radii_to_pts(r_test))
            if s_test > current_smooth:
                current_r = r_test
                current_smooth = s_test
                improved = True
        
        if not improved:
            step_size *= 0.9
            if step_size < 0.001:
                break
        
        # Update best with discrete score
        try_candidate(current_r)
        
        # Occasionally restart from a random point near best
        if rng.random() < 0.1:
            current_r = best_radii + 0.2 * rng.normal(0, 1, n)
            current_r = np.clip(current_r, 0.5, 2.5)
            current_smooth = _smooth_score(radii_to_pts(current_r))

    # Phase 2: Precise Local Search on Discrete Score
    # This is where the vectorization shines. We can afford many small steps.
    
    current_r = best_radii.copy()
    current_score = best_score
    
    # Multi-start local search
    for restart in range(10):
        if time.time() > deadline:
            break
        
        # Start from best or a perturbation of it
        if restart == 0:
            current_r = best_radii.copy()
        else:
            current_r = best_radii + 0.1 * rng.normal(0, 1, n)
            current_r = np.clip(current_r, 0.5, 2.5)
        
        current_score = _score_polygon(radii_to_pts(current_r))
        
        step = 0.02
        for _ in range(500): # Max steps per restart
            if time.time() > deadline:
                break
            improved = False
            # Try all coordinates
            for i in range(n):
                if time.time() > deadline:
                    break
                for delta in [step, -step]:
                    r_test = current_r.copy()
                    r_test[i] += delta
                    if r_test[i] < 0.5 or r_test[i] > 2.5:
                        continue
                    
                    s_test = _score_polygon(radii_to_pts(r_test))
                    if s_test > current_score:
                        current_r = r_test
                        current_score = s_test
                        improved = True
                        # Update global best
                        if s_test > best_score:
                            best_score = s_test
                            best_radii = current_r.copy()
            
            if not improved:
                step *= 0.8
                if step < 0.0005:
                    break

    return radii_to_pts(best_radii)

# EVOLVE-BLOCK-END
