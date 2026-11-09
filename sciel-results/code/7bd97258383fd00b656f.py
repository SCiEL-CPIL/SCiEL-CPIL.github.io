import numpy as np
import time

def _vertex_score(v, pts):
    """Compute the vertex score for vertex v given all points pts."""
    diffs = pts - v
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    # Exclude self
    mask = np.ones(len(pts), dtype=bool)
    mask[np.argmin(dists)] = False  # self distance is 0
    d = np.sort(dists[mask])
    # d has 19 elements
    if np.any(d < 0.05):
        return 0.0
    gaps = np.diff(d)
    k = np.argmin(gaps)
    D = (d[k] + d[k+1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        selected = d[0:4]
    elif k == 17:
        selected = d[15:19]
    else:
        selected = d[k-1:k+3]
    ratios = np.maximum(selected / D, D / selected)
    raw = np.mean(ratios)
    return min(1.0, 1.0 / raw)

def _min_vertex_score(pts):
    scores = [_vertex_score(pts[i], pts) for i in range(len(pts))]
    return min(scores)

def _is_valid(pts):
    if len(pts) != 20:
        return False
    if not np.all(np.isfinite(pts)):
        return False
    # Check no repeated points
    for i in range(20):
        for j in range(i+1, 20):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-10:
                return False
    # Check convexity: all cross products same sign
    signs = []
    for i in range(20):
        p0 = pts[i]
        p1 = pts[(i+1) % 20]
        p2 = pts[(i+2) % 20]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        if abs(cross) < 1e-12:
            return False
        signs.append(np.sign(cross))
    if not all(s == signs[0] for s in signs):
        return False
    return True

def _normalize(pts):
    centroid = np.mean(pts, axis=0)
    pts_norm = pts - centroid
    dists = np.linalg.norm(pts_norm, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return pts_norm, False
    pts_norm = pts_norm / mean_dist
    return pts_norm, True

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = 620.0  # safety margin
    
    n = 20
    # Start with regular 20-gon
    base_angles = np.linspace(0.0, 2.0*np.pi, n, endpoint=False)
    pts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    norm_pts, valid = _normalize(pts)
    if valid and _is_valid(norm_pts):
        best_score = _min_vertex_score(norm_pts)
        best_pts = pts.copy()
    else:
        best_score = 0.0
        best_pts = pts.copy()
    
    # Try structured angular perturbations
    # The key is to adjust angles to make distances more uniform
    rng = np.random.default_rng(42)
    
    # First, try small uniform angular shifts
    for shift in np.linspace(-0.05, 0.05, 21):
        if time.time() > deadline:
            break
        trial_angles = base_angles + shift
        trial = np.column_stack([np.cos(trial_angles), np.sin(trial_angles)])
        norm_trial, valid = _normalize(trial)
        if not valid or not _is_valid(norm_trial):
            continue
        score = _min_vertex_score(norm_trial)
        if score > best_score:
            best_score = score
            best_pts = trial.copy()
    
    # Try perturbing individual angles
    for _ in range(500):
        if time.time() > deadline:
            break
        trial_angles = base_angles.copy()
        # Perturb a random subset of angles
        num_perturb = rng.integers(1, 5)
        indices = rng.choice(n, size=num_perturb, replace=False)
        trial_angles[indices] += rng.normal(0, 0.02, size=num_perturb)
        # Ensure angles are in [0, 2*pi)
        trial_angles = trial_angles % (2*np.pi)
        # Sort angles to maintain cyclic order
        trial_angles = np.sort(trial_angles)
        # Re-center to ensure they span [0, 2*pi)
        trial_angles = (trial_angles - trial_angles[0]) % (2*np.pi)
        trial = np.column_stack([np.cos(trial_angles), np.sin(trial_angles)])
        norm_trial, valid = _normalize(trial)
        if not valid or not _is_valid(norm_trial):
            continue
        score = _min_vertex_score(norm_trial)
        if score > best_score:
            best_score = score
            best_pts = trial.copy()
    
    # Try more systematic perturbation: adjust angles to equalize edge lengths
    # For a convex polygon, we can parameterize by angles and optimize
    current_angles = base_angles.copy()
    for outer_iter in range(100):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            # Try small adjustments to angle i
            for delta in np.linspace(-0.05, 0.05, 21):
                if delta == 0:
                    continue
                trial_angles = current_angles.copy()
                trial_angles[i] += delta
                trial_angles = trial_angles % (2*np.pi)
                trial_angles = np.sort(trial_angles)
                trial_angles = (trial_angles - trial_angles[0]) % (2*np.pi)
                trial = np.column_stack([np.cos(trial_angles), np.sin(trial_angles)])
                norm_trial, valid = _normalize(trial)
                if not valid or not _is_valid(norm_trial):
                    continue
                score = _min_vertex_score(norm_trial)
                if score > best_score + 1e-10:
                    best_score = score
                    best_pts = trial.copy()
                    current_angles = trial_angles.copy()
                    improved = True
                    break
        if not improved:
            break
    
    return best_pts
