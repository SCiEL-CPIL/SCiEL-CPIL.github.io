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

def _vertex_score(p, v_idx):
    n = len(p)
    v = p[v_idx]
    # distances to all other vertices
    d = np.array([np.linalg.norm(v - p[j]) for j in range(n) if j != v_idx])
    d_sorted = np.sort(d)
    if np.any(d_sorted < 0.05):
        return 0.0
    # find k: first index minimizing d_{k+1} - d_k for k in 0..17
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
    raw = np.mean([max(di / D, D / di) for di in selected])
    if raw < 1e-12:
        return 0.0
    score = min(1.0, 1.0 / raw)
    return score

def _score_polygon(p):
    """Score a polygon. Returns min vertex score (0 if invalid)."""
    pn, _, _ = _normalize(p)
    if pn is None:
        return 0.0
    if not _check_strictly_convex(pn):
        return 0.0
    min_vs = 1.0
    for i in range(len(pn)):
        vs = _vertex_score(pn, i)
        if vs < min_vs:
            min_vs = vs
        if min_vs < 0.01:  # early exit
            break
    return min_vs

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0
    
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    def radii_to_pts(r):
        return np.column_stack([r * np.cos(base_angles), r * np.sin(base_angles)])
    
    best_score = 0.0
    best_radii = np.ones(n)
    
    def try_candidate(r):
        nonlocal best_score, best_radii
        if time.time() > deadline:
            return
        pts = radii_to_pts(r)
        s = _score_polygon(pts)
        if s > best_score:
            best_score = s
            best_radii = r.copy()
            
    # Initial: Regular polygon
    try_candidate(np.ones(n))
    
    # Strategy: Deterministic Symmetric Coordinate Ascent
    # We assume the optimal solution has high symmetry.
    # The regular 20-gon has D20 symmetry. Perturbations might break this to D10, D5, or C20.
    # We will search over symmetric modes.
    
    # Mode 1: Alternating radii (D10 symmetry) r_i = 1 + a * (-1)^i
    # Mode 2: 5-fold symmetry r_i = 1 + b * cos(2 * i * pi / 5)
    # Mode 3: 10-fold symmetry r_i = 1 + c * cos(2 * i * pi / 10)
    # Mode 4: General D10 (even indices same, odd indices same, but not necessarily alternating sign relative to 1)
    
    # Let's define a generic symmetric parameterization for D10:
    # r[0], r[2], ..., r[18] are all R_even
    # r[1], r[3], ..., r[19] are all R_odd
    # This is the most likely symmetry for a "stable" equidistant-like polygon if it deviates from regular.
    
    # We can also consider D5:
    # r[0]=r[4]=...=r[16]
    # r[1]=r[5]=...=r[17]
    # ...
    
    # Let's try a grid search on (R_even, R_odd) for D10 symmetry first.
    # This covers the "alternating" case and general D10.
    
    # Range for radii: 0.5 to 2.0 seems safe.
    # Grid resolution: 50 steps -> 0.03 step size.
    
    # To be faster, we can do a coarse grid then refine.
    
    # Coarse Grid Search for D10
    r_even_vals = np.linspace(0.6, 1.6, 21)
    r_odd_vals = np.linspace(0.6, 1.6, 21)
    
    for re in r_even_vals:
        for ro in r_odd_vals:
            r = np.ones(n)
            r[::2] = re
            r[1::2] = ro
            try_candidate(r)
            
    if time.time() > deadline:
        return radii_to_pts(best_radii)
        
    # Refine around the best found so far with a finer grid
    # We assume the best found is close to the optimum.
    # We will perform a local coordinate ascent on the full 20-dim space, 
    # but restricted to small steps, to escape the D10 constraint if necessary,
    # or simply to fine-tune the D10 solution.
    
    # Actually, let's try a generic local search from the best D10 candidate.
    # We will use a simple steepest ascent with line search on the full 20 radii.
    
    current_r = best_radii.copy()
    step_size = 0.05
    
    # Coordinate ascent
    for _ in range(200): # Max iterations
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            # Try positive and negative steps
            for delta in [step_size, -step_size]:
                new_r = current_r.copy()
                new_r[i] += delta
                if new_r[i] < 0.2 or new_r[i] > 3.0:
                    continue
                s_new = _score_polygon(radii_to_pts(new_r))
                if s_new > best_score + 1e-6:
                    best_score = s_new
                    best_radii = new_r
                    current_r = new_r
                    improved = True
        if not improved:
            step_size *= 0.8
            if step_size < 0.001:
                break
                
    # Final check
    try_candidate(best_radii)
    
    return radii_to_pts(best_radii)

# EVOLVE-BLOCK-END
