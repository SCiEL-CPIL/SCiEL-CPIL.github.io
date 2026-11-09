# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

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
    for i in range(n):
        if np.linalg.norm(p[i] - p[(i + 1) % n]) < 1e-9:
            return False
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
    if signs.count(signs[0]) != n:
        return False
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
    if abs(abs(total_turn) - 2 * np.pi) > 1e-4:
        return False
    return True

def _vectorized_vertex_scores(p):
    n = len(p)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.linalg.norm(p[i] - p[j])
    
    scores = np.zeros(n)
    for i in range(n):
        d = np.delete(dist[i], i)
        d_sorted = np.sort(d)
        
        if np.any(d_sorted < 0.05):
            scores[i] = 0.0
            continue
            
        gaps = d_sorted[1:] - d_sorted[:-1]
        k = int(np.argmin(gaps))
        D = (d_sorted[k] + d_sorted[k + 1]) / 2
        
        if D < 0.1:
            scores[i] = 0.0
            continue
            
        if k == 0:
            selected = d_sorted[0:4]
        elif k == 17:
            selected = d_sorted[15:19]
        else:
            selected = d_sorted[k - 1:k + 3]
            
        if len(selected) < 4:
            scores[i] = 0.0
            continue
            
        ratios = np.maximum(selected / D, D / selected)
        raw = np.mean(ratios)
        if raw < 1e-12:
            scores[i] = 0.0
        else:
            scores[i] = min(1.0, 1.0 / raw)
            
    return scores

def _score_polygon(p):
    pn, _, _ = _normalize(p)
    if pn is None:
        return 0.0
    if not _check_strictly_convex(pn):
        return 0.0
    scores = _vectorized_vertex_scores(pn)
    return np.min(scores)

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0
    
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    best_score = 0.0
    best_pts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    def radii_angles_to_pts(radii, angles):
        return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    # Phase 1: Optimize radii with fixed angles
    x0 = np.ones(n)
    
    def objective_radii(radii):
        pts = radii_angles_to_pts(radii, base_angles)
        s = _score_polygon(pts)
        return -s
    
    if time.time() < deadline - 20:
        try:
            res = minimize(objective_radii, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'disp': False})
            if res.x is not None:
                candidate_pts = radii_angles_to_pts(res.x, base_angles)
                s = _score_polygon(candidate_pts)
                if s > best_score:
                    best_score = s
                    best_pts = candidate_pts
        except Exception:
            pass

    # Phase 2: Local search on radii and angles
    if best_score > 0 and time.time() < deadline - 10:
        cur_radii = np.linalg.norm(best_pts, axis=1)
        cur_angles = np.arctan2(best_pts[:, 1], best_pts[:, 0])
        
        step_r = 0.01
        step_a = 0.01
        no_improve_count = 0
        
        while time.time() < deadline - 10 and (step_r > 0.0005 or step_a > 0.0005):
            improved = False
            
            # Adjust radii
            for i in range(n):
                if time.time() > deadline - 10:
                    break
                for delta in [step_r, -step_r]:
                    new_radii = cur_radii.copy()
                    new_radii[i] = max(0.5, min(2.0, new_radii[i] + delta))
                    new_pts = radii_angles_to_pts(new_radii, cur_angles)
                    s = _score_polygon(new_pts)
                    if s > best_score:
                        best_score = s
                        best_pts = new_pts
                        cur_radii = new_radii
                        improved = True
            
            # Adjust angles
            for i in range(n):
                if time.time() > deadline - 10:
                    break
                for delta in [step_a, -step_a]:
                    new_angles = cur_angles.copy()
                    new_angles[i] += delta
                    # Ensure angles are sorted to maintain convexity/cyclic order
                    # If we perturb too much, order might break. 
                    # For small steps, it's fine.
                    new_pts = radii_angles_to_pts(cur_radii, new_angles)
                    s = _score_polygon(new_pts)
                    if s > best_score:
                        best_score = s
                        best_pts = new_pts
                        cur_angles = new_angles
                        improved = True
            
            if not improved:
                step_r *= 0.7
                step_a *= 0.7
                no_improve_count += 1
                if no_improve_count > 5:
                    break
            else:
                no_improve_count = 0

    return best_pts
# EVOLVE-BLOCK-END
