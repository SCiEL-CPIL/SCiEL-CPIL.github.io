import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations, product

N = 14

# Precompute triple indices
_TRIPLE_INDICES = np.array(list(combinations(range(N), 3)), dtype=int)

def _score(points):
    """
    Compute the score: min triangle area / convex hull area.
    """
    if len(points) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    # Calculate Minimum Triangle Area
    i_idx = _TRIPLE_INDICES[:, 0]
    j_idx = _TRIPLE_INDICES[:, 1]
    k_idx = _TRIPLE_INDICES[:, 2]
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    # Cross product in 2D
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _neg_score(points):
    """
    Objective function for minimization.
    Returns negative score.
    """
    s = _score(points)
    return -s

def _generate_ring_config(counts, radii, angles_offset):
    """
    Generate points from multiple rings.
    counts: list of point counts per ring, e.g., [7, 7]
    radii: list of radii per ring, e.g., [1.0, 0.5]
    angles_offset: list of angle offsets per ring
    """
    pts = []
    for c, r, off in zip(counts, radii, angles_offset):
        if c == 0:
            continue
        angles = off + np.linspace(0, 2*np.pi, c, endpoint=False)
        ring_pts = r * np.column_stack([np.cos(angles), np.sin(angles)])
        pts.append(ring_pts)
    if not pts:
        return np.zeros((N, 2))
    return np.vstack(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin

    best_score = 0
    best_points = None
    candidates = []

    # 1. Regular 14-gon
    angles_reg = np.linspace(0, 2*np.pi, N, endpoint=False)
    init_reg = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
    candidates.append(init_reg)

    # 2. Grid search for 2-ring (7+7) - Keep this as it was good
    # Radii: outer=1, inner in [0.1, 0.9]
    # Offset: inner ring rotation
    best_ring2_score = 0
    best_ring2_pts = None
    r_inner_vals = np.linspace(0.2, 0.8, 8)
    theta_vals = np.linspace(0, np.pi/7, 8, endpoint=False)
    
    for r_in in r_inner_vals:
        for theta in theta_vals:
            if time.time() > deadline - 60:
                break
            counts = [7, 7]
            radii = [1.0, r_in]
            offsets = [0.0, theta]
            pts = _generate_ring_config(counts, radii, offsets)
            s = _score(pts)
            if s > best_ring2_score:
                best_ring2_score = s
                best_ring2_pts = pts.copy()
        
        if time.time() > deadline - 60:
            break
            
    if best_ring2_pts is not None:
        candidates.append(best_ring2_pts)

    # 3. Grid search for 3-ring (6+2+6)
    # Outer: 6 pts at R=1
    # Middle: 2 pts at R=r_mid
    # Inner: 6 pts at R=r_in
    best_ring3_score = 0
    best_ring3_pts = None
    r_mid_vals = np.linspace(0.4, 0.8, 5)
    r_in_vals = np.linspace(0.1, 0.5, 5)
    theta_mid_vals = np.linspace(0, np.pi, 4, endpoint=False) # 2 pts, so offset matters less, but relative to outer
    theta_in_vals = np.linspace(0, np.pi/6, 4, endpoint=False)
    
    for r_mid in r_mid_vals:
        for r_in in r_in_vals:
            for th_mid in theta_mid_vals:
                for th_in in theta_in_vals:
                    if time.time() > deadline - 40:
                        break
                    counts = [6, 2, 6]
                    radii = [1.0, r_mid, r_in]
                    offsets = [0.0, th_mid, th_in]
                    pts = _generate_ring_config(counts, radii, offsets)
                    s = _score(pts)
                    if s > best_ring3_score:
                        best_ring3_score = s
                        best_ring3_pts = pts.copy()
                    if time.time() > deadline - 40:
                        break
                if time.time() > deadline - 40:
                    break
            if time.time() > deadline - 40:
                break
        if time.time() > deadline - 40:
            break
            
    if best_ring3_pts is not None:
        candidates.append(best_ring3_pts)

    # 4. Add a few random candidates to break symmetry
    np.random.seed(42)
    for _ in range(5):
        if time.time() > deadline - 20:
            break
        pts = np.random.rand(N, 2) * 2 - 1
        # Normalize to unit disk roughly, but keep some spread
        norms = np.linalg.norm(pts, axis=1)
        pts = pts / np.maximum(norms, 1e-6)[:, np.newaxis]
        candidates.append(pts)

    # 5. Score and sort candidates
    scored_candidates = []
    for pts in candidates:
        s = _score(pts)
        scored_candidates.append((s, pts))
    
    # Sort by score descending
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Deduplicate roughly
    unique_candidates = []
    for s, pts in scored_candidates:
        if not unique_candidates:
            unique_candidates.append((s, pts))
        else:
            too_similar = False
            for _, existing_pts in unique_candidates:
                dist = np.linalg.norm(pts - existing_pts)
                if dist < 0.1:
                    too_similar = True
                    break
            if not too_similar:
                unique_candidates.append((s, pts))
        
        if len(unique_candidates) >= 5: # Keep top 5 diverse candidates
            break

    # 6. Local optimization on the best candidates
    # Use Nelder-Mead for non-smooth objective
    for s_init, pts_init in unique_candidates:
        if time.time() > deadline - 10:
            break
            
        try:
            # Nelder-Mead is robust for this small dimension (28)
            # maxiter is limited to ensure we finish in time
            res = minimize(_neg_score, pts_init, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6, 'disp': False})
            
            if res.success or res.fun < -s_init:
                s_opt = -res.fun
                if s_opt > best_score:
                    best_score = s_opt
                    best_points = res.x.copy()
        except Exception:
            pass

    # If no optimization succeeded or improved, use the best initial
    if best_points is None:
        if scored_candidates:
            best_score = scored_candidates[0][0]
            best_points = scored_candidates[0][1]
        else:
            best_points = init_reg
            best_score = _score(init_reg)
            
    return best_points
