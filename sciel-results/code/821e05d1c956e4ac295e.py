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
    expected = 2 * np.pi if signs[0] > 0 else -2 * np.pi
    if abs(abs(total_turn) - 2 * np.pi) > 1e-4:
        return False
    return True

def _vertex_score(p, v_idx):
    n = len(p)
    v = p[v_idx]
    d = np.array([np.linalg.norm(v - p[j]) for j in range(n) if j != v_idx])
    d_sorted = np.sort(d)
    if np.any(d_sorted < 0.05):
        return 0.0
    gaps = d_sorted[1:] - d_sorted[:-1]
    k = int(np.argmin(gaps))
    D = (d_sorted[k] + d_sorted[k + 1]) / 2
    if D < 0.1:
        return 0.0
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
    pn, _, _ = _normalize(p)
    if pn is None:
        return 0.0
    if not _check_strictly_convex(pn):
        return 0.0
    n = len(pn)
    # Compute all pairwise distances
    diff = pn[:, np.newaxis, :] - pn[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    min_vs = 1.0
    for i in range(n):
        d = np.delete(dists[i, :], i)
        d_sorted = np.sort(d)
        if np.any(d_sorted < 0.05):
            return 0.0
        gaps = d_sorted[1:] - d_sorted[:-1]
        k = int(np.argmin(gaps))
        D = (d_sorted[k] + d_sorted[k + 1]) / 2
        if D < 0.1:
            return 0.0
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
        vs = min(1.0, 1.0 / raw)
        if vs < min_vs:
            min_vs = vs
        if min_vs < 0.01:
            break
    return min_vs

def _smooth_vertex_score(p, v_idx, beta=50.0):
    """Smooth surrogate for vertex score using soft-min and soft-max."""
    n = len(p)
    v = p[v_idx]
    d = np.array([np.linalg.norm(v - p[j]) for j in range(n) if j != v_idx])
    d_sorted = np.sort(d)
    
    # Soft penalty for d < 0.05: use soft-min
    # score contribution from min check: if any d < 0.05, score -> 0
    # Use: min_d = d_sorted[0], penalty = exp(-beta * max(0, 0.05 - min_d))
    min_d = d_sorted[0]
    if min_d < 0.05:
        penalty_min = np.exp(-beta * (0.05 - min_d))
    else:
        penalty_min = 1.0
    
    gaps = d_sorted[1:] - d_sorted[:-1]
    # Soft argmin: use softmax over negative gaps
    neg_gaps = -gaps
    max_neg = np.max(neg_gaps)
    weights = np.exp(beta * (neg_gaps - max_neg))
    weights /= np.sum(weights)
    
    # Soft D: weighted average of midpoints
    midpoints = (d_sorted[:-1] + d_sorted[1:]) / 2
    D_soft = np.sum(weights * midpoints)
    
    if D_soft < 0.1:
        penalty_D = np.exp(-beta * (0.1 - D_soft))
    else:
        penalty_D = 1.0
    
    # For the selected distances, we need to handle the k-dependent selection smoothly
    # Use the soft k to blend between the three selection cases
    k_soft = np.sum(weights * np.arange(len(gaps)))  # weighted index
    
    # Case 1: k near 0 -> select d[0:4]
    # Case 2: k near 17 -> select d[15:19]  
    # Case 3: else -> select d[k-1:k+3]
    
    # Compute raw for each case and blend
    def raw_for_selection(indices):
        if len(indices) < 4:
            return 1.0
        di = d_sorted[indices]
        ratios = np.maximum(di / D_soft, D_soft / di)
        return np.mean(ratios)
    
    # Case 1: k=0
    raw1 = raw_for_selection([0, 1, 2, 3])
    # Case 2: k=17
    raw2 = raw_for_selection([15, 16, 17, 18])
    # Case 3: general, use rounded k_soft
    k_round = int(np.round(k_soft))
    if k_round == 0:
        k_round = 1
    elif k_round >= 17:
        k_round = 16
    raw3 = raw_for_selection([k_round - 1, k_round, k_round + 1, k_round + 2])
    
    # Blend weights based on distance of k_soft to 0, 17, and middle
    w1 = np.exp(-beta * k_soft)
    w2 = np.exp(-beta * (17 - k_soft))
    w3 = np.exp(-beta * np.minimum(k_soft, 17 - k_soft))
    w_sum = w1 + w2 + w3
    w1 /= w_sum
    w2 /= w_sum
    w3 /= w_sum
    
    raw_soft = w1 * raw1 + w2 * raw2 + w3 * raw3
    
    if raw_soft < 1e-12:
        return 0.0
    
    score = min(1.0, 1.0 / raw_soft)
    return score * penalty_min * penalty_D

def _smooth_score_polygon(p, beta=50.0):
    """Smooth surrogate for polygon score using soft-min over vertices."""
    pn, _, _ = _normalize(p)
    if pn is None:
        return 0.0
    if not _check_strictly_convex(pn):
        return 0.0
    scores = np.array([_smooth_vertex_score(pn, i, beta) for i in range(len(pn))])
    # Soft min: -log-sum-exp(-beta * scores)
    neg_scores = -beta * scores
    max_neg = np.max(neg_scores)
    soft_min = -np.log(np.sum(np.exp(neg_scores - max_neg))) / beta - max_neg / beta
    return soft_min

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0
    
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    best_score = 0.0
    best_pts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    def try_candidate(pts):
        nonlocal best_score, best_pts
        s = _score_polygon(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
    
    # Start with regular polygon
    try_candidate(best_pts)
    
    # Strategy: explore angular perturbations using sinusoidal modes
    modes = [1, 2, 3, 4, 5, 7, 10]
    
    def make_perturbation(mode, phase, amp_a, amp_r):
        a = base_angles
        angles_p = a + amp_a * np.sin(mode * a + phase)
        radii_p = 1.0 + amp_r * np.sin(mode * a + phase * 1.3)
        pts = np.column_stack([radii_p * np.cos(angles_p), radii_p * np.sin(angles_p)])
        return pts
    
    # Phase 1: Broad exploration
    phases = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    amps_a = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    amps_r = [0.01, 0.02, 0.05, 0.1, 0.15]
    
    for mode in modes:
        if time.time() > deadline:
            break
        for phase in phases:
            if time.time() > deadline:
                break
            for amp_a in amps_a:
                if time.time() > deadline:
                    break
                for amp_r in amps_r:
                    if time.time() > deadline:
                        break
                    pts = make_perturbation(mode, phase, amp_a, amp_r)
                    try_candidate(pts)
    
    # Phase 2: Nelder-Mead optimization on smooth surrogate
    if best_score > 0 and time.time() < deadline:
        current = best_pts.copy()
        cur_radii = np.linalg.norm(current, axis=1)
        cur_angles = np.arctan2(current[:, 1], current[:, 0])
        
        def reconstruct(radii, angles):
            return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        def objective(params):
            radii = params[:n]
            angles = params[n:]
            radii = np.clip(radii, 0.5, 2.0)
            pts = reconstruct(radii, angles)
            # Use exact score for optimization
            s_exact = _score_polygon(pts)
            if s_exact == 0.0:
                return 0.5  # Penalize invalid
            return -s_exact
        
        x0 = np.concatenate([cur_radii, cur_angles])
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 800, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            
            if result.fun < 0:
                opt_radii = np.clip(result.x[:n], 0.5, 2.0)
                opt_angles = result.x[n:]
                opt_pts = reconstruct(opt_radii, opt_angles)
                try_candidate(opt_pts)
        except Exception:
            pass
    
    # Phase 3: Local refinement with coordinate descent
    if best_score > 0 and time.time() < deadline:
        current = best_pts.copy()
        cur_radii = np.linalg.norm(current, axis=1)
        cur_angles = np.arctan2(current[:, 1], current[:, 0])
        
        def reconstruct(radii, angles):
            return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        step = 0.01
        no_improve_count = 0
        
        while time.time() < deadline and step > 0.0003:
            improved = False
            
            for i in range(n):
                if time.time() > deadline:
                    break
                for delta in [step, -step]:
                    new_angles = cur_angles.copy()
                    new_angles[i] += delta
                    new_pts = reconstruct(cur_radii, new_angles)
                    s = _score_polygon(new_pts)
                    if s > best_score:
                        best_score = s
                        best_pts = new_pts
                        current = new_pts
                        cur_angles = new_angles
                        improved = True
            
            for i in range(n):
                if time.time() > deadline:
                    break
                for delta in [step, -step]:
                    new_radii = cur_radii.copy()
                    new_radii[i] = max(0.5, min(2.0, new_radii[i] + delta))
                    new_pts = reconstruct(new_radii, cur_angles)
                    s = _score_polygon(new_pts)
                    if s > best_score:
                        best_score = s
                        best_pts = new_pts
                        current = new_pts
                        cur_radii = new_radii
                        improved = True
            
            if not improved:
                step *= 0.6
                no_improve_count += 1
                if no_improve_count > 8:
                    break
            else:
                no_improve_count = 0
    
    # Phase 4: Simulated annealing
    if best_score > 0 and time.time() < deadline:
        current = best_pts.copy()
        cur_radii = np.linalg.norm(current, axis=1)
        cur_angles = np.arctan2(current[:, 1], current[:, 0])
        
        def reconstruct(radii, angles):
            return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        T = 0.03
        cooling = 0.97
        
        for iteration in range(8000):
            if time.time() > deadline:
                break
            
            new_angles = cur_angles + T * np.random.randn(n)
            new_radii = cur_radii + T * np.random.randn(n)
            new_radii = np.clip(new_radii, 0.5, 2.0)
            
            new_pts = reconstruct(new_radii, new_angles)
            s_new = _score_polygon(new_pts)
            
            if s_new > best_score:
                best_score = s_new
                best_pts = new_pts
                current = new_pts
                cur_angles = new_angles
                cur_radii = new_radii
            else:
                if s_new > 0:
                    prob = np.exp((s_new - best_score) / T)
                    if np.random.random() < prob:
                        current = new_pts
                        cur_angles = new_angles
                        cur_radii = new_radii
            
            T *= cooling
    
    # Phase 5: Final high-precision polish
    if time.time() < deadline:
        current = best_pts.copy()
        cur_radii = np.linalg.norm(current, axis=1)
        cur_angles = np.arctan2(current[:, 1], current[:, 0])
        
        def reconstruct(radii, angles):
            return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        # Use Nelder-Mead with tighter tolerances
        def objective_tight(params):
            radii = params[:n]
            angles = params[n:]
            radii = np.clip(radii, 0.5, 2.0)
            pts = reconstruct(radii, angles)
            s = _score_polygon(pts)
            if s == 0.0:
                return 0.5
            return -s
        
        x0 = np.concatenate([cur_radii, cur_angles])
        
        try:
            result = minimize(objective_tight, x0, method='Nelder-Mead',
                             options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            
            if result.fun < 0:
                opt_radii = np.clip(result.x[:n], 0.5, 2.0)
                opt_angles = result.x[n:]
                opt_pts = reconstruct(opt_radii, opt_angles)
                try_candidate(opt_pts)
        except Exception:
            pass
        
        # Final coordinate polish
        if time.time() < deadline:
            step = 0.003
            cur_radii = np.linalg.norm(best_pts, axis=1)
            cur_angles = np.arctan2(best_pts[:, 1], best_pts[:, 0])
            
            for _ in range(80):
                if time.time() > deadline:
                    break
                improved = False
                for i in range(n):
                    for delta in [step, -step]:
                        new_angles = cur_angles.copy()
                        new_angles[i] += delta
                        new_pts = reconstruct(cur_radii, new_angles)
                        s = _score_polygon(new_pts)
                        if s > best_score:
                            best_score = s
                            best_pts = new_pts
                            cur_angles = new_angles
                            improved = True
                for i in range(n):
                    for delta in [step, -step]:
                        new_radii = cur_radii.copy()
                        new_radii[i] = max(0.5, min(2.0, new_radii[i] + delta))
                        new_pts = reconstruct(new_radii, cur_angles)
                        s = _score_polygon(new_pts)
                        if s > best_score:
                            best_score = s
                            best_pts = new_pts
                            cur_radii = new_radii
                            improved = True
                if not improved:
                    step *= 0.6
                    if step < 0.0003:
                        break
    
    return best_pts

# EVOLVE-BLOCK-END
