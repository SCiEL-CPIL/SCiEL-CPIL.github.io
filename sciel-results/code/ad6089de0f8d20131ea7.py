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
    min_vs = 1.0
    for i in range(len(pn)):
        vs = _vertex_score(pn, i)
        if vs < min_vs:
            min_vs = vs
        if min_vs < 0.01:
            break
    return min_vs

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
    # Each mode k creates a perturbation: angle[i] += amp * sin(k * base_angles[i] + phase)
    # We also allow radial perturbations: r[i] = 1 + amp_r * sin(k * base_angles[i] + phase_r)
    
    # Precompute mode vectors for angular and radial perturbations
    modes = [1, 2, 3, 4, 5, 7, 10]
    
    def make_perturbation(mode, phase, amp_a, amp_r):
        a = base_angles
        angles_p = a + amp_a * np.sin(mode * a + phase)
        radii_p = 1.0 + amp_r * np.sin(mode * a + phase * 1.3)
        pts = np.column_stack([radii_p * np.cos(angles_p), radii_p * np.sin(angles_p)])
        return pts
    
    # Phase 1: Broad exploration over modes and amplitudes
    # Use a grid of (mode, phase, amp_a, amp_r) combinations
    phases = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    amps_a = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    amps_r = [0.01, 0.02, 0.05, 0.1, 0.15]
    
    # Evaluate all combinations in Phase 1
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
    
    # Phase 2: Local refinement around the best candidate found
    # Do coordinate descent on individual angle and radius adjustments
    if best_score > 0 and time.time() < deadline:
        current = best_pts.copy()
        
        # Extract current angles and radii
        cur_radii = np.linalg.norm(current, axis=1)
        cur_angles = np.arctan2(current[:, 1], current[:, 0])
        
        def reconstruct(radii, angles):
            return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        step = 0.02
        no_improve_count = 0
        
        while time.time() < deadline and step > 0.0005:
            improved = False
            
            # Try adjusting each angle
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
            
            # Try adjusting each radius
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
                step *= 0.7
                no_improve_count += 1
                if no_improve_count > 5:
                    break
            else:
                no_improve_count = 0
    
    # Phase 3: Simulated annealing-style random restarts around best
    if best_score > 0 and time.time() < deadline:
        current = best_pts.copy()
        cur_radii = np.linalg.norm(current, axis=1)
        cur_angles = np.arctan2(current[:, 1], current[:, 0])
        
        def reconstruct(radii, angles):
            return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        T = 0.05  # Initial "temperature"
        cooling = 0.95
        
        for iteration in range(5000):
            if time.time() > deadline:
                break
            
            # Random perturbation
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
                # Accept with probability exp((s_new - best_score) / T)
                if s_new > 0:
                    prob = np.exp((s_new - best_score) / T)
                    if np.random.random() < prob:
                        current = new_pts
                        cur_angles = new_angles
                        cur_radii = new_radii
            
            T *= cooling
        
        # Final local polish
        if time.time() < deadline:
            step = 0.01
            cur_radii = np.linalg.norm(current, axis=1)
            cur_angles = np.arctan2(current[:, 1], current[:, 0])
            
            for _ in range(100):
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
                            current = new_pts
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
                            current = new_pts
                            cur_radii = new_radii
                            improved = True
                if not improved:
                    step *= 0.7
                    if step < 0.001:
                        break
    
    return best_pts

# EVOLVE-BLOCK-END
