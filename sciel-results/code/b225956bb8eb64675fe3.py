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
    deadline = start_time + 580.0  # 50s safety margin
    
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Strategy:
    # The vertex score depends on the ratio of distances around the "gap".
    # For a regular 20-gon, the distances from a vertex are chords to other vertices.
    # The gap is typically between the 2nd and 3rd nearest neighbors (indices 1 and 2 in 0-indexed sorted list, 
    # but note the sorted list excludes self, so d_0 is nearest, d_1 is 2nd nearest, etc).
    # Let's analyze the regular 20-gon first.
    # Distances from v_i to v_j are |v_i - v_j|.
    # For regular polygon, distances are determined by step size m.
    # d_m = 2 * R * sin(m * pi / n).
    # Sorted distances for n=20:
    # m=1: 2R sin(pi/20)
    # m=2: 2R sin(2pi/20)
    # ...
    # m=10: 2R sin(10pi/20) = 2R
    
    # The "gap" logic:
    # gaps = d_sorted[1:] - d_sorted[:-1]
    # k = argmin(gaps)
    # D = (d_sorted[k] + d_sorted[k+1]) / 2
    
    # Let's compute the scores for the regular polygon to see where we stand.
    # But we want to improve it.
    # The score is min(1, 1/raw) where raw is mean of max(d/D, D/d) for 4 distances.
    # To maximize score, we want raw to be close to 1.
    # This means d should be close to D.
    # D is the midpoint of the gap.
    # So we want the distances surrounding the gap to be as close to the gap midpoint as possible.
    
    # Let's try to optimize the radii.
    # Since the angles are fixed, the geometry is constrained.
    # We can use a simple local search on the radii.
    
    best_score = 0.0
    best_radii = np.ones(n)
    
    def radii_to_pts(r):
        return np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    
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
    try_candidate(np.ones(n))
    
    # 2. Perturbed regular
    # Try to adjust radii to make the distances around the gap more uniform relative to D.
    # For a regular polygon, the gap is between m=2 and m=3? Or m=1 and m=2?
    # Let's check the gaps for regular 20-gon.
    # d_m = 2 sin(m * pi/20) (assuming R=1)
    # d_1 = 2 sin(0.157) ~ 0.309
    # d_2 = 2 sin(0.314) ~ 0.618
    # d_3 = 2 sin(0.471) ~ 0.924
    # d_4 = 2 sin(0.628) ~ 1.176
    # ...
    # gaps:
    # d_2 - d_1 = 0.309
    # d_3 - d_2 = 0.306
    # d_4 - d_3 = 0.252
    # d_5 - d_4 = 0.195
    # ...
    # The minimum gap is between d_2 and d_3 (indices 1 and 2 in 0-indexed d_sorted).
    # So k=1.
    # D = (d_2 + d_3) / 2.
    # Selected distances: d_{k-1}..d_{k+2} => d_0, d_1, d_2, d_3.
    # i.e., d_1, d_2, d_3, d_4 in step notation? No, d_0 is nearest.
    # d_0 (step 1), d_1 (step 2), d_2 (step 3), d_3 (step 4).
    # Wait, d_sorted[0] is step 1, d_sorted[1] is step 2, etc.
    # So selected are step 1, 2, 3, 4.
    # D is midpoint of step 2 and step 3.
    # We want d_1, d_2, d_3, d_4 to be close to D.
    # Currently d_1 < D < d_2 < d_3 < d_4? No, D is between d_1 and d_2 (step 2 and 3).
    # So d_0 (step 1) is smaller than D.
    # d_1 (step 2) is smaller than D.
    # d_2 (step 3) is larger than D.
    # d_3 (step 4) is larger than D.
    
    # To improve, we might want to make the polygon "fatter" or "thinner" to shift these distances.
    # However, for a regular polygon, all vertices are symmetric.
    # If we perturb radii, we break symmetry.
    
    # Let's try a coordinate ascent on radii.
    current = best_radii.copy()
    step = 0.05
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            for delta in [step, -step]:
                new_r = current.copy()
                new_r[i] += delta
                if new_r[i] < 0.5 or new_r[i] > 2.0:
                    continue
                if try_candidate(new_r):
                    current = new_r.copy()
                    improved = True
        if improved:
            step *= 0.8
        else:
            # Try smaller steps or break if no improvement
            step *= 0.5
            if step < 0.001:
                break
                
    # If we have time, try a more global approach with random restarts
    if time.time() < deadline - 10:
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            # Random perturbation
            r = np.ones(n) + 0.2 * np.random.randn(n)
            r = np.clip(r, 0.5, 2.0)
            try_candidate(r)
            
            # Local search from this new start
            current = r.copy()
            step = 0.05
            for _ in range(100): # limited iterations
                if time.time() > deadline - 5:
                    break
                imp = False
                for i in range(n):
                    for delta in [step, -step]:
                        new_r = current.copy()
                        new_r[i] += delta
                        if new_r[i] < 0.5 or new_r[i] > 2.0:
                            continue
                        if try_candidate(new_r):
                            current = new_r.copy()
                            imp = True
                if not imp:
                    step *= 0.5
                    if step < 0.001:
                        break

    return radii_to_pts(best_radii)

# EVOLVE-BLOCK-END
