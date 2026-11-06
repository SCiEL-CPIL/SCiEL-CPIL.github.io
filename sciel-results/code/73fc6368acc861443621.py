import time
import random
import numpy as np
from collections import Counter

def _fast_score(points):
    """Calculate max multiplicity of a positive doubled triangle area."""
    P = np.array(points, dtype=np.int64)
    n = len(P)
    max_area = 6000
    counts = np.zeros(max_area, dtype=np.int64)
    for i in range(n):
        V = P[i+1:] - P[i]
        n_v = len(V)
        if n_v < 2:
            continue
        idx_i, idx_j = np.triu_indices(n_v, k=1)
        vi = V[idx_i]
        vj = V[idx_j]
        cross = vi[:, 0] * vj[:, 1] - vi[:, 1] * vj[:, 0]
        areas = np.abs(cross)
        mask = (areas > 0) & (areas < max_area)
        if np.any(mask):
            counts += np.bincount(areas[mask], minlength=max_area)
    return int(np.max(counts))

def _score_and_best_area(points):
    """Return (max_multiplicity, best_area_value)."""
    P = np.array(points, dtype=np.int64)
    n = len(P)
    max_area = 6000
    counts = np.zeros(max_area, dtype=np.int64)
    for i in range(n):
        V = P[i+1:] - P[i]
        n_v = len(V)
        if n_v < 2:
            continue
        idx_i, idx_j = np.triu_indices(n_v, k=1)
        vi = V[idx_i]
        vj = V[idx_j]
        cross = vi[:, 0] * vj[:, 1] - vi[:, 1] * vj[:, 0]
        areas = np.abs(cross)
        mask = (areas > 0) & (areas < max_area)
        if np.any(mask):
            counts += np.bincount(areas[mask], minlength=max_area)
    best_area = int(np.argmax(counts))
    return int(counts[best_area]), best_area

def _all_dets(points):
    """Compute absolute doubled area for every triple (i<j<k). Returns (areas, triples) as numpy arrays."""
    P = np.array(points, dtype=np.int64)
    n = len(P)
    # Precompute all pairs (i, j) with i < j
    ii, jj = np.triu_indices(n, k=1)
    pairs = list(zip(ii.tolist(), jj.tolist()))
    # For each pair, we need a third point k > j? No, any k not in {i,j}.
    # To avoid double counting, for each triple (i,j,k) with i<j<k, compute det.
    # Let's just enumerate all triples i<j<k.
    areas = []
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = P[i]
                x2, y2 = P[j]
                x3, y3 = P[k]
                d = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                if d > 0:
                    areas.append(d)
                    triples.append((i, j, k))
    return np.array(areas, dtype=np.int64), triples

def _greedy_set_cover_for_area(points_pool, target_area, k=30):
    """
    Given a pool of candidate points (list of tuples) and a target doubled area,
    greedily select k distinct points from the pool to maximize the number of
    triples among the selected points that have doubled area == target_area.
    """
    n_pool = len(points_pool)
    if n_pool < k:
        return list(points_pool)
    
    # Precompute for each pair (a, b) in pool, the set of points c such that
    # area(triple a,b,c) == target_area. We'll use a dictionary: pair -> list of c indices.
    # But n_pool can be large. Let's limit pool size to something manageable, say 200.
    # Actually, the pool will be the union of initial candidates + some randoms. Let's cap.
    
    # For efficiency, we only consider the pool.
    # Build a mapping: for each unordered pair (a,b), find all c != a,b with area(a,b,c) == target_area.
    # We'll store as a dict: (a,b) -> set of c indices.
    
    # To keep memory reasonable, assume pool <= 500.
    # Precompute areas for all triples in pool? That's O(n_pool^3). For n_pool=200, that's ~1.3M triples, okay.
    
    # Let's cap pool to 300 for safety.
    if n_pool > 300:
        random.shuffle(points_pool)
        points_pool = points_pool[:300]
        n_pool = len(points_pool)
    
    # Precompute all triple areas in pool
    # We need to know, for each triple, its area.
    # Then for greedy: start with empty set, repeatedly add the point that maximizes new triples.
    
    # Build a 3D-ish structure: for each pair (a,b), count how many c give target_area.
    # Actually, for greedy, when we have a current set S, adding point p increases the count by:
    #   number of pairs (a,b) in S such that area(a,b,p) == target_area.
    # So we need, for each candidate point p and each pair (a,b) in S, whether area(a,b,p) == target_area.
    
    # Precompute: for each p in pool, and each pair (a,b) in pool (a<b), area(a,b,p).
    # That's O(n_pool^3) again. For n_pool=300, 27M, might be slow in Python.
    
    # Alternative: for each p, precompute a set of pairs (a,b) that form target_area with p.
    # Then when S is known, for candidate p, the gain = number of pairs in S that are in p's pair-set.
    
    # Let's do it this way:
    # For each p, compute set of (a,b) with a<b, a!=p, b!=p, area(a,b,p)==target_area.
    # Store as a set of frozenset or just a set of tuples.
    
    # To speed up, we can precompute all areas for triples in pool using numpy.
    P = np.array(points_pool, dtype=np.int64)
    n = n_pool
    # Compute all triple areas
    # We'll create a 3D array? No, just loop but vectorize per i.
    # For each i, compute areas of (i, j, k) for all j>i, k>j.
    # Actually, we need area for any triple, not just ordered.
    
    # Let's just do a nested loop but with numpy for the inner part.
    # For each i, for each j>i, compute area(i,j,k) for all k>j.
    # But we need all unordered triples.
    
    # Simpler: for each point p, compute its "compatible pairs" set.
    # For point p (index p_idx), for each pair (a,b) with a<b, a!=p_idx, b!=p_idx:
    #   compute area(a,b,p_idx). If == target_area, add (a,b) to p's set.
    
    # This is O(n_pool^3) in worst case. For n_pool=200, that's 8M, acceptable in numpy if vectorized.
    
    # Let's vectorize: for a fixed p, compute areas of all pairs (a,b) with p.
    # Vectors from p to all other points: V = P - P[p]
    # For each pair (a,b) from the other points, cross product.
    # But we need area(a,b,p) = area(p,a,b) which is the same absolute value.
    # So for fixed p, compute all areas of triples (p, a, b) for a<b, a!=p, b!=p.
    
    # Build a dict: p_idx -> set of (a,b) tuples (with a<b) such that area(p,a,b)==target_area.
    
    compat = [set() for _ in range(n)]
    
    for p_idx in range(n):
        # Other points
        mask = np.ones(n, dtype=bool)
        mask[p_idx] = False
        other_indices = np.where(mask)[0]
        V = P[other_indices] - P[p_idx]  # shape (m, 2)
        m = len(V)
        if m < 2:
            continue
        idx_a, idx_b = np.triu_indices(m, k=1)
        va = V[idx_a]
        vb = V[idx_b]
        cross = va[:, 0] * vb[:, 1] - va[:, 1] * vb[:, 0]
        areas = np.abs(cross)
        # Find where areas == target_area
        match = np.where(areas == target_area)[0]
        for idx in match:
            a_local = idx_a[idx]
            b_local = idx_b[idx]
            a_global = other_indices[a_local]
            b_global = other_indices[b_local]
            if a_global < b_global:
                compat[p_idx].add((a_global, b_global))
            else:
                compat[p_idx].add((b_global, a_global))
    
    # Now greedy: start with empty set S.
    # At each step, for each candidate p not in S, compute gain = number of pairs (a,b) in S that are in compat[p].
    # Pick p with max gain. Tie-break randomly or by index.
    
    S = set()
    selected = []
    
    for _ in range(k):
        best_p = -1
        best_gain = -1
        for p in range(n):
            if p in S:
                continue
            # Count how many pairs in S are in compat[p]
            gain = 0
            # S is a set of indices. Pairs in S: all (a,b) with a<b, a in S, b in S.
            # We need to count how many of those pairs are in compat[p].
            # Since |S| <= 29, we can enumerate pairs in S.
            s_list = list(S)
            for i in range(len(s_list)):
                for j in range(i+1, len(s_list)):
                    a, b = s_list[i], s_list[j]
                    if a > b:
                        a, b = b, a
                    if (a, b) in compat[p]:
                        gain += 1
            if gain > best_gain:
                best_gain = gain
                best_p = p
        if best_p == -1:
            break
        S.add(best_p)
        selected.append(best_p)
    
    return [points_pool[i] for i in selected]

def _generate_grid_candidates():
    """Generate various grid and structured candidates."""
    candidates = []
    g1 = [(x, y) for x in range(-2, 3) for y in range(-2, 4)]
    candidates.append(g1)
    g2 = [(x, y) for x in range(-3, 3) for y in range(-2, 3)]
    candidates.append(g2)
    g3 = [(x, y) for x in range(-1, 2) for y in range(-5, 5)]
    candidates.append(g3)
    g4 = [(x, y) for x in range(-5, 5) for y in range(-1, 2)]
    candidates.append(g4)
    g5 = [(x, y + x) for x, y in g1]
    valid = [p for p in g5 if -30 <= p[0] <= 30 and -30 <= p[1] <= 30]
    if len(set(valid)) == 30:
        candidates.append(valid)
    g6 = [(x + y, y) for x, y in g2]
    valid = [p for p in g6 if -30 <= p[0] <= 30 and -30 <= p[1] <= 30]
    if len(set(valid)) == 30:
        candidates.append(valid)
    for shift in range(-5, 6):
        pts = []
        for x in range(-5, 6):
            y = x * x + shift
            if -30 <= y <= 30:
                pts.append((x, y))
        if len(pts) >= 30:
            pts = pts[:30]
            if len(set(pts)) == 30:
                candidates.append(pts)
    for a in [1, 2]:
        for shift_x in range(-3, 4):
            for shift_y in range(-10, 11):
                pts = []
                for x in range(-6, 7):
                    y = a * x * x + shift_y
                    px = x + shift_x
                    py = y
                    if -30 <= px <= 30 and -30 <= py <= 30:
                        pts.append((px, py))
                if len(pts) >= 30:
                    pts = pts[:30]
                    if len(set(pts)) == 30:
                        candidates.append(pts)
    for px, py in [(2, 2), (3, 2), (2, 3), (3, 3), (4, 2), (2, 4)]:
        pts = []
        for x in range(-15, 16, px):
            for y in range(-15, 16, py):
                pts.append((x, y))
                if len(pts) >= 30:
                    break
            if len(pts) >= 30:
                break
        if len(pts) == 30 and len(set(pts)) == 30:
            candidates.append(pts)
    random.seed(42)
    all_coords = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    for seed in [42, 123, 777, 2024]:
        random.seed(seed)
        pts = random.sample(all_coords, 30)
        candidates.append(pts)
    return candidates

def _local_search(points, time_limit, target_area=None):
    """Local search optimizing for max area multiplicity."""
    start_time = time.time()
    points = [tuple(p) for p in points]
    current_score, current_best_area = _score_and_best_area(points)
    best_points = list(points)
    best_score = current_score
    
    T = 5.0
    cooling = 0.997
    min_T = 0.05
    occupied = set(points)
    
    while time.time() - start_time < time_limit:
        idx = random.randint(0, 29)
        old_p = points[idx]
        found_move = False
        for _ in range(5):
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
            new_p = (new_x, new_y)
            if new_p == old_p or new_p in occupied:
                continue
            found_move = True
            break
        if not found_move:
            continue
        points[idx] = new_p
        occupied.remove(old_p)
        occupied.add(new_p)
        new_score, new_best_area = _score_and_best_area(points)
        delta = new_score - current_score
        if delta > 0 or random.random() < np.exp(delta / T):
            current_score = new_score
            current_best_area = new_best_area
            if new_score > best_score:
                best_score = new_score
                best_points = list(points)
        else:
            points[idx] = old_p
            occupied.remove(new_p)
            occupied.add(old_p)
        T *= cooling
        if T < min_T:
            T = min_T
    return best_points, best_score

def _polish(points, time_limit):
    """Focused polish: try all single-point moves to nearby positions."""
    start_time = time.time()
    points = [tuple(p) for p in points]
    current_score, _ = _score_and_best_area(points)
    best_points = list(points)
    best_score = current_score
    occupied = set(points)
    for idx in range(30):
        if time.time() - start_time > time_limit:
            break
        old_p = points[idx]
        ox, oy = old_p
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = ox + dx, oy + dy
                if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                    continue
                new_p = (nx, ny)
                if new_p in occupied:
                    continue
                points[idx] = new_p
                occupied.remove(old_p)
                occupied.add(new_p)
                new_score, _ = _score_and_best_area(points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = list(points)
                    current_score = new_score
                else:
                    points[idx] = old_p
                    occupied.remove(new_p)
                    occupied.add(old_p)
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Generate candidates
    candidates = _generate_grid_candidates()
    
    # Score all candidates
    scored = []
    for c in candidates:
        if len(c) != 30 or len(set(c)) != 30:
            continue
        s, a = _score_and_best_area(c)
        scored.append((s, a, c))
    
    if not scored:
        return [(x, y) for x in range(-2, 3) for y in range(-2, 4)]
    
    scored.sort(key=lambda x: x[0], reverse=True)
    
    best_points = [tuple(p) for p in scored[0][2]]
    best_score = scored[0][0]
    best_area = scored[0][1]
    
    # Try greedy set-cover for top areas
    # Use the pool of all candidate points + some random points
    all_pool = []
    seen = set()
    for s, a, c in scored:
        for p in c:
            if p not in seen:
                all_pool.append(p)
                seen.add(p)
    
    # Add some random points to the pool
    random.seed(999)
    all_coords = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    for _ in range(200):
        p = random.choice(all_coords)
        if p not in seen:
            all_pool.append(p)
            seen.add(p)
    
    # Try greedy for top 3 areas
    top_areas = []
    seen_areas = set()
    for s, a, c in scored:
        if a not in seen_areas and a > 0:
            top_areas.append(a)
            seen_areas.add(a)
        if len(top_areas) >= 3:
            break
    
    for target_a in top_areas:
        if time.time() > deadline - 10.0:
            break
        greedy_pts = _greedy_set_cover_for_area(all_pool, target_a, k=30)
        if len(greedy_pts) == 30:
            gs, ga = _score_and_best_area(greedy_pts)
            if gs > best_score:
                best_score = gs
                best_points = greedy_pts
                best_area = ga
    
    # Run local search on top candidates
    used_keys = set()
    top_n = min(5, len(scored))
    for i in range(top_n):
        if time.time() > deadline - 5.0:
            break
        s, a, c = scored[i]
        key = tuple(sorted(c))
        if key in used_keys:
            continue
        used_keys.add(key)
        remaining = deadline - time.time() - 2.0
        if remaining < 2.0:
            break
        if i == 0:
            t_limit = min(remaining * 0.35, 12.0)
        elif i == 1:
            t_limit = min(remaining * 0.25, 10.0)
        else:
            t_limit = min(remaining * 0.15, 6.0)
        t_limit = max(1.0, t_limit)
        fp, fs = _local_search(c, time_limit=t_limit)
        if fs > best_score:
            best_score = fs
            best_points = fp
    
    # Polish the best result
    if time.time() < deadline - 2.0:
        polish_time = min(deadline - time.time() - 1.0, 8.0)
        if polish_time > 1.0:
            fp, fs = _polish(best_points, time_limit=polish_time)
            if fs > best_score:
                best_score = fs
                best_points = fp
    
    return [tuple(p) for p in best_points]

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_fast_score(pts)}")
    print(f"Points: {pts}")
