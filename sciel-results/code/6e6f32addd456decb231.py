import time
import random
import numpy as np

def _count_areas(points):
    """Compute area counts using vectorized approach."""
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
    
    return counts

def _score(points):
    """Return (max multiplicity, best area value)."""
    counts = _count_areas(points)
    best_area = int(np.argmax(counts))
    return int(counts[best_area]), best_area

def _delta_score(points, idx, new_p):
    """Compute delta in max multiplicity when moving point idx to new_p."""
    # Count areas that involve point idx
    P = np.array(points, dtype=np.int64)
    P_new = P.copy()
    P_new[idx] = new_p
    n = len(P)
    max_area = 6000
    
    # Old areas involving point idx
    old_counts = np.zeros(max_area, dtype=np.int64)
    for j in range(n):
        if j == idx:
            continue
        v = P[idx] - P[j]
        for k in range(n):
            if k == idx or k == j:
                continue
            w = P[j] - P[k]
            cross = v[0] * w[1] - v[1] * w[0]
            area = abs(cross)
            if 0 < area < max_area:
                old_counts[area] += 1
    
    # New areas involving point idx
    new_counts = np.zeros(max_area, dtype=np.int64)
    for j in range(n):
        if j == idx:
            continue
        v = P_new[idx] - P[j]
        for k in range(n):
            if k == idx or k == j:
                continue
            w = P[j] - P[k]
            cross = v[0] * w[1] - v[1] * w[0]
            area = abs(cross)
            if 0 < area < max_area:
                new_counts[area] += 1
    
    # Areas NOT involving point idx stay the same
    base_counts = _count_areas([tuple(p) for p in P])
    # But we need to subtract old contributions and add new ones
    # This is getting complex, so let's just compute full scores
    
    old_total = _count_areas([tuple(p) for p in P])
    new_total = _count_areas([tuple(p) for p in P_new])
    
    old_best = int(np.argmax(old_total))
    new_best = int(np.argmax(new_total))
    old_max = int(old_total[old_best])
    new_max = int(new_total[new_best])
    
    return new_max - old_max, new_best

def _generate_candidates():
    """Generate diverse starting configurations."""
    candidates = []
    
    # 5x6 grid
    candidates.append([(x, y) for x in range(-2, 3) for y in range(-2, 4)])
    
    # 6x5 grid
    candidates.append([(x, y) for x in range(-3, 3) for y in range(-2, 3)])
    
    # Sheared grids
    for shear in [1, -1, 2, -2]:
        pts = [(x, y + shear * x) for x in range(-2, 3) for y in range(-2, 4)]
        valid = [p for p in pts if -30 <= p[0] <= 30 and -30 <= p[1] <= 30]
        if len(valid) == 30:
            candidates.append(valid)
    
    # Parabolic curves
    for a in [1, 2]:
        for shift in range(-10, 11, 5):
            pts = [(x, a * x * x + shift) for x in range(-5, 6)]
            valid = [p for p in pts if -30 <= p[0] <= 30 and -30 <= p[1] <= 30]
            if len(valid) >= 30:
                candidates.append(valid[:30])
    
    # Lattice with period
    for px, py in [(2, 2), (3, 2), (2, 3), (4, 2)]:
        pts = []
        for x in range(-15, 16, px):
            for y in range(-15, 16, py):
                pts.append((x, y))
                if len(pts) >= 30:
                    break
            if len(pts) >= 30:
                break
        if len(pts) == 30:
            candidates.append(pts)
    
    # Random seeds
    all_coords = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    for seed in [42, 123, 777, 2024, 999]:
        random.seed(seed)
        candidates.append(random.sample(all_coords, 30))
    
    return candidates

def _local_search(points, time_limit):
    """Simulated annealing local search."""
    start_time = time.time()
    points = [tuple(p) for p in points]
    current_score, _ = _score(points)
    best_points = list(points)
    best_score = current_score
    occupied = set(points)
    
    T = 8.0
    cooling = 0.995
    min_T = 0.05
    
    while time.time() - start_time < time_limit:
        idx = random.randint(0, 29)
        old_p = points[idx]
        
        new_x = random.randint(-30, 30)
        new_y = random.randint(-30, 30)
        new_p = (new_x, new_y)
        if new_p == old_p or new_p in occupied:
            continue
        
        points[idx] = new_p
        occupied.remove(old_p)
        occupied.add(new_p)
        
        new_score, _ = _score(points)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < np.exp(delta / T):
            current_score = new_score
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
    """Greedy polish with local moves."""
    start_time = time.time()
    points = [tuple(p) for p in points]
    current_score, _ = _score(points)
    best_points = list(points)
    best_score = current_score
    occupied = set(points)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for idx in range(30):
            if time.time() - start_time > time_limit:
                break
            
            current_p = points[idx]
            cx, cy = current_p
            best_delta = 0
            best_new_p = None
            
            for dx in range(-6, 7):
                for dy in range(-6, 7):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                        continue
                    new_p = (nx, ny)
                    if new_p in occupied or new_p == current_p:
                        continue
                    
                    points[idx] = new_p
                    occupied.remove(current_p)
                    occupied.add(new_p)
                    
                    new_score, _ = _score(points)
                    delta = new_score - current_score
                    
                    if delta > best_delta:
                        best_delta = delta
                        best_new_p = new_p
                    
                    points[idx] = current_p
                    occupied.remove(new_p)
                    occupied.add(current_p)
            
            if best_delta > 0 and best_new_p is not None:
                points[idx] = best_new_p
                occupied.remove(current_p)
                occupied.add(best_new_p)
                current_score += best_delta
                if current_score > best_score:
                    best_score = current_score
                    best_points = list(points)
                improved = True
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Generate and score candidates
    candidates = _generate_candidates()
    scored = []
    for c in candidates:
        if len(c) != 30 or len(set(c)) != 30:
            continue
        s, _ = _score(c)
        scored.append((s, c))
    
    if not scored:
        return [(x, y) for x in range(-2, 3) for y in range(-2, 4)]
    
    scored.sort(key=lambda x: x[0], reverse=True)
    
    best_points = [tuple(p) for p in scored[0][1]]
    best_score = scored[0][0]
    
    # Run local search on top candidates
    for i in range(min(3, len(scored))):
        if time.time() > deadline - 5.0:
            break
        s, c = scored[i]
        remaining = deadline - time.time() - 5.0
        if remaining < 3.0:
            break
        
        if i == 0:
            t_limit = min(remaining * 0.5, 20.0)
        else:
            t_limit = min(remaining * 0.25, 10.0)
        
        t_limit = max(1.0, t_limit)
        fp, fs = _local_search(c, time_limit=t_limit)
        if fs > best_score:
            best_score = fs
            best_points = fp
    
    # Polish
    if time.time() < deadline - 2.0:
        polish_time = min(deadline - time.time() - 1.0, 15.0)
        if polish_time > 1.0:
            fp, fs = _polish(best_points, time_limit=polish_time)
            if fs > best_score:
                best_score = fs
                best_points = fp
    
    return [tuple(p) for p in best_points]

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_score(pts)[0]}")
    print(f"Points: {pts}")
