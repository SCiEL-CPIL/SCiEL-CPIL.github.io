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

def _generate_grid_candidates():
    """Generate various grid and structured candidates."""
    candidates = []
    
    # 5x6 grid
    g1 = [(x, y) for x in range(-2, 3) for y in range(-2, 4)]
    candidates.append(g1)
    
    # 6x5 grid
    g2 = [(x, y) for x in range(-3, 3) for y in range(-2, 3)]
    candidates.append(g2)
    
    # 3x10 grid
    g3 = [(x, y) for x in range(-1, 2) for y in range(-5, 5)]
    candidates.append(g3)
    
    # 10x3 grid
    g4 = [(x, y) for x in range(-5, 5) for y in range(-1, 2)]
    candidates.append(g4)
    
    # Sheared 5x6
    g5 = [(x, y + x) for x, y in g1]
    valid = [p for p in g5 if -30 <= p[0] <= 30 and -30 <= p[1] <= 30]
    if len(set(valid)) == 30:
        candidates.append(valid)
    
    # Sheared 6x5
    g6 = [(x + y, y) for x, y in g2]
    valid = [p for p in g6 if -30 <= p[0] <= 30 and -30 <= p[1] <= 30]
    if len(set(valid)) == 30:
        candidates.append(valid)
    
    # Parabolic: y = x^2 shifted
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
        elif len(pts) == 30:
            candidates.append(pts)
    
    # More parabolic variants
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
                elif len(pts) == 30:
                    candidates.append(pts)
    
    # Lattice with period
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
    
    # Random with seed
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
        
        # Try several candidate moves
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
    
    # For each point, try moving to nearby unoccupied positions
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
    deadline = start_time + 55.0  # Safety margin for 90s kill
    
    # Generate candidates
    candidates = _generate_grid_candidates()
    
    # Score all candidates
    scored = []
    for c in candidates:
        if len(c) != 30 or len(set(c)) != 30:
            continue
        s, _ = _score_and_best_area(c)
        scored.append((s, c))
    
    if not scored:
        return [(x, y) for x in range(-2, 3) for y in range(-2, 4)]
    
    scored.sort(key=lambda x: x[0], reverse=True)
    
    best_points = [tuple(p) for p in scored[0][1]]
    best_score = scored[0][0]
    
    # Run local search on top candidates
    used_keys = set()
    top_n = min(5, len(scored))
    
    for i in range(top_n):
        if time.time() > deadline - 3.0:
            break
        s, c = scored[i]
        key = tuple(sorted(c))
        if key in used_keys:
            continue
        used_keys.add(key)
        
        remaining = deadline - time.time() - 2.0
        if remaining < 2.0:
            break
        
        # Allocate time based on rank
        if i == 0:
            t_limit = min(remaining * 0.35, 15.0)
        elif i == 1:
            t_limit = min(remaining * 0.25, 12.0)
        else:
            t_limit = min(remaining * 0.15, 8.0)
        
        t_limit = max(1.0, t_limit)
        
        fp, fs = _local_search(c, time_limit=t_limit)
        if fs > best_score:
            best_score = fs
            best_points = fp
    
    # Polish the best result
    if time.time() < deadline - 2.0:
        polish_time = min(deadline - time.time() - 1.0, 10.0)
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
