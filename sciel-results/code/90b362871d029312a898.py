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

def _generate_candidates():
    """Generate diverse initial candidates."""
    candidates = []
    random.seed(42)
    
    # Standard grids
    g1 = [(x, y) for x in range(-2, 3) for y in range(-2, 4)]  # 5x6
    candidates.append(g1)
    
    g2 = [(x, y) for x in range(-3, 3) for y in range(-2, 3)]  # 6x5
    candidates.append(g2)
    
    g3 = [(x, y) for x in range(-1, 2) for y in range(-5, 5)]  # 3x10
    candidates.append(g3)
    
    g4 = [(x, y) for x in range(-5, 5) for y in range(-1, 2)]  # 10x3
    candidates.append(g4)
    
    g5 = [(x, y) for x in range(-4, 4) for y in range(-1, 3)]  # 8x3
    candidates.append(g5)
    
    g6 = [(x, y) for x in range(-1, 3) for y in range(-4, 4)]  # 3x8
    candidates.append(g6)
    
    # Sheared grids
    for base in [g1, g2, g3, g4]:
        for shear in [1, -1, 2, -2]:
            sheared = [(x, y + shear * x) for x, y in base]
            valid = [p for p in sheared if -30 <= p[0] <= 30 and -30 <= p[1] <= 30]
            if len(set(valid)) == 30:
                candidates.append(valid)
            sheared2 = [(x + shear * y, y) for x, y in base]
            valid2 = [p for p in sheared2 if -30 <= p[0] <= 30 and -30 <= p[1] <= 30]
            if len(set(valid2)) == 30:
                candidates.append(valid2)
    
    # Parabolic curves
    for a in [1, 2]:
        for sx in range(-3, 4):
            for sy in range(-10, 11):
                pts = []
                for x in range(-8, 9):
                    y = a * x * x + sy
                    px = x + sx
                    py = y
                    if -30 <= px <= 30 and -30 <= py <= 30:
                        pts.append((px, py))
                if len(pts) >= 30:
                    pts = pts[:30]
                    if len(set(pts)) == 30:
                        candidates.append(pts)
    
    # Lattice periods
    for px, py in [(2, 2), (3, 2), (2, 3), (3, 3), (4, 2), (2, 4), (5, 2), (2, 5)]:
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
    
    # Random points with various seeds
    all_coords = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    for seed in [42, 123, 777, 2024, 555, 999]:
        random.seed(seed)
        pts = random.sample(all_coords, 30)
        candidates.append(pts)
    
    # Clustered random: points near a few centers
    for seed in [10, 20, 30]:
        random.seed(seed)
        centers = [random.randint(-20, 20) for _ in range(2)]
        pts = []
        for c in centers:
            for _ in range(15):
                px = c + random.randint(-5, 5)
                py = random.randint(-30, 30)
                if -30 <= px <= 30 and (px, py) not in pts:
                    pts.append((px, py))
        if len(pts) < 30:
            remaining = [p for p in all_coords if p not in pts]
            pts.extend(random.sample(remaining, 30 - len(pts)))
        if len(set(pts)) == 30:
            candidates.append(pts)
    
    return candidates

def _sa_search(points, time_limit, seed=0):
    """Simulated annealing with area-targeted and swap moves."""
    random.seed(seed)
    start_time = time.time()
    points = [tuple(p) for p in points]
    current_score, current_best_area = _score_and_best_area(points)
    best_points = list(points)
    best_score = current_score
    
    T = 5.0
    cooling = 0.997
    min_T = 0.05
    occupied = set(points)
    n = len(points)
    
    while time.time() - start_time < time_limit:
        move_type = random.random()
        
        if move_type < 0.15:
            # 2-point swap move
            i, j = random.sample(range(n), 2)
            pi, pj = points[i], points[j]
            # Try to find new positions for both
            new_i = None
            new_j = None
            for _ in range(20):
                nx1 = random.randint(-30, 30)
                ny1 = random.randint(-30, 30)
                nx2 = random.randint(-30, 30)
                ny2 = random.randint(-30, 30)
                p1 = (nx1, ny1)
                p2 = (nx2, ny2)
                if p1 != pi and p2 != pj and p1 != p2 and p1 not in occupied and p2 not in occupied:
                    new_i, new_j = p1, p2
                    break
            if new_i is None:
                continue
            points[i] = new_i
            points[j] = new_j
            occupied.discard(pi)
            occupied.discard(pj)
            occupied.add(new_i)
            occupied.add(new_j)
            
            new_score, new_best_area = _score_and_best_area(points)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < np.exp(delta / T):
                current_score = new_score
                current_best_area = new_best_area
                if new_score > best_score:
                    best_score = new_score
                    best_points = list(points)
            else:
                points[i] = pi
                points[j] = pj
                occupied.discard(new_i)
                occupied.discard(new_j)
                occupied.add(pi)
                occupied.add(pj)
                
        elif move_type < 0.55:
            # Area-targeted move: try to place point to create target area with existing pairs
            idx = random.randint(0, n - 1)
            old_p = points[idx]
            ox, oy = old_p
            target_a = current_best_area
            
            # Find a pair (j, k) such that if we place idx at position p,
            # the area of triangle (p, points[j], points[k]) = target_a
            # We just try random positions but bias toward creating the target area
            found_move = False
            for _ in range(10):
                # Random nearby position
                ref_idx = random.randint(0, n - 1)
                rx, ry = points[ref_idx]
                new_x = rx + random.randint(-6, 6)
                new_y = ry + random.randint(-6, 6)
                new_p = (new_x, new_y)
                if new_p == old_p or new_p in occupied:
                    continue
                if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                    continue
                found_move = True
                break
            
            if not found_move:
                # Fallback to fully random
                new_x = random.randint(-30, 30)
                new_y = random.randint(-30, 30)
                new_p = (new_x, new_y)
                if new_p == old_p or new_p in occupied:
                    continue
            
            points[idx] = new_p
            occupied.discard(old_p)
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
                occupied.discard(new_p)
                occupied.add(old_p)
                
        else:
            # Standard random move
            idx = random.randint(0, n - 1)
            old_p = points[idx]
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
            new_p = (new_x, new_y)
            
            if new_p == old_p or new_p in occupied:
                continue
            
            points[idx] = new_p
            occupied.discard(old_p)
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
                occupied.discard(new_p)
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
    n = len(points)
    
    for radius in [1, 2, 3, 4, 5]:
        if time.time() - start_time > time_limit:
            break
        for idx in range(n):
            if time.time() - start_time > time_limit:
                break
            old_p = points[idx]
            ox, oy = old_p
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ox + dx, oy + dy
                    if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                        continue
                    new_p = (nx, ny)
                    if new_p in occupied:
                        continue
                    
                    points[idx] = new_p
                    occupied.discard(old_p)
                    occupied.add(new_p)
                    
                    new_score, _ = _score_and_best_area(points)
                    
                    if new_score > best_score:
                        best_score = new_score
                        best_points = list(points)
                        current_score = new_score
                    else:
                        points[idx] = old_p
                        occupied.discard(new_p)
                        occupied.add(old_p)
    
    return best_points, best_score

def _final_polish(points, time_limit):
    """Final high-precision polish with radius 1-2 and swap moves."""
    start_time = time.time()
    points = [tuple(p) for p in points]
    current_score, _ = _score_and_best_area(points)
    best_points = list(points)
    best_score = current_score
    occupied = set(points)
    n = len(points)
    
    # Single-point moves with small radius
    for radius in [1, 2]:
        if time.time() - start_time > time_limit:
            break
        for idx in range(n):
            if time.time() - start_time > time_limit:
                break
            old_p = points[idx]
            ox, oy = old_p
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ox + dx, oy + dy
                    if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                        continue
                    new_p = (nx, ny)
                    if new_p in occupied:
                        continue
                    
                    points[idx] = new_p
                    occupied.discard(old_p)
                    occupied.add(new_p)
                    
                    new_score, _ = _score_and_best_area(points)
                    
                    if new_score > best_score:
                        best_score = new_score
                        best_points = list(points)
                        current_score = new_score
                    else:
                        points[idx] = old_p
                        occupied.discard(new_p)
                        occupied.add(old_p)
    
    # Try some random swaps
    if time.time() - start_time < time_limit:
        for _ in range(200):
            if time.time() - start_time > time_limit:
                break
            i, j = random.sample(range(n), 2)
            pi, pj = points[i], points[j]
            # Swap positions
            points[i] = pj
            points[j] = pi
            occupied.discard(pi)
            occupied.discard(pj)
            occupied.add(pj)
            occupied.add(pi)
            
            new_score, _ = _score_and_best_area(points)
            
            if new_score > best_score:
                best_score = new_score
                best_points = list(points)
                current_score = new_score
            else:
                points[i] = pi
                points[j] = pj
                occupied.discard(pj)
                occupied.discard(pi)
                occupied.add(pi)
                occupied.add(pj)
    
    return best_points, best_score

def construct():
    start_time = time.time()
    hard_deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # Generate and score initial candidates
    candidates = _generate_candidates()
    
    scored_candidates = []
    for c in candidates:
        if len(c) != 30 or len(set(c)) != 30:
            continue
        s, ba = _score_and_best_area(c)
        scored_candidates.append((s, ba, c))
    
    if not scored_candidates:
        # Fallback: return a valid grid
        return [(x, y) for x in range(-2, 3) for y in range(-2, 4)]
    
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_points = None
    best_score = -1
    
    for s, ba, c in scored_candidates[:5]:
        if s > best_score:
            best_score = s
            best_points = [tuple(p) for p in c]
    
    if best_points is None:
        best_points = [tuple(p) for p in scored_candidates[0][2]]
        best_score = scored_candidates[0][0]
    
    # Run SA on top candidates with multiple restarts
    used_keys = set()
    sa_results = []
    
    # Allocate time: ~50s for SA, ~20s for polish, ~15s buffer
    sa_time_budget = 50.0
    polish_time_budget = 20.0
    
    top_k = min(4, len(scored_candidates))
    for i in range(top_k):
        if time.time() > hard_deadline - polish_time_budget - 5:
            break
        
        s, ba, c = scored_candidates[i]
        key = tuple(sorted(c))
        if key in used_keys:
            continue
        used_keys.add(key)
        
        time_left = hard_deadline - time.time() - polish_time_budget - 5
        if time_left < 3.0:
            break
        
        # Allocate time per candidate
        if i == 0:
            t_limit = min(0.35 * sa_time_budget, time_left)
        elif i == 1:
            t_limit = min(0.25 * sa_time_budget, time_left)
        elif i == 2:
            t_limit = min(0.2 * sa_time_budget, time_left)
        else:
            t_limit = min(0.2 * sa_time_budget, time_left)
        
        t_limit = max(2.0, min(t_limit, time_left))
        
        # Run SA with 2 restarts for the first candidate
        num_restarts = 2 if i == 0 else 1
        per_restart = t_limit / num_restarts
        
        for r in range(num_restarts):
            if time.time() > hard_deadline - polish_time_budget - 5:
                break
            remaining = hard_deadline - time.time() - polish_time_budget - 5
            if remaining < 2.0:
                break
            fp, fs = _sa_search(c, time_limit=min(per_restart, remaining), seed=100 + i * 10 + r)
            if fs > best_score:
                best_score = fs
                best_points = fp
            sa_results.append((fs, fp))
    
    # Also run SA on a few random candidates if time permits
    if time.time() < hard_deadline - polish_time_budget - 10:
        for seed in [7, 13, 42]:
            if time.time() > hard_deadline - polish_time_budget - 5:
                break
            random.seed(seed)
            all_coords = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
            rand_pts = random.sample(all_coords, 30)
            remaining = hard_deadline - time.time() - polish_time_budget - 5
            if remaining < 3.0:
                break
            fp, fs = _sa_search(rand_pts, time_limit=min(5.0, remaining), seed=200 + seed)
            if fs > best_score:
                best_score = fs
                best_points = fp
    
    # Polish phase
    if best_points is not None and time.time() < hard_deadline - 5:
        polish_time = min(polish_time_budget, hard_deadline - time.time() - 5)
        if polish_time > 2.0:
            fp, fs = _polish(best_points, time_limit=polish_time * 0.6)
            if fs > best_score:
                best_score = fs
                best_points = fp
        
        # Final polish
        if time.time() < hard_deadline - 3:
            final_time = min(5.0, hard_deadline - time.time() - 3)
            if final_time > 1.0:
                fp, fs = _final_polish(best_points, time_limit=final_time)
                if fs > best_score:
                    best_score = fs
                    best_points = fp
    
    if best_points is None:
        best_points = [tuple(p) for p in scored_candidates[0][2]]
    
    return [tuple(p) for p in best_points]

# Quick test
if __name__ == "__main__":
    t0 = time.time()
    pts = construct()
    t1 = time.time()
    print(f"Score: {_fast_score(pts)}")
    print(f"Time: {t1 - t0:.2f}s")
    print(f"Points: {pts}")
