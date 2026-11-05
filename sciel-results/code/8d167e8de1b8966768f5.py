"""Construct 30 integer points in [-30,30]^2 maximizing largest multiplicity of a single positive doubled area."""
import time
import random
from collections import Counter

DEADLINE = 85.0  # safety margin before 90s kill

def _doubled_area(p, q, r):
    """Absolute doubled area of triangle pqr (integer)."""
    return abs((q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0]))

def _compute_all_areas(points):
    """Return Counter of positive doubled areas for all triples."""
    n = len(points)
    counts = Counter()
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = _doubled_area(points[i], points[j], points[k])
                if a > 0:
                    counts[a] += 1
    return counts

def _score_from_counts(counts):
    """Largest multiplicity of a single positive doubled area."""
    if not counts:
        return 0
    return max(counts.values())

def _update_area_counts(counts, points, idx, old_pt, new_pt):
    """Update area counts when points[idx] changes from old_pt to new_pt. O(n^2)."""
    n = len(points)
    for i in range(n):
        if i == idx:
            continue
        for j in range(i+1, n):
            if j == idx:
                continue
            a_old = _doubled_area(old_pt, points[i], points[j])
            a_new = _doubled_area(new_pt, points[i], points[j])
            if a_old > 0:
                counts[a_old] -= 1
                if counts[a_old] == 0:
                    del counts[a_old]
            if a_new > 0:
                counts[a_new] = counts.get(a_new, 0) + 1

def _make_initial_points(seed=42):
    """Create two congruent 15-point triangular lattice subsets shifted by a vector."""
    rng = random.Random(seed)
    
    # 15-point triangular arrangement: rows 1,2,3,4,5
    pts1 = []
    for row in range(5):
        for col in range(row + 1):
            pts1.append((col, row))
    
    shift_x = rng.randint(3, 8)
    shift_y = rng.randint(3, 8)
    pts2 = [(x + shift_x, y + shift_y) for (x, y) in pts1]
    
    all_pts = pts1 + pts2
    seen = set()
    valid = []
    for p in all_pts:
        if p not in seen:
            seen.add(p)
            valid.append(p)
    
    if len(valid) < 30:
        for sx in range(1, 15):
            for sy in range(1, 15):
                pts2_try = [(x + sx, y + sy) for (x, y) in pts1]
                all_try = pts1 + pts2_try
                seen_try = set()
                valid_try = []
                for p in all_try:
                    if p not in seen_try:
                        seen_try.add(p)
                        valid_try.append(p)
                if len(valid_try) == 30:
                    valid = valid_try
                    break
            if len(valid) == 30:
                break
    
    xs = [p[0] for p in valid]
    ys = [p[1] for p in valid]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    offset_x = -(min_x + max_x) // 2
    offset_y = -(min_y + max_y) // 2
    valid = [(x + offset_x, y + offset_y) for (x, y) in valid]
    valid = [(max(-30, min(30, x)), max(-30, min(30, y))) for (x, y) in valid]
    
    seen = set()
    final = []
    for p in valid:
        if p not in seen:
            seen.add(p)
            final.append(p)
    
    while len(final) < 30:
        x = rng.randint(-30, 30)
        y = rng.randint(-30, 30)
        if (x, y) not in seen:
            seen.add((x, y))
            final.append((x, y))
    
    return final[:30]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    best_points = None
    best_score = 0
    
    for seed in range(30):
        if time.time() - start_time > DEADLINE - 15:
            break
        
        points = _make_initial_points(seed)
        if len(points) != 30 or len(set(points)) != 30:
            continue
        
        counts = _compute_all_areas(points)
        score = _score_from_counts(counts)
        
        if score > best_score:
            best_score = score
            best_points = points[:]
    
    if best_points is None:
        best_points = [(x, y) for x in range(6) for y in range(5)]
        best_score = _score_from_counts(_compute_all_areas(best_points))
    
    points = best_points[:]
    counts = _compute_all_areas(points)
    current_score = _score_from_counts(counts)
    best_found = current_score
    best_points_found = points[:]
    
    no_improve_streak = 0
    max_no_improve = 3000
    
    while time.time() - start_time < DEADLINE - 5:
        improved = False
        best_move = None
        best_move_score = current_score
        
        for idx in range(30):
            if time.time() - start_time > DEADLINE - 5:
                break
            ox, oy = points[idx]
            
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ox + dx, oy + dy
                    if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                        continue
                    new_pt = (nx, ny)
                    if any(new_pt == points[j] for j in range(30) if j != idx):
                        continue
                    
                    old_counts_snapshot = dict(counts)
                    _update_area_counts(counts, points, idx, (ox, oy), new_pt)
                    new_score = _score_from_counts(counts)
                    counts.clear()
                    counts.update(old_counts_snapshot)
                    
                    if new_score > best_move_score:
                        best_move_score = new_score
                        best_move = (idx, new_pt)
            
            if best_move_score > current_score + 10:
                break
        
        if best_move is not None:
            idx, new_pt = best_move
            _update_area_counts(counts, points, idx, points[idx], new_pt)
            points[idx] = new_pt
            current_score = best_move_score
            improved = True
            no_improve_streak = 0
            
            if current_score > best_found:
                best_found = current_score
                best_points_found = points[:]
        else:
            no_improve_streak += 1
            if no_improve_streak >= max_no_improve:
                if time.time() - start_time > DEADLINE - 10:
                    break
                for _ in range(5):
                    idx = random.randint(0, 29)
                    ox, oy = points[idx]
                    placed = False
                    for _ in range(50):
                        dx = random.randint(-4, 4)
                        dy = random.randint(-4, 4)
                        nx, ny = ox + dx, oy + dy
                        if -30 <= nx <= 30 and -30 <= ny <= 30:
                            if not any((nx, ny) == points[j] for j in range(30) if j != idx):
                                _update_area_counts(counts, points, idx, points[idx], (nx, ny))
                                points[idx] = (nx, ny)
                                placed = True
                                break
                current_score = _score_from_counts(counts)
                no_improve_streak = 0
    
    return best_points_found
# EVOLVE-BLOCK-END
