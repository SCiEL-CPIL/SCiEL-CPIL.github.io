import time
import random
import itertools
from collections import Counter

def _doubled_areas(points):
    n = len(points)
    counts = Counter()
    pts = [(p[0], p[1]) for p in points]
    for i in range(n - 2):
        x1, y1 = pts[i]
        for j in range(i + 1, n - 1):
            x2, y2 = pts[j]
            dx1 = x2 - x1
            dy1 = y2 - y1
            for k in range(j + 1, n):
                x3, y3 = pts[k]
                det = dx1 * (y3 - y1) - dy1 * (x3 - x1)
                area = abs(det)
                if area > 0:
                    counts[area] += 1
    return counts

def _score(points):
    counts = _doubled_areas(points)
    if not counts:
        return 0
    return max(counts.values())

def _generate_grid(w, h, ox=0, oy=0, shear=0):
    pts = []
    for i in range(w):
        for j in range(h):
            x = i + shear * j + ox
            y = j + oy
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.append((x, y))
    return pts

def construct():
    random.seed(42)
    start_time = time.time()
    deadline = start_time + 85.0
    
    candidates = []
    
    # 1. Standard rectangular grids
    for w, h in [(1, 30), (2, 15), (3, 10), (5, 6), (6, 5), (10, 3), (15, 2), (30, 1)]:
        ox = -w // 2
        oy = -h // 2
        pts = _generate_grid(w, h, ox, oy)
        if len(pts) == 30:
            candidates.append(pts)
            
    # 2. Sheared grids
    for w, h in [(5, 6), (6, 5), (3, 10), (10, 3)]:
        for s in range(-4, 5):
            if s == 0: continue
            pts = []
            valid = True
            for i in range(w):
                for j in range(h):
                    x = i + s * j
                    y = j
                    x += -w // 2 - s * (h // 2)
                    y += -h // 2
                    if -30 <= x <= 30 and -30 <= y <= 30:
                        pts.append((x, y))
                    else:
                        valid = False
                        break
                if not valid: break
            if len(pts) == 30 and valid:
                candidates.append(pts)

    # 3. Random configurations
    for _ in range(10):
        pts = []
        while len(pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x, y) not in pts:
                pts.append((x, y))
        candidates.append(pts)

    best_points = candidates[0]
    best_score = _score(best_points)
    
    # Sort candidates
    candidates_with_scores = [(_score(c), c) for c in candidates]
    candidates_with_scores.sort(key=lambda x: x[0], reverse=True)
    
    def local_search(points, max_time):
        current_points = points[:]
        current_score = _score(current_points)
        current_set = set(current_points)
        
        improved = True
        while improved and time.time() < max_time:
            improved = False
            indices = list(range(30))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > max_time: break
                ox, oy = current_points[i]
                
                for _ in range(30):
                    dx = random.randint(-5, 5)
                    dy = random.randint(-5, 5)
                    nx = ox + dx
                    ny = oy + dy
                    if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                        continue
                    new_pt = (nx, ny)
                    if new_pt == (ox, oy):
                        continue
                    if new_pt in current_set:
                        continue
                    
                    new_points = current_points[:]
                    new_points[i] = new_pt
                    new_set = current_set.copy()
                    new_set.discard((ox, oy))
                    new_set.add(new_pt)
                    
                    new_score = _score(new_points)
                    
                    if new_score > current_score:
                        current_points = new_points
                        current_set = new_set
                        current_score = new_score
                        improved = True
                        break
        
        return current_points, current_score

    # Run local search on the best seed
    best_seed, _ = candidates_with_scores[0]
    final_points, final_score = local_search(best_seed, deadline)
    
    return final_points
