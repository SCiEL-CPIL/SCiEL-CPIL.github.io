import time
import itertools
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s limit
    
    # Two parallel lines: 15 points on y=0, 15 on y=1
    # x = 0..14 for both
    best_points = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
    best_score = _score(best_points)
    
    # Try a few alternative configurations quickly
    alternatives = []
    
    # Try different splits
    for a in range(10, 21):
        b = 30 - a
        pts = [(x, 0) for x in range(a)] + [(x, 1) for x in range(b)]
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts
    
    # Try with h=2 to see if different areas give higher multiplicity
    for a in range(10, 21):
        b = 30 - a
        pts = [(x, 0) for x in range(a)] + [(x, 2) for x in range(b)]
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts
    
    # Quick local search: perturb points slightly
    if time.time() < deadline:
        import random
        random.seed(42)
        current = list(best_points)
        current_score = best_score
        for _ in range(200):
            if time.time() > deadline:
                break
            # Pick a random point and move it
            idx = random.randint(0, 29)
            old = current[idx]
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
            new_pt = (new_x, new_y)
            if new_pt in current:
                continue
            current[idx] = new_pt
            s = _score(current)
            if s > current_score:
                current_score = s
                best_points = list(current)
                best_score = s
            else:
                current[idx] = old
    
    return best_points

def _score(points):
    """Compute the largest multiplicity of a single positive doubled area."""
    n = len(points)
    if n < 3:
        return 0
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                det = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                if det > 0:
                    areas.append(det)
    if not areas:
        return 0
    counts = Counter(areas)
    return max(counts.values())
