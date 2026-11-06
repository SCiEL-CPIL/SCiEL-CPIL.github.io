"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dx * y1 - dy * x1
    return (dx, dy, c)

def _evaluate(points):
    line_map = defaultdict(set)
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            line_map[key].add(i)
    count4 = 0
    for key, pts in line_map.items():
        if len(pts) >= 5:
            return -1
        if len(pts) == 4:
            count4 += 1
    return count4

def _make_grid4x4():
    return [[x, y] for x in range(-2, 2) for y in range(-2, 2)]

def _make_grid5x5_subset():
    all_pts = [[x, y] for x in range(-2, 3) for y in range(-2, 3)]
    random.shuffle(all_pts)
    return [list(p) for p in all_pts[:16]]

def _make_random():
    pts = []
    seen = set()
    while len(pts) < 16:
        p = [random.randint(-100, 100), random.randint(-100, 100)]
        if (p[0], p[1]) not in seen:
            seen.add((p[0], p[1]))
            pts.append(p)
    return pts

def construct():
    deadline = time.time() + 138
    
    best_pts = None
    best_score = -1
    
    inits = [_make_grid4x4(), _make_grid5x5_subset(), _make_random()]
    for _ in range(2):
        inits.append(_make_random())
    
    for init in inits:
        if time.time() > deadline:
            break
        s = _evaluate(init)
        if s > best_score:
            best_score = s
            best_pts = [list(p) for p in init]
    
    if best_pts is None:
        best_pts = _make_grid4x4()
        best_score = _evaluate(best_pts)
    
    cur = [list(p) for p in best_pts]
    cur_score = best_score
    
    T = 100.0
    cool = 0.9997
    min_T = 0.5
    
    while time.time() < deadline:
        if T < min_T:
            break
        
        r = random.random()
        new_pts = [list(p) for p in cur]
        
        if r < 0.5:
            idx = random.randrange(16)
            step = random.randint(1, 4)
            dx = random.choice([-step, step])
            dy = random.choice([-step, step])
            new_pts[idx][0] += dx
            new_pts[idx][1] += dy
            if abs(new_pts[idx][0]) > 1000 or abs(new_pts[idx][1]) > 1000:
                continue
            if len(set(map(tuple, new_pts))) < 16:
                continue
        elif r < 0.7:
            idx = random.randrange(16)
            new_pts[idx] = [
                max(-1000, min(1000, cur[idx][0] + random.randint(-8, 8))),
                max(-1000, min(1000, cur[idx][1] + random.randint(-8, 8)))
            ]
            if len(set(map(tuple, new_pts))) < 16:
                continue
        elif r < 0.85:
            i, j = random.sample(range(16), 2)
            new_pts[i], new_pts[j] = new_pts[j], new_pts[i]
        else:
            # Snap to nearby grid point
            idx = random.randrange(16)
            gx = round(cur[idx][0] / 2.0) * 2
            gy = round(cur[idx][1] / 2.0) * 2
            new_pts[idx] = [max(-1000, min(1000, gx)), max(-1000, min(1000, gy))]
            if len(set(map(tuple, new_pts))) < 16:
                continue
        
        ns = _evaluate(new_pts)
        
        if ns >= 0:
            if ns > cur_score:
                cur = new_pts
                cur_score = ns
                if cur_score > best_score:
                    best_score = cur_score
                    best_pts = [list(p) for p in cur]
            elif ns == cur_score and random.random() < 0.3:
                cur = new_pts
                cur_score = ns
            elif random.random() < 0.05:
                cur = new_pts
                cur_score = ns
        else:
            if random.random() < 0.02:
                cur = new_pts
                cur_score = ns
        
        T *= cool
    
    return best_pts
# EVOLVE-BLOCK-END
