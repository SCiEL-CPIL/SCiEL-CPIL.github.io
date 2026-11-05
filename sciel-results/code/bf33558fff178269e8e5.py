import time
import random
import math
from collections import defaultdict

DEADLINE = 140.0

def collinear(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)

def has_four_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            count = 2
            for k in range(n):
                if k == i or k == j:
                    continue
                if collinear(points[i], points[j], points[k]):
                    count += 1
                    if count >= 4:
                        return True
    return False

def max_no_three_collinear(points):
    n = len(points)
    coll_pairs = [set() for _ in range(n)]
    for k in range(n):
        for i in range(n):
            if i == k:
                continue
            for j in range(i + 1, n):
                if j == k:
                    continue
                if collinear(points[i], points[j], points[k]):
                    coll_pairs[k].add((min(i, j), max(i, j)))
    
    order = sorted(range(n), key=lambda k: len(coll_pairs[k]), reverse=True)
    best_val = [0]
    
    def search(pos, mask, size):
        if size + (n - pos) <= best_val[0]:
            return
        if pos == n:
            if size > best_val[0]:
                best_val[0] = size
            return
        k = order[pos]
        search(pos + 1, mask, size)
        ok = True
        for (i, j) in coll_pairs[k]:
            if (mask >> i) & 1 and (mask >> j) & 1:
                ok = False
                break
        if ok:
            search(pos + 1, mask | (1 << k), size + 1)
    
    search(0, 0, 0)
    return best_val[0]

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def generate_initial():
        for attempt in range(100):
            points = []
            used = set()
            num_lines = random.randint(8, 12)
            for _ in range(num_lines):
                if len(points) >= 19:
                    break
                if len(points) >= 2:
                    i, j = random.sample(range(len(points)), 2)
                    p1, p2 = points[i], points[j]
                else:
                    p1 = (random.randint(-80, 80), random.randint(-80, 80))
                    p2 = (random.randint(-80, 80), random.randint(-80, 80))
                    if p1 == p2:
                        continue
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                g = math.gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
                if g == 0:
                    continue
                step_x = dx // g
                step_y = dy // g
                for t in list(range(-6, 7)) + list(range(2, 8)):
                    if t in (0, 1):
                        continue
                    nx = p1[0] + t * step_x
                    ny = p1[1] + t * step_y
                    if abs(nx) > 100 or abs(ny) > 100:
                        continue
                    p3 = (nx, ny)
                    if p3 in used or p3 == p1 or p3 == p2:
                        continue
                    points.append(p3)
                    used.add(p3)
                    break
            while len(points) < 20:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                p = (x, y)
                if p not in used:
                    points.append(p)
                    used.add(p)
            seen = set()
            unique = []
            for p in points:
                if p not in seen:
                    seen.add(p)
                    unique.append(p)
            points = unique[:20]
            while len(points) < 20:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                p = (x, y)
                if p not in points:
                    points.append(p)
            if not has_four_collinear(points):
                return points
        for _ in range(300):
            points = []
            used = set()
            for _ in range(20):
                for _att in range(300):
                    x = random.randint(-100, 100)
                    y = random.randint(-100, 100)
                    p = (x, y)
                    if p in used:
                        continue
                    new_pts = points + [p]
                    if not has_four_collinear(new_pts):
                        points.append(p)
                        used.add(p)
                        break
            if len(points) == 20:
                return points
        points = [(x, x * x) for x in range(-10, 10)]
        return points
    
    points = generate_initial()
    current_score = max_no_three_collinear(points)
    best_points = points[:]
    best_score = current_score
    
    T = 5.0
    cooling = 0.9995
    
    while time_left() > 5.0:
        move_type = random.random()
        
        if move_type < 0.5:
            idx = random.randint(0, 19)
            old_x, old_y = points[idx]
            moved = False
            for _ in range(30):
                new_x = old_x + random.randint(-4, 4)
                new_y = old_y + random.randint(-4, 4)
                new_x = max(-100, min(100, new_x))
                new_y = max(-100, min(100, new_y))
                new_p = (new_x, new_y)
                if new_p in points:
                    continue
                old_p = points[idx]
                points[idx] = new_p
                if has_four_collinear(points):
                    points[idx] = old_p
                    continue
                new_score = max_no_three_collinear(points)
                if new_score < current_score or random.random() < math.exp(-(new_score - current_score) / T):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_points = points[:]
                else:
                    points[idx] = old_p
                moved = True
                break
            if not moved:
                continue
        else:
            i, j = random.sample(range(20), 2)
            pi, pj = points[i], points[j]
            if pi == pj:
                continue
            points[i], points[j] = pj, pi
            if has_four_collinear(points):
                points[i], points[j] = pi, pj
                continue
            new_score = max_no_three_collinear(points)
            if new_score < current_score or random.random() < math.exp(-(new_score - current_score) / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_points = points[:]
            else:
                points[i], points[j] = pi, pj
        
        T *= cooling
        if T < 0.01:
            T = 0.01
    
    return [[p[0], p[1]] for p in best_points]
