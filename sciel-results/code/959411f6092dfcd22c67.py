import math
import random
import time
from collections import defaultdict

def _count_exact_four(points):
    n = len(points)
    line_points = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            g = math.gcd(abs(dx), abs(dy))
            dx_n = dx // g
            dy_n = dy // g
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            a = dy_n
            b = -dx_n
            c = a * x1 + b * y1
            g2 = math.gcd(abs(a), abs(b))
            if g2 != 0:
                a //= g2
                b //= g2
                c //= g2
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
                c = -c
            key = (a, b, c)
            line_points[key].add(i)
            line_points[key].add(j)

    exact_four = 0
    has_five = False
    for key, idx_set in line_points.items():
        cnt = len(idx_set)
        if cnt == 4:
            exact_four += 1
        elif cnt >= 5:
            has_five = True
    return exact_four, has_five

def _score(points):
    exact_four, has_five = _count_exact_four(points)
    if has_five:
        return exact_four - 1000
    return exact_four

def construct():
    start_time = time.time()
    deadline = start_time + 140

    points = [[x, y] for x in range(4) for y in range(4)]
    best_points = [p[:] for p in points]
    best_score = _score(best_points)

    random.seed(42)
    T = 5.0
    T_min = 0.05
    cooling = 0.9998

    current_score = best_score
    current_points = [p[:] for p in points]

    while time.time() < deadline:
        move_type = random.random()
        old_point = None
        old_idx = None

        if move_type < 0.5:
            idx = random.randint(0, 15)
            old_x, old_y = current_points[idx]
            new_x = old_x + random.randint(-20, 20)
            new_y = old_y + random.randint(-20, 20)
            new_x = max(-1000, min(1000, new_x))
            new_y = max(-1000, min(1000, new_y))
            if [new_x, new_y] in current_points:
                continue
            old_idx = idx
            old_point = [old_x, old_y]
            current_points[idx] = [new_x, new_y]
        elif move_type < 0.8:
            i, j = random.sample(range(16), 2)
            x1, y1 = current_points[i]
            x2, y2 = current_points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            g = math.gcd(abs(dx), abs(dy))
            dx_n = dx // g
            dy_n = dy // g
            if dx_n != 0:
                if dx_n > 0:
                    t_min_x = math.ceil((-1000 - x1) / dx_n)
                    t_max_x = math.floor((1000 - x1) / dx_n)
                else:
                    t_min_x = math.ceil((1000 - x1) / dx_n)
                    t_max_x = math.floor((-1000 - x1) / dx_n)
            else:
                t_min_x, t_max_x = -1000000, 1000000
            if dy_n != 0:
                if dy_n > 0:
                    t_min_y = math.ceil((-1000 - y1) / dy_n)
                    t_max_y = math.floor((1000 - y1) / dy_n)
                else:
                    t_min_y = math.ceil((1000 - y1) / dy_n)
                    t_max_y = math.floor((-1000 - y1) / dy_n)
            else:
                t_min_y, t_max_y = -1000000, 1000000
            t_lo = max(t_min_x, t_min_y, -1000000)
            t_hi = min(t_max_x, t_max_y, 1000000)
            if t_lo > t_hi:
                continue
            placed = False
            for _ in range(100):
                t = random.randint(t_lo, t_hi)
                nx = x1 + t * dx_n
                ny = y1 + t * dy_n
                if [nx, ny] in current_points:
                    continue
                candidates = [k for k in range(16) if k not in (i, j)]
                idx = random.choice(candidates)
                old_x, old_y = current_points[idx]
                old_idx = idx
                old_point = [old_x, old_y]
                current_points[idx] = [nx, ny]
                placed = True
                break
            if not placed:
                continue
        else:
            idx = random.randint(0, 15)
            new_x = random.randint(-1000, 1000)
            new_y = random.randint(-1000, 1000)
            if [new_x, new_y] in current_points:
                continue
            old_idx = idx
            old_point = [current_points[idx][0], current_points[idx][1]]
            current_points[idx] = [new_x, new_y]

        new_score = _score(current_points)

        if new_score >= current_score or (T > T_min and random.random() < math.exp((new_score - current_score) / T)):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
        else:
            current_points[old_idx] = old_point

        T = max(T * cooling, T_min)

    return best_points
