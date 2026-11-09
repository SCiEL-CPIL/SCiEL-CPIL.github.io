import time
import sys
import math
from math import gcd
from collections import defaultdict
import random

def _count_ordinary_lines(points):
    """Count the number of lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 3:
        return 0
    
    line_map = defaultdict(set)
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            a = dy
            b = -dx
            c = a * x1 + b * y1
            g2 = gcd(abs(a), abs(b))
            if g2 > 0:
                a //= g2
                b //= g2
                c //= g2
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
                c = -c
            line_map[(a, b, c)].add(i)
            line_map[(a, b, c)].add(j)
    
    ordinary = 0
    for line, pts in line_map.items():
        if len(pts) == 2:
            ordinary += 1
    
    return ordinary

def _try_config(points):
    """Validate and count ordinary lines for a configuration."""
    if len(points) != 12:
        return None
    if len(set(tuple(p) for p in points)) != 12:
        return None
    for p in points:
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return None
    collinear = True
    for i in range(1, 12):
        dx = points[i][0] - points[0][0]
        dy = points[i][1] - points[0][1]
        for j in range(i + 1, 12):
            dx2 = points[j][0] - points[0][0]
            dy2 = points[j][1] - points[0][1]
            if dx * dy2 != dy * dx2:
                collinear = False
                break
        if not collinear:
            break
    if collinear:
        return None
    return _count_ordinary_lines(points)

def construct():
    """
    Constructs 12 distinct integer points minimizing ordinary lines.
    Uses a combination of curated configurations, systematic search over
    structured families, and local optimization.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_points = None
    best_count = float('inf')
    
    def update_best(pts):
        nonlocal best_points, best_count
        if pts is None:
            return
        count = _try_config(pts)
        if count is not None and count < best_count:
            best_count = count
            best_points = [list(p) for p in pts]
    
    # Phase 1: Try many structured configurations
    
    # 4x3 grid
    pts = [[x, y] for x in range(4) for y in range(3)]
    update_best(pts)
    
    # 3x4 grid
    pts = [[x, y] for x in range(3) for y in range(4)]
    update_best(pts)
    
    # 2x6 grid
    pts = [[x, y] for x in range(2) for y in range(6)]
    update_best(pts)
    
    # 6x2 grid
    pts = [[x, y] for x in range(6) for y in range(2)]
    update_best(pts)
    
    # 12 on a line (all collinear - invalid, but check)
    # 11 on line + 1 off
    for off_x in range(-2, 3):
        for off_y in range(1, 4):
            pts = [[i, 0] for i in range(11)] + [[off_x, off_y]]
            update_best(pts)
    
    # 10 on line + 2 off
    for off1_x in range(-1, 3):
        for off1_y in range(1, 4):
            for off2_x in range(off1_x, 4):
                for off2_y in range(1, 4):
                    if off1_x == off2_x and off1_y == off2_y:
                        continue
                    pts = [[i, 0] for i in range(10)] + [[off1_x, off1_y], [off2_x, off2_y]]
                    update_best(pts)
    
    # 9 on line + 3 off
    for off1_x in range(0, 3):
        for off1_y in range(1, 3):
            for off2_x in range(off1_x, 4):
                for off2_y in range(1, 3):
                    for off3_x in range(off2_x, 5):
                        for off3_y in range(1, 3):
                            if (off1_x, off1_y) == (off2_x, off2_y) or \
                               (off1_x, off1_y) == (off3_x, off3_y) or \
                               (off2_x, off2_y) == (off3_x, off3_y):
                                continue
                            pts = [[i, 0] for i in range(9)] + \
                                  [[off1_x, off1_y], [off2_x, off2_y], [off3_x, off3_y]]
                            update_best(pts)
    
    # Two parallel lines with 6 each
    for shift in range(-3, 4):
        for dy in [1, 2, 3]:
            pts = [[i, 0] for i in range(6)] + [[i + shift, dy] for i in range(6)]
            update_best(pts)
    
    # Two parallel lines with 5 each + 2 on third line
    for shift in range(-2, 3):
        for dy in [1, 2]:
            for extra_x in range(-1, 3):
                for extra_y in [3, 4]:
                    pts = [[i, 0] for i in range(5)] + \
                          [[i + shift, dy] for i in range(5)] + \
                          [[extra_x, extra_y], [extra_x + 1, extra_y]]
                    update_best(pts)
    
    # Three concurrent lines through origin
    # 4 points on each of 3 lines
    slopes = [(0, 1), (1, 1), (2, 1), (1, 2), (3, 1), (1, 3), (2, 3), (3, 2), (4, 1), (1, 4)]
    for s1 in range(len(slopes)):
        for s2 in range(s1 + 1, len(slopes)):
            for s3 in range(s2 + 1, len(slopes)):
                dy1, dx1 = slopes[s1]
                dy2, dx2 = slopes[s2]
                dy3, dx3 = slopes[s3]
                pts = [[0, 0]]
                for t in range(1, 4):
                    pts.append([t * dx1, t * dy1])
                for t in range(1, 4):
                    pts.append([t * dx2, t * dy2])
                for t in range(1, 4):
                    pts.append([t * dx3, t * dy3])
                if len(pts) == 12:
                    update_best(pts)
    
    # 4 points on x-axis, 4 on y-axis, 4 on diagonal
    for t1 in range(1, 4):
        for t2 in range(1, 4):
            for t3 in range(1, 4):
                pts = [[t1, 0], [t1*2, 0], [t1*3, 0], [t1*4, 0],
                       [0, t2], [0, t2*2], [0, t2*3], [0, t2*4],
                       [t3, t3], [t3*2, t3*2], [t3*3, t3*3], [t3*4, t3*4]]
                update_best(pts)
    
    # Points on a circle (unit circle scaled)
    # Use rational points on circle
    for r in range(2, 10):
        pts = []
        for a in range(-r, r + 1):
            b2 = r * r - a * a
            if b2 >= 0:
                b = int(math.isqrt(b2))
                if b * b == b2:
                    if b != 0:
                        pts.append([a, b])
                    if a != 0 and b != 0:
                        pts.append([a, -b])
                    if a != 0:
                        pts.append([-a, b])
                    if a != 0 and b != 0:
                        pts.append([-a, -b])
        if len(pts) >= 12:
            # Take first 12 distinct
            seen = set()
            unique_pts = []
            for p in pts:
                if tuple(p) not in seen:
                    seen.add(tuple(p))
                    unique_pts.append(p)
            if len(unique_pts) >= 12:
                update_best(unique_pts[:12])
    
    # Points on a parabola
    for x0 in range(-3, 4):
        for x1 in range(x0 + 1, x0 + 12):
            pts = [[x, (x - x0) * (x - x0)] for x in range(x0, x0 + 12)]
            update_best(pts)
    
    # Points on a hyperbola xy = k
    for k in range(1, 8):
        pts = []
        for x in range(1, k + 1):
            if k % x == 0:
                y = k // x
                pts.append([x, y])
                pts.append([x, -y])
                pts.append([-x, y])
                pts.append([-x, -y])
        if len(pts) >= 12:
            seen = set()
            unique_pts = []
            for p in pts:
                if tuple(p) not in seen:
                    seen.add(tuple(p))
                    unique_pts.append(p)
            if len(unique_pts) >= 12:
                update_best(unique_pts[:12])
    
    # 3 lines, 4 points each, not concurrent
    # Line 1: y = 0, x = 0,1,2,3
    # Line 2: y = 1, x = 0,1,2,3
    # Line 3: y = 2, x = 0,1,2,3
    # This is just a 4x3 grid, already tried
    
    # Try some random configurations
    random.seed(42)
    for _ in range(2000):
        if time.time() > deadline - 30:
            break
        pts = []
        for _ in range(12):
            x = random.randint(-10, 10)
            y = random.randint(-10, 10)
            pts.append([x, y])
        update_best(pts)
    
    # Phase 2: Local search
    if best_points is not None and time.time() < deadline - 20:
        improved = True
        while improved and time.time() < deadline - 20:
            improved = False
            for i in range(12):
                if time.time() > deadline - 20:
                    break
                orig_x, orig_y = best_points[i]
                for dx in range(-5, 6):
                    for dy in range(-5, 6):
                        if dx == 0 and dy == 0:
                            continue
                        new_pt = [orig_x + dx, orig_y + dy]
                        if abs(new_pt[0]) > 10**9 or abs(new_pt[1]) > 10**9:
                            continue
                        if any(tuple(new_pt) == tuple(best_points[j]) for j in range(12) if j != i):
                            continue
                        trial = [list(p) for p in best_points]
                        trial[i] = new_pt
                        count = _try_config(trial)
                        if count is not None and count < best_count:
                            best_points = trial
                            best_count = count
                            improved = True
                            break
                    if improved:
                        break
    
    # Phase 3: Simulated annealing
    if best_points is not None and time.time() < deadline - 5:
        current_points = [list(p) for p in best_points]
        current_count = best_count
        T = 5.0
        T_min = 0.1
        cooling = 0.998
        max_iter = 3000
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            i = random.randint(0, 11)
            orig_x, orig_y = current_points[i]
            new_pt = [orig_x + random.randint(-10, 10), orig_y + random.randint(-10, 10)]
            
            if abs(new_pt[0]) > 10**9 or abs(new_pt[1]) > 10**9:
                continue
            if any(tuple(new_pt) == tuple(current_points[j]) for j in range(12) if j != i):
                continue
            
            trial = [list(p) for p in current_points]
            trial[i] = new_pt
            count = _try_config(trial)
            if count is None:
                continue
            
            delta = count - current_count
            if delta < 0:
                current_points = trial
                current_count = count
                if current_count < best_count:
                    best_points = [list(p) for p in current_points]
                    best_count = current_count
            else:
                if random.random() < math.exp(-delta / T):
                    current_points = trial
                    current_count = count
            
            T = max(T_min, T * cooling)
    
    # Phase 4: Final polish
    if best_points is not None and time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(12):
                if time.time() > deadline - 2:
                    break
                orig_x, orig_y = best_points[i]
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        if dx == 0 and dy == 0:
                            continue
                        new_pt = [orig_x + dx, orig_y + dy]
                        if abs(new_pt[0]) > 10**9 or abs(new_pt[1]) > 10**9:
                            continue
                        if any(tuple(new_pt) == tuple(best_points[j]) for j in range(12) if j != i):
                            continue
                        trial = [list(p) for p in best_points]
                        trial[i] = new_pt
                        count = _try_config(trial)
                        if count is not None and count < best_count:
                            best_points = trial
                            best_count = count
                            improved = True
                            break
                    if improved:
                        break
    
    return best_points
