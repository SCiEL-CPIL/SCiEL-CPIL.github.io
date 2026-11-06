import random
import time
import math
from collections import defaultdict
from itertools import combinations

def count_ordinary_lines(points):
    n = len(points)
    if n < 3:
        return 0
    lines = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            if dx < 0:
                dx = -dx
                dy = -dy
            elif dx == 0 and dy < 0:
                dy = -dy

            a = dy
            b = -dx
            c = a * p1[0] + b * p1[1]
            key = (a, b, c)
            lines[key].add(i)
            lines[key].add(j)

    ordinary = 0
    for pts in lines.values():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def _count_ordinary_fast(points):
    """Faster counting using tuple keys and early exit if we can't beat best."""
    n = len(points)
    if n < 3:
        return 0
    lines = {}
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            if dx < 0:
                dx = -dx
                dy = -dy
            elif dx == 0 and dy < 0:
                dy = -dy

            a = dy
            b = -dx
            c = a * xi + b * yi
            key = (a, b, c)
            if key in lines:
                lines[key].add(i)
                lines[key].add(j)
            else:
                lines[key] = {i, j}

    ordinary = 0
    for pts in lines.values():
        if len(pts) == 2:
            ordinary += 1
    return ordinary


def _make_initial_configs():
    configs = []

    # 3x4 grid (baseline)
    configs.append([[x, y] for x in range(4) for y in range(3)])

    # 2x6 grid
    configs.append([[x, y] for x in range(6) for y in range(2)])

    # 6x2 grid
    configs.append([[x, y] for x in range(2) for y in range(6)])

    # Diagonal-shift grid: y in 0..3, x in 0..2, point = (x+y, y)
    pts_diag = []
    for y in range(4):
        for x in range(3):
            pts_diag.append([x + y, y])
    configs.append(pts_diag)

    # 4x4 minus corners
    pts = [[x, y] for x in range(4) for y in range(4)]
    for corner in [[0, 0], [3, 0], [0, 3], [3, 3]]:
        pts.remove(corner)
    configs.append(pts)

    # 5x3 grid (first 12)
    configs.append([[x, y] for x in range(5) for y in range(3)][:12])

    # Hexagonal-like: two rows with offset
    pts_hex = []
    for x in range(6):
        pts_hex.append([2 * x, 0])
    for x in range(6):
        pts_hex.append([2 * x + 1, 2])
    configs.append(pts_hex)

    # Regular hexagon vertices + doubled vertices
    R = 10**6
    hex_pts = [
        [R, 0],
        [R // 2, int(R * 0.866025)],
        [-R // 2, int(R * 0.866025)],
        [-R, 0],
        [-R // 2, -int(R * 0.866025)],
        [R // 2, -int(R * 0.866025)],
    ]
    hex_pts2 = list(hex_pts)
    for p in hex_pts:
        hex_pts2.append([2 * p[0], 2 * p[1]])
    configs.append(hex_pts2)

    # 3 points on each of 4 lines through a common point (near-pencil variant)
    near_pencil = []
    for d in [1, 2, 3]:
        near_pencil.append([d, 0])
        near_pencil.append([d, d])
        near_pencil.append([-d, -d])
        near_pencil.append([0, d])
    configs.append(near_pencil)

    # Random structured configs
    rng = random.Random(12345)
    for _ in range(20):
        pts = []
        for _ in range(12):
            pts.append([rng.randint(-8, 8), rng.randint(-8, 8)])
        if len(set(map(tuple, pts))) == 12:
            configs.append(pts)

    # Additional structured configurations from inspirations
    # 3 points on y=0, 3 on y=1, 3 on y=2, 3 on y=3, with offsets to create diagonals
    pts_offset = []
    for y in range(4):
        for x in range(3):
            pts_offset.append([x + y, y])
    configs.append(pts_offset)

    # Another variant: staggered rows
    pts_stag = []
    for y in range(4):
        offset = y % 2
        for x in range(3):
            pts_stag.append([x + offset * 2, y])
    configs.append(pts_stag)

    # NEW: 3 points on each of 4 parallel lines (vertical strips)
    # x=0: y=0,1,2; x=1: y=0,1,2; x=2: y=0,1,2; x=3: y=0,1,2
    # This creates many 3-point vertical lines
    configs.append([[x, y] for x in range(4) for y in range(3)])

    # NEW: 4 points on each of 3 horizontal lines
    configs.append([[x, y] for x in range(4) for y in range(3)])

    # NEW: 3 points on 4 lines forming a complete quadrilateral
    # Lines: y=0, y=1, y=2, y=3 with points placed to create diagonal 3-point lines
    pts_quad = []
    for y in range(4):
        for x in range(3):
            pts_quad.append([x, y])
    configs.append(pts_quad)

    # NEW: 12 points on 4 lines of 3 each, arranged as a parallelogram grid
    # (0,0),(1,0),(2,0),(0,2),(1,2),(2,2),(0,4),(1,4),(2,4),(0,6),(1,6),(2,6)
    configs.append([[x, 2*y] for y in range(4) for x in range(3)])

    # NEW: 3x4 grid with one point moved to create extra collinearities
    base = [[x, y] for x in range(4) for y in range(3)]
    variants = []
    # Move corner points to create diagonal 3-point lines
    for move_idx, (mx, my) in [(0, (3, 1)), (11, (0, 1)), (3, (1, 2)), (8, (2, 2))]:
        v = [p[:] for p in base]
        v[move_idx] = [mx, my]
        if len(set(map(tuple, v))) == 12:
            variants.append(v)
    configs.extend(variants)

    # Filter valid configs
    valid_configs = []
    for cfg in configs:
        if len(cfg) != 12:
            continue
        if len(set(map(tuple, cfg))) < 12:
            continue
        if any(abs(p[0]) > 10**9 or abs(p[1]) > 10**9 for p in cfg):
            continue
        # Check not all collinear
        (x0, y0), (x1, y1), (x2, y2) = cfg[0], cfg[1], cfg[2]
        all_col = True
        for i in range(3, 12):
            xi, yi = cfg[i]
            if (x1 - x0) * (yi - y0) != (y1 - y0) * (xi - x0):
                all_col = False
                break
        if all_col:
            continue
        valid_configs.append(cfg)

    return valid_configs


def _find_collinear_targets(points):
    """Find positions where a new point would be collinear with 2 existing points."""
    n = len(points)
    targets = []
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            # Points on the line: p1 + k*(dx, dy) for integer k
            for k in [-3, -2, -1, 1, 2, 3]:
                nx = p1[0] + k * dx
                ny = p1[1] + k * dy
                if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                    if [nx, ny] not in points:
                        targets.append([nx, ny])
    return targets


def _evaluate_and_update(points, best_points, best_score):
    """Evaluate a configuration and update best if improved."""
    if len(set(map(tuple, points))) < 12:
        return best_points, best_score
    if any(abs(p[0]) > 10**9 or abs(p[1]) > 10**9 for p in points):
        return best_points, best_score
    
    score = _count_ordinary_fast(points)
    if score < best_score:
        return [p[:] for p in points], score
    return best_points, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin (evaluator kills at 150s)

    initial_configs = _make_initial_configs()

    best_points = None
    best_score = float('inf')

    for cfg in initial_configs:
        score = _count_ordinary_fast(cfg)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in cfg]
        if best_score == 6:
            break

    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(3)]
        best_score = _count_ordinary_fast(best_points)

    # Local search with multiple strategies
    rng = random.Random(42)
    no_improve_count = 0
    max_no_improve = 3000  # Lower threshold for faster restarts
    iter_count = 0

    while time.time() < deadline:
        iter_count += 1
        strategy = rng.random()

        if strategy < 0.30:
            # Single point random move
            idx = rng.randint(0, 11)
            old_p = best_points[idx][:]

            step_size = rng.choice([1, 2, 3, 5, 8, 12, 16, 24, 32, 48])
            dx = rng.randint(-step_size, step_size)
            dy = rng.randint(-step_size, step_size)
            if dx == 0 and dy == 0:
                continue

            new_p = [old_p[0] + dx, old_p[1] + dy]

            new_points = [p[:] for p in best_points]
            new_points[idx] = new_p
            best_points, best_score = _evaluate_and_update(new_points, best_points, best_score)
            
            if best_score == 6:
                break
            no_improve_count += 1

        elif strategy < 0.55:
            # Move a point to a collinear target position
            targets = _find_collinear_targets(best_points)
            if targets:
                idx = rng.randint(0, 11)
                target = rng.choice(targets)
                if target == best_points[idx]:
                    continue

                new_points = [p[:] for p in best_points]
                new_points[idx] = target
                best_points, best_score = _evaluate_and_update(new_points, best_points, best_score)
                
                if best_score == 6:
                    break
                no_improve_count += 1

        elif strategy < 0.75:
            # Try to create a new 3-point line by moving one point
            # Pick two random points, find a third point that would be collinear
            i, j = rng.sample(range(12), 2)
            p1 = best_points[i]
            p2 = best_points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx_n = dx // g
            dy_n = dy // g
            
            # Try to move a third point to be collinear with p1, p2
            k = rng.randint(0, 11)
            if k == i or k == j:
                continue
            
            # Find a position on the line p1-p2 that's not already occupied
            found = False
            for t in range(-6, 7):
                if t == 0:
                    continue
                nx = p1[0] + t * dx_n
                ny = p1[1] + t * dy_n
                if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                    if [nx, ny] not in best_points:
                        new_points = [p[:] for p in best_points]
                        new_points[k] = [nx, ny]
                        best_points, best_score = _evaluate_and_update(new_points, best_points, best_score)
                        found = True
                        break
            
            if best_score == 6:
                break
            no_improve_count += 1

        elif strategy < 0.85:
            # Swap two points' positions (perturb both)
            i, j = rng.sample(range(12), 2)
            p1 = best_points[i][:]
            p2 = best_points[j][:]

            step_size = rng.choice([1, 2, 3, 5, 8, 12])
            dx1 = rng.randint(-step_size, step_size)
            dy1 = rng.randint(-step_size, step_size)
            dx2 = rng.randint(-step_size, step_size)
            dy2 = rng.randint(-step_size, step_size)

            new_p1 = [p1[0] + dx1, p1[1] + dy1]
            new_p2 = [p2[0] + dx2, p2[1] + dy2]

            new_points = [p[:] for p in best_points]
            new_points[i] = new_p1
            new_points[j] = new_p2
            best_points, best_score = _evaluate_and_update(new_points, best_points, best_score)
            
            if best_score == 6:
                break
            no_improve_count += 1

        else:
            # Occasional restart from a good initial config
            if time.time() < deadline - 10:
                cfg = rng.choice(initial_configs)
                s = _count_ordinary_fast(cfg)
                if s < best_score:
                    best_score = s
                    best_points = [p[:] for p in cfg]
                    no_improve_count = 0
                else:
                    no_improve_count += 1

        # If stuck for too long, try a more aggressive restart
        if no_improve_count > max_no_improve:
            cfg = rng.choice(initial_configs)
            s = _count_ordinary_fast(cfg)
            if s <= best_score:
                best_score = s
                best_points = [p[:] for p in cfg]
            no_improve_count = 0

    return best_points
