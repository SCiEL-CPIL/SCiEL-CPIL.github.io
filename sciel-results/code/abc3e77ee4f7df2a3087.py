"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from math import gcd
from collections import defaultdict

def _line_key(x1, y1, x2, y2):
    """Compute canonical line key from two points."""
    dx = x2 - x1
    dy = y2 - y1
    g = gcd(abs(dx), abs(dy))
    if g == 0:
        return None
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _compute_lines(points):
    """Compute all lines and their point index sets. Returns dict of line_key -> set of indices."""
    n = len(points)
    line_points = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i][0], points[i][1], points[j][0], points[j][1])
            if key is not None:
                line_points[key].add(i)
                line_points[key].add(j)
    return line_points

def _evaluate(points):
    """Compute score and validity. Returns (is_valid, score)."""
    n = len(points)
    if n != 16:
        return False, 0
    if len(set(tuple(p) for p in points)) != 16:
        return False, 0
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False, 0
    line_points = _compute_lines(points)
    for indices in line_points.values():
        if len(indices) >= 5:
            return False, 0
    score = sum(1 for indices in line_points.values() if len(indices) == 4)
    return True, score

def _remove_point_from_lines(line_points, idx):
    """Remove point idx from all lines in line_points."""
    for key in list(line_points.keys()):
        if idx in line_points[key]:
            line_points[key].discard(idx)
            if not line_points[key]:
                del line_points[key]

def _add_point_to_lines(line_points, idx, points):
    """Add point idx to all lines with other points."""
    x_m, y_m = points[idx]
    for j in range(len(points)):
        if j == idx:
            continue
        key = _line_key(x_m, y_m, points[j][0], points[j][1])
        if key is not None:
            line_points[key].add(idx)
            line_points[key].add(j)

def _get_score_from_lines(line_points):
    """Get score and validity from line_points dict."""
    for indices in line_points.values():
        if len(indices) >= 5:
            return False, 0
    score = sum(1 for indices in line_points.values() if len(indices) == 4)
    return True, score

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    random.seed(42)

    # Generate diverse initial configurations
    initial_configs = []

    # 4x4 grid
    initial_configs.append([[x, y] for x in range(4) for y in range(4)])

    # Various grid offsets
    for ox in range(-3, 4):
        for oy in range(-3, 4):
            initial_configs.append([[x + ox, y + oy] for x in range(4) for y in range(4)])

    # Perturbed grids - more aggressive
    for trial in range(60):
        pts = [[x, y] for x in range(4) for y in range(4)]
        num_moves = random.randint(3, 8)
        for _ in range(num_moves):
            idx = random.randint(0, 15)
            pts[idx] = [random.randint(-20, 20), random.randint(-20, 20)]
        initial_configs.append(pts)

    # Structured random with clustering
    for trial in range(80):
        pts = []
        # Create 4 clusters
        for c in range(4):
            cx = random.randint(-100, 100)
            cy = random.randint(-100, 100)
            for _ in range(4):
                pts.append([cx + random.randint(-15, 15), cy + random.randint(-15, 15)])
        initial_configs.append(pts)

    # Pure random
    for trial in range(80):
        pts = []
        for _ in range(16):
            pts.append([random.randint(-100, 100), random.randint(-100, 100)])
        initial_configs.append(pts)

    # Evaluate all initial configs, keep best
    best_points = None
    best_score = 0
    best_line_points = None

    for cfg in initial_configs:
        valid, score = _evaluate(cfg)
        if valid and score > best_score:
            best_score = score
            best_points = [p[:] for p in cfg]
            best_line_points = _compute_lines(best_points)

    if best_points is None:
        # Fallback: use 4x4 grid
        best_points = [[x, y] for x in range(4) for y in range(4)]
        valid, best_score = _evaluate(best_points)
        best_line_points = _compute_lines(best_points)

    # Phase 1: Greedy local search with small moves
    improved = True
    while improved and time.time() < deadline - 60:
        improved = False
        for i in range(16):
            if time.time() > deadline - 60:
                break
            old_x, old_y = best_points[i]

            # Try small moves
            for dx in range(-8, 9):
                for dy in range(-8, 9):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = old_x + dx
                    new_y = old_y + dy
                    if abs(new_x) > 1000 or abs(new_y) > 1000:
                        continue

                    # Check if new position collides with existing points
                    new_pos = (new_x, new_y)
                    if any((p[0], p[1]) == new_pos for p in best_points if best_points.index([p[0], p[1]]) != i):
                        continue

                    # Save old position
                    old_pt = best_points[i][:]
                    best_points[i] = [new_x, new_y]
                    _remove_point_from_lines(best_line_points, i)
                    _add_point_to_lines(best_line_points, i, best_points)
                    valid, score = _get_score_from_lines(best_line_points)

                    if valid and score > best_score:
                        best_score = score
                        improved = True
                    else:
                        # Revert
                        best_points[i] = old_pt
                        _remove_point_from_lines(best_line_points, i)
                        _add_point_to_lines(best_line_points, i, best_points)

    # Phase 2: Simulated annealing
    current_points = [p[:] for p in best_points]
    current_line_points = _compute_lines(current_points)
    current_valid, current_score = _get_score_from_lines(current_line_points)
    if not current_valid:
        current_score = 0

    best_sa_points = [p[:] for p in best_points]
    best_sa_score = best_score

    # SA parameters
    T_start = 10.0
    T_end = 0.01
    iterations = 50000

    # Calculate cooling schedule
    if iterations > 1:
        cool_rate = (T_end / T_start) ** (1.0 / iterations)
    else:
        cool_rate = 1.0

    T = T_start

    for it in range(iterations):
        if time.time() > deadline - 20:
            break

        # Choose a random point to move
        i = random.randint(0, 15)
        old_x, old_y = current_points[i]
        old_pt = current_points[i][:]

        # Generate new position with Gaussian-like distribution
        scale = 5 + 20 * T / T_start  # Scale decreases as T decreases
        new_x = int(round(old_x + random.gauss(0, scale)))
        new_y = int(round(old_y + random.gauss(0, scale)))

        # Clamp to bounds
        new_x = max(-1000, min(1000, new_x))
        new_y = max(-1000, min(1000, new_y))

        # Check for collision
        new_pos = (new_x, new_y)
        collision = False
        for j in range(16):
            if j != i and (current_points[j][0], current_points[j][1]) == new_pos:
                collision = True
                break
        if collision:
            continue

        # Apply move
        current_points[i] = [new_x, new_y]
        _remove_point_from_lines(current_line_points, i)
        _add_point_to_lines(current_line_points, i, current_points)
        new_valid, new_score = _get_score_from_lines(current_line_points)

        if not new_valid:
            new_score = -100  # Penalty for invalid

        delta = new_score - current_score

        # Acceptance criterion
        if delta > 0 or random.random() < math.exp(delta / max(T, 0.001)):
            current_score = new_score
            if new_valid and new_score > best_sa_score:
                best_sa_score = new_score
                best_sa_points = [p[:] for p in current_points]
        else:
            # Revert
            current_points[i] = old_pt
            _remove_point_from_lines(current_line_points, i)
            _add_point_to_lines(current_line_points, i, current_points)

        # Cool down
        T *= cool_rate

    # Update best if SA found better
    if best_sa_score > best_score:
        best_score = best_sa_score
        best_points = best_sa_points
        best_line_points = _compute_lines(best_points)

    # Phase 3: Final local polish
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(16):
            if time.time() > deadline - 5:
                break
            old_x, old_y = best_points[i]
            old_pt = best_points[i][:]

            # Try small moves again
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = old_x + dx
                    new_y = old_y + dy
                    if abs(new_x) > 1000 or abs(new_y) > 1000:
                        continue

                    # Check collision
                    new_pos = (new_x, new_y)
                    if any((p[0], p[1]) == new_pos for j, p in enumerate(best_points) if j != i):
                        continue

                    best_points[i] = [new_x, new_y]
                    _remove_point_from_lines(best_line_points, i)
                    _add_point_to_lines(best_line_points, i, best_points)
                    valid, score = _get_score_from_lines(best_line_points)

                    if valid and score > best_score:
                        best_score = score
                        improved = True
                    else:
                        best_points[i] = old_pt
                        _remove_point_from_lines(best_line_points, i)
                        _add_point_to_lines(best_line_points, i, best_points)

    # Final verification
    valid, final_score = _evaluate(best_points)
    if not valid:
        # Fallback
        best_points = [[x, y] for x in range(4) for y in range(4)]
        valid, final_score = _evaluate(best_points)

    return best_points
# EVOLVE-BLOCK-END
