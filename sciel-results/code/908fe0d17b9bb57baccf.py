import time
import random
from math import gcd

def _line_key(x1, y1, x2, y2):
    """Canonical key for the line through two integer points."""
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
    # c such that dx*x + dy*y + c = 0, normalized
    c = -(dx * x1 + dy * y1)
    g2 = gcd(gcd(abs(dx), abs(dy)), abs(c))
    if g2 > 0:
        dx //= g2
        dy //= g2
        c //= g2
    return (dx, dy, c)

def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 3:
        return 0
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i][0], points[i][1], points[j][0], points[j][1])
            if key is not None:
                line_counts[key] = line_counts.get(key, 0) + 1
    return sum(1 for v in line_counts.values() if v == 1)

def _not_all_collinear(points):
    if len(points) < 3:
        return True
    x1, y1 = points[0]
    x2, y2 = points[1]
    dx, dy = x2 - x1, y2 - y1
    for i in range(2, len(points)):
        if dx * (points[i][1] - y1) != dy * (points[i][0] - x1):
            return True
    return False

def _build_line_data(points):
    """Build line_counts (line -> number of points on it) and point_lines (point_idx -> set of line keys)."""
    n = len(points)
    line_counts = {}
    point_lines = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i][0], points[i][1], points[j][0], points[j][1])
            if key is None:
                continue
            line_counts[key] = line_counts.get(key, 0) + 1
            point_lines[i].add(key)
            point_lines[j].add(key)
    return line_counts, point_lines

def _ordinary_from_counts(line_counts):
    return sum(1 for v in line_counts.values() if v == 1)

def _try_move(points, idx, new_x, new_y, line_counts, point_lines, limit=10**9):
    """Try moving point idx to (new_x, new_y). Returns (new_score, new_line_counts, new_point_lines) or None."""
    n = len(points)
    old_x, old_y = points[idx]
    if new_x == old_x and new_y == old_y:
        return None
    if abs(new_x) > limit or abs(new_y) > limit:
        return None
    # Check duplicate
    for j in range(n):
        if j != idx and points[j][0] == new_x and points[j][1] == new_y:
            return None

    # Update line_counts and point_lines
    new_line_counts = dict(line_counts)
    new_point_lines = [set(s) for s in point_lines]

    # Remove old point from its lines
    for key in list(point_lines[idx]):
        cnt = new_line_counts.get(key, 0)
        if cnt <= 1:
            del new_line_counts[key]
        else:
            new_line_counts[key] = cnt - 1
        # Remove from other points on this line
        for j in range(n):
            if j != idx and key in new_point_lines[j]:
                new_point_lines[j].discard(key)

    # Add new point to its lines
    for j in range(n):
        if j == idx:
            continue
        key = _line_key(new_x, new_y, points[j][0], points[j][1])
        if key is None:
            continue
        cnt = new_line_counts.get(key, 0)
        if cnt == 0:
            new_line_counts[key] = 1
        else:
            new_line_counts[key] = cnt + 1
        new_point_lines[idx].add(key)
        new_point_lines[j].add(key)

    # Check not all collinear
    pts_check = [p[:] for p in points]
    pts_check[idx] = [new_x, new_y]
    if not _not_all_collinear(pts_check):
        return None

    score = _ordinary_from_counts(new_line_counts)
    return score, new_line_counts, new_point_lines

def _local_search(points, deadline, verbose=False):
    """Fast local search with incremental line data."""
    current = [p[:] for p in points]
    line_counts, point_lines = _build_line_data(current)
    current_score = _ordinary_from_counts(line_counts)
    best = [p[:] for p in current]
    best_score = current_score
    best_lc = dict(line_counts)
    best_pl = [set(s) for s in point_lines]

    n = len(current)
    no_improve_count = 0

    while time.time() < deadline:
        improved = False
        no_improve_count += 1

        # Phase 1: targeted "join a 2-point line" moves
        # For each line with exactly 2 points, try to move a third point onto it
        two_point_lines = [k for k, v in line_counts.items() if v == 2]
        random.shuffle(two_point_lines)

        for line_key in two_point_lines:
            if time.time() > deadline:
                break
            dx, dy, c = line_key
            # Find the two points on this line
            pts_on_line = []
            for i in range(n):
                if line_key in point_lines[i]:
                    pts_on_line.append(i)
            if len(pts_on_line) != 2:
                continue
            i1, i2 = pts_on_line

            # Try moving each other point onto this line
            for idx in range(n):
                if idx == i1 or idx == i2:
                    continue
                ox, oy = current[idx]
                # Find a point on the line (dx,dy,c) near (ox,oy)
                # Line: dx*x + dy*y + c = 0
                # We want integer (nx, ny) near (ox, oy) on this line
                # Parametrize: if dx != 0, x = t, y = (-c - dx*t)/dy (if dy != 0)
                # Better: find integer solutions to dx*nx + dy*ny = -c near (ox,oy)
                # General solution: if g = gcd(dx,dy), then dx/g * nx + dy/g * ny = -c/g
                # Need -c divisible by g
                g = gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                if (-c) % g != 0:
                    continue
                dxg = dx // g
                dyg = dy // g
                rhs = -c // g
                # Find one particular solution
                # dxg * nx + dyg * ny = rhs
                # Use extended gcd
                # dxg * x0 + dyg * y0 = 1, then multiply by rhs
                def _egcd(a, b):
                    if b == 0:
                        return (a, 1, 0)
                    g0, x1, y1 = _egcd(b, a % b)
                    return (g0, y1, x1 - (a // b) * y1)
                g_eg, x0, y0 = _egcd(dxg, dyg)
                # dxg*x0 + dyg*y0 = g_eg = 1
                px = x0 * rhs
                py = y0 * rhs
                # General: nx = px + dyg * t, ny = py - dxg * t
                # Find t closest to making (nx, ny) near (ox, oy)
                # Minimize (px + dyg*t - ox)^2 + (py - dxg*t - oy)^2
                # d/dt: 2*dyg*(px + dyg*t - ox) - 2*dxg*(py - dxg*t - oy) = 0
                # dyg*(px - ox) + dyg^2*t - dxg*(py - oy) + dxg^2*t = 0
                # t*(dxg^2 + dyg^2) = dxg*(py - oy) - dyg*(px - ox)
                denom = dxg * dxg + dyg * dyg
                if denom == 0:
                    continue
                num = dxg * (py - oy) - dyg * (px - ox)
                t0 = num / denom
                for t in [int(t0) - 2, int(t0) - 1, int(t0), int(t0) + 1, int(t0) + 2, int(t0) + 3, int(t0) - 3]:
                    nx = px + dyg * t
                    ny = py - dxg * t
                    if abs(nx) > 10**9 or abs(ny) > 10**9:
                        continue
                    # Check not duplicate
                    dup = False
                    for j in range(n):
                        if j != idx and current[j][0] == nx and current[j][1] == ny:
                            dup = True
                            break
                    if dup:
                        continue
                    result = _try_move(current, idx, nx, ny, line_counts, point_lines)
                    if result is None:
                        continue
                    new_score, new_lc, new_pl = result
                    if new_score < current_score:
                        current[idx] = [nx, ny]
                        line_counts = new_lc
                        point_lines = new_pl
                        current_score = new_score
                        improved = True
                        if new_score < best_score:
                            best_score = new_score
                            best = [p[:] for p in current]
                            best_lc = dict(line_counts)
                            best_pl = [set(s) for s in point_lines]
                        break
                if improved:
                    break
            if improved:
                break

        # Phase 2: small neighborhood moves
        if not improved:
            for idx in range(n):
                if time.time() > deadline:
                    break
                ox, oy = current[idx]
                for dx_m in range(-4, 5):
                    for dy_m in range(-4, 5):
                        if dx_m == 0 and dy_m == 0:
                            continue
                        nx = ox + dx_m
                        ny = oy + dy_m
                        if abs(nx) > 10**9 or abs(ny) > 10**9:
                            continue
                        dup = False
                        for j in range(n):
                            if j != idx and current[j][0] == nx and current[j][1] == ny:
                                dup = True
                                break
                        if dup:
                            continue
                        result = _try_move(current, idx, nx, ny, line_counts, point_lines)
                        if result is None:
                            continue
                        new_score, new_lc, new_pl = result
                        if new_score < current_score:
                            current[idx] = [nx, ny]
                            line_counts = new_lc
                            point_lines = new_pl
                            current_score = new_score
                            improved = True
                            if new_score < best_score:
                                best_score = new_score
                                best = [p[:] for p in current]
                                best_lc = dict(line_counts)
                                best_pl = [set(s) for s in point_lines]
                            break
                    if improved:
                        break
                if improved:
                    break

        # Phase 3: random perturbation if stuck
        if not improved and no_improve_count > 3:
            idx = random.randint(0, n - 1)
            ox, oy = current[idx]
            for _ in range(80):
                nx = ox + random.randint(-15, 15)
                ny = oy + random.randint(-15, 15)
                if abs(nx) > 10**9 or abs(ny) > 10**9:
                    continue
                dup = False
                for j in range(n):
                    if j != idx and current[j][0] == nx and current[j][1] == ny:
                        dup = True
                        break
                if dup:
                    continue
                result = _try_move(current, idx, nx, ny, line_counts, point_lines)
                if result is None:
                    continue
                new_score, new_lc, new_pl = result
                if new_score < current_score:
                    current[idx] = [nx, ny]
                    line_counts = new_lc
                    point_lines = new_pl
                    current_score = new_score
                    improved = True
                    no_improve_count = 0
                    if new_score < best_score:
                        best_score = new_score
                        best = [p[:] for p in current]
                        best_lc = dict(line_counts)
                        best_pl = [set(s) for s in point_lines]
                    break
                else:
                    # Accept with small probability (simulated annealing-like)
                    if random.random() < 0.02:
                        current[idx] = [nx, ny]
                        line_counts = new_lc
                        point_lines = new_pl
                        current_score = new_score
                        break
            if not improved:
                # Full restart: randomize one point more aggressively
                idx = random.randint(0, n - 1)
                for _ in range(200):
                    nx = random.randint(-30, 30)
                    ny = random.randint(-30, 30)
                    dup = False
                    for j in range(n):
                        if j != idx and current[j][0] == nx and current[j][1] == ny:
                            dup = True
                            break
                    if dup:
                        continue
                    result = _try_move(current, idx, nx, ny, line_counts, point_lines)
                    if result is None:
                        continue
                    new_score, new_lc, new_pl = result
                    if new_score < current_score:
                        current[idx] = [nx, ny]
                        line_counts = new_lc
                        point_lines = new_pl
                        current_score = new_score
                        improved = True
                        no_improve_count = 0
                        if new_score < best_score:
                            best_score = new_score
                            best = [p[:] for p in current]
                            best_lc = dict(line_counts)
                            best_pl = [set(s) for s in point_lines]
                        break
                if not improved:
                    break

        if not improved and no_improve_count > 15:
            break

    return best, best_score

def _elliptic_curve_points():
    """Generate 12 points from an elliptic curve structure for good initial collinearity."""
    # Use y^2 = x^3 - x (j=1728 curve)
    # Known rational points: (0,0), (1,0), (-1,0), (2, 2*sqrt(3))... not integer
    # Better: use a curve with many integer points
    # y^2 = x^3 + 1: (0,1), (0,-1), (-1,0), (2,3), (2,-3), ...
    # Let's use a different strategy: generate points on a conic + perturb
    
    # Strategy: use a 3x4 grid but shift diagonally to create extra collinearities
    # Points: (i, j) for i in 0..3, j in 0..2, but shift row j by j
    pts = []
    for j in range(3):
        for i in range(4):
            pts.append([i + j, j])
    return pts

def _initial_configs():
    """Generate diverse initial configurations."""
    configs = []
    
    # 3x4 grid
    configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # Diagonal-shift grid
    configs.append(_elliptic_curve_points())
    
    # Two rows of 6
    configs.append([[x, 0] for x in range(6)] + [[x, 1] for x in range(6)])
    
    # 4x3 grid
    configs.append([[x, y] for x in range(3) for y in range(4)])
    
    # 2x6 grid
    configs.append([[x, y] for x in range(6) for y in range(2)])
    
    # 6x2 grid
    configs.append([[x, y] for x in range(2) for y in range(6)])
    
    # 3x4 with offset rows
    pts = []
    for j in range(3):
        for i in range(4):
            pts.append([i, j * 2])
    configs.append(pts)
    
    # Near-elliptic: points on y^2 = x^3 + x (small integer points)
    # (0,0), (1,±sqrt(2)) not integer... use y^2 = x^3 - 2x
    # Just use structured grids with various spacings
    for spacing in [2, 3]:
        pts = [[i * spacing, j * spacing] for i in range(4) for j in range(3)]
        configs.append(pts)
    
    # Random configs
    for _ in range(30):
        pts = []
        while len(pts) < 12:
            x = random.randint(-20, 20)
            y = random.randint(-20, 20)
            if [x, y] not in pts:
                pts.append([x, y])
        if _not_all_collinear(pts):
            configs.append(pts)
    
    # Random configs on a small grid
    for _ in range(20):
        pts = []
        while len(pts) < 12:
            x = random.randint(-5, 5)
            y = random.randint(-5, 5)
            if [x, y] not in pts:
                pts.append([x, y])
        if _not_all_collinear(pts):
            configs.append(pts)
    
    return configs

def construct():
    deadline = time.time() + 135  # Safety margin before 150s kill
    random.seed(42)
    
    configs = _initial_configs()
    
    # Score all initial configs
    scored = []
    for cfg in configs:
        if not _not_all_collinear(cfg):
            continue
        score = _count_ordinary(cfg)
        scored.append((score, cfg))
    
    scored.sort(key=lambda x: x[0])
    
    best_config = scored[0][1] if scored else [[i, 0] for i in range(11)] + [[0, 1]]
    best_score = scored[0][0] if scored else float('inf')
    
    # Local search from top configs
    for score, cfg in scored[:10]:
        if time.time() > deadline:
            break
        if score > best_score + 3:
            break
        lc, pl = _build_line_data(cfg)
        # Quick check: if already at best, skip
        if score == best_score:
            continue
        result_config, result_score = _local_search(cfg, deadline)
        if result_score < best_score:
            best_score = result_score
            best_config = result_config
    
    # Also try local search from the absolute best initial if we haven't
    if scored:
        top_score, top_cfg = scored[0]
        if top_score == best_score:
            # Already tried, but try again with more time
            pass
    
    # Final verification
    if not _not_all_collinear(best_config):
        best_config = [[i, 0] for i in range(11)] + [[0, 1]]
    
    # Ensure exactly 12 distinct points
    seen = set()
    final = []
    for p in best_config:
        key = (p[0], p[1])
        if key not in seen:
            seen.add(key)
            final.append(p)
    while len(final) < 12:
        # Add a point that doesn't make all collinear
        for x in range(-50, 51):
            for y in range(-50, 51):
                if (x, y) not in seen:
                    test = final + [[x, y]]
                    if _not_all_collinear(test):
                        final.append([x, y])
                        seen.add((x, y))
                        break
            if len(final) >= 12:
                break
    
    return final[:12]

