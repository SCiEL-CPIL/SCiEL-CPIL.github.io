"""Improved search for 16 integer points maximizing exact-4-point lines."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def _line_key_from_points(p1, p2):
    """Compute canonical line key (a,b,c) for line through p1, p2 where a*x+b*y=c."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    g = math.gcd(abs(dx), abs(dy))
    if g == 0:
        return None
    dx_n, dy_n = dx // g, dy // g
    if dx_n < 0 or (dx_n == 0 and dy_n < 0):
        dx_n, dy_n = -dx_n, -dy_n
    a, b = dy_n, -dx_n
    c = a * x1 + b * y1
    g2 = math.gcd(abs(a), abs(b))
    if g2 > 1:
        a, b, c = a // g2, b // g2, c // g2
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _build_line_map(points):
    """Build dict: line_key -> set of point indices on that line."""
    n = len(points)
    line_pts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key_from_points(points[i], points[j])
            if key is None:
                continue
            if key not in line_pts:
                line_pts[key] = set()
            line_pts[key].add(i)
            line_pts[key].add(j)
    return line_pts

def _score_from_line_map(line_pts):
    """Count exact-4 lines and check for 5+ collinear."""
    exact_4 = 0
    has_5 = False
    for key, pts in line_pts.items():
        cnt = len(pts)
        if cnt == 4:
            exact_4 += 1
        elif cnt >= 5:
            has_5 = True
    return exact_4, has_5

def _score_points(points):
    line_pts = _build_line_map(points)
    return _score_from_line_map(line_pts)

def _incremental_score(points, moved_idx, old_line_map):
    """
    After moving point moved_idx, rebuild only lines involving that point.
    old_line_map: dict from before the move.
    Returns (new_line_map, exact_4, has_5).
    """
    n = len(points)
    # Remove old lines that included moved_idx
    # We need to figure out which lines had moved_idx
    # Simpler: rebuild all lines but only for pairs involving moved_idx,
    # and keep the rest from old_line_map
    new_line_map = {}
    # Copy lines that don't involve moved_idx
    for key, pts in old_line_map.items():
        if moved_idx not in pts:
            new_line_map[key] = set(pts)
    
    # Add new lines involving moved_idx
    for j in range(n):
        if j == moved_idx:
            continue
        key = _line_key_from_points(points[moved_idx], points[j])
        if key is None:
            continue
        if key not in new_line_map:
            new_line_map[key] = set()
        new_line_map[key].add(moved_idx)
        new_line_map[key].add(j)
    
    exact_4, has_5 = _score_from_line_map(new_line_map)
    return new_line_map, exact_4, has_5

def _find_3_lines(points, exclude_idx):
    """Find all lines with exactly 3 points among points excluding exclude_idx."""
    n = len(points)
    other = [i for i in range(n) if i != exclude_idx]
    lines_3 = {}
    for i in range(len(other)):
        for j in range(i + 1, len(other)):
            for k in range(j + 1, len(other)):
                p1, p2, p3 = points[other[i]], points[other[j]], points[other[k]]
                dx1 = p2[0] - p1[0]
                dy1 = p2[1] - p1[1]
                dx2 = p3[0] - p1[0]
                dy2 = p3[1] - p1[1]
                if dx1 * dy2 - dy1 * dx2 != 0:
                    continue
                key = _line_key_from_points(p1, p2)
                if key is None:
                    continue
                if key not in lines_3:
                    lines_3[key] = set()
                lines_3[key].add(other[i])
                lines_3[key].add(other[j])
                lines_3[key].add(other[k])
    return lines_3

def _completion_candidates(points, exclude_idx, rng, max_candidates=50):
    """Find candidate points that would complete a 3-point line to 4."""
    lines_3 = _find_3_lines(points, exclude_idx)
    candidates = []
    existing = set(tuple(p) for p in points)
    
    for key, pts_on_line in lines_3.items():
        a, b, c = key
        # Get x-coordinates of existing points on this line
        xs = set()
        for p in pts_on_line:
            xs.add(points[p][0])
        
        # Try nearby x values
        for x_base in sorted(xs):
            for dx in range(-3, 4):
                x_try = x_base + dx
                if abs(x_try) > 1000:
                    continue
                if b != 0:
                    num = c - a * x_try
                    if num % b == 0:
                        y_try = num // b
                        if abs(y_try) <= 1000:
                            pt = [x_try, y_try]
                            if tuple(pt) not in existing:
                                if pt not in candidates:
                                    candidates.append(pt)
                else:
                    # Vertical line: x = c/a
                    if a != 0 and c % a == 0:
                        x_val = c // a
                        if x_val == x_try:
                            for p in pts_on_line:
                                for dy in [-2, -1, 1, 2]:
                                    y_try = points[p][1] + dy
                                    if abs(y_try) <= 1000:
                                        pt = [x_try, y_try]
                                        if tuple(pt) not in existing:
                                            if pt not in candidates:
                                                candidates.append(pt)
    
    if not candidates:
        return []
    if len(candidates) > max_candidates:
        return rng.sample(candidates, max_candidates)
    return candidates

def _generate_moves(points, moved_idx, rng):
    """Generate a list of candidate new positions for point moved_idx."""
    old = points[moved_idx]
    moves = []
    
    # Completion candidates
    comp = _completion_candidates(points, moved_idx, rng)
    moves.extend(comp)
    
    # Random local moves
    for _ in range(8):
        nx = old[0] + rng.randint(-6, 6)
        ny = old[1] + rng.randint(-6, 6)
        if abs(nx) <= 1000 and abs(ny) <= 1000:
            pt = [nx, ny]
            if pt not in moves:
                moves.append(pt)
    
    # Larger jumps
    for _ in range(3):
        nx = old[0] + rng.randint(-15, 15)
        ny = old[1] + rng.randint(-15, 15)
        if abs(nx) <= 1000 and abs(ny) <= 1000:
            pt = [nx, ny]
            if pt not in moves:
                moves.append(pt)
    
    return moves

def _make_grid_4x4():
    return [[x, y] for x in range(4) for y in range(4)]

def _make_affine_grid(rng):
    """Generate a 4x4 grid with affine transformation."""
    base = _make_grid_4x4()
    # Random affine: x' = a*x + b*y + c, y' = d*x + e*y + f
    a = rng.randint(1, 3)
    b = rng.randint(-1, 1)
    d = rng.randint(-1, 1)
    e = rng.randint(1, 3)
    c = rng.randint(-5, 5)
    f = rng.randint(-5, 5)
    # Ensure determinant non-zero
    det = a * e - b * d
    if det == 0:
        e += 1
    pts = []
    for x, y in base:
        nx = a * x + b * y + c
        ny = d * x + e * y + f
        pts.append([nx, ny])
    # Check distinctness
    if len(set(tuple(p) for p in pts)) != 16:
        return _make_grid_4x4()
    return pts

def _make_perturbed_grid(rng):
    base = _make_grid_4x4()
    pts = [p[:] for p in base]
    for _ in range(rng.randint(2, 8)):
        idx = rng.randint(0, 15)
        pts[idx] = [pts[idx][0] + rng.randint(-3, 3), pts[idx][1] + rng.randint(-3, 3)]
    # Ensure distinct
    seen = set()
    for i, p in enumerate(pts):
        if tuple(p) in seen:
            pts[i] = [p[0] + rng.randint(1, 5), p[1] + rng.randint(1, 5)]
        seen.add(tuple(p))
    return pts

def _make_random_points(rng):
    pts = []
    seen = set()
    while len(pts) < 16:
        x = rng.randint(-20, 20)
        y = rng.randint(-20, 20)
        if (x, y) not in seen:
            seen.add((x, y))
            pts.append([x, y])
    return pts

def construct():
    deadline = time.time() + 140
    rng = random.Random(42)
    
    best_score = -1
    best_points = None
    
    # Starting configurations
    start_configs = [
        _make_grid_4x4(),
        _make_grid_4x4(),
        _make_grid_4x4(),
    ]
    for _ in range(6):
        start_configs.append(_make_affine_grid(rng))
    for _ in range(6):
        start_configs.append(_make_perturbed_grid(rng))
    for _ in range(4):
        start_configs.append(_make_random_points(rng))
    
    rng.shuffle(start_configs)
    
    for cfg_idx, start_pts in enumerate(start_configs):
        if time.time() > deadline - 15:
            break
        
        current = [p[:] for p in start_pts]
        
        # Check validity
        s, h5 = _score_points(current)
        if h5:
            # Try to fix by moving a point
            fixed = False
            for _ in range(20):
                idx = rng.randint(0, 15)
                old = current[idx][:]
                for _ in range(10):
                    nx = old[0] + rng.randint(-5, 5)
                    ny = old[1] + rng.randint(-5, 5)
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    if any([nx, ny] == p for p in current):
                        continue
                    current[idx] = [nx, ny]
                    s, h5 = _score_points(current)
                    if not h5:
                        fixed = True
                        break
                    current[idx] = old
                if fixed:
                    break
            if not h5:
                s, h5 = _score_points(current)
            else:
                continue
        
        line_map = _build_line_map(current)
        current_score = s
        
        # Local search
        T = 1.0
        no_improve = 0
        max_no_improve = 500
        steps = 0
        
        while time.time() < deadline - 10 and no_improve < max_no_improve:
            steps += 1
            idx = rng.randint(0, 15)
            old = current[idx][:]
            
            moves = _generate_moves(current, idx, rng)
            
            # Filter valid moves
            valid_moves = []
            for mv in moves:
                if abs(mv[0]) > 1000 or abs(mv[1]) > 1000:
                    continue
                if any(mv == p for p in current):
                    continue
                valid_moves.append(mv)
            
            if not valid_moves:
                no_improve += 1
                continue
            
            # Pick best move or use simulated annealing
            best_move = None
            best_move_score = -1
            best_move_map = None
            
            # Evaluate a subset of moves
            if len(valid_moves) > 15:
                eval_moves = rng.sample(valid_moves, 15)
            else:
                eval_moves = valid_moves
            
            for mv in eval_moves:
                current[idx] = mv
                new_map, s_new, h5_new = _incremental_score(current, idx, line_map)
                if h5_new:
                    continue
                if s_new > best_move_score:
                    best_move_score = s_new
                    best_move = mv
                    best_move_map = new_map
            
            if best_move is None:
                current[idx] = old
                no_improve += 1
                continue
            
            # Accept based on simulated annealing
            accept = False
            if best_move_score >= current_score:
                accept = True
            elif T > 0.05:
                delta = best_move_score - current_score
                if rng.random() < math.exp(delta / T):
                    accept = True
            
            if accept:
                current[idx] = best_move
                line_map = best_move_map
                if best_move_score > current_score:
                    no_improve = 0
                    if best_move_score > best_score:
                        best_score = best_move_score
                        best_points = [p[:] for p in current]
                else:
                    no_improve += 1
                current_score = best_move_score
            else:
                no_improve += 1
            
            T = max(0.05, T * 0.995)
        
        # Update global best
        s_final, h5_final = _score_points(current)
        if not h5_final and s_final > best_score:
            best_score = s_final
            best_points = [p[:] for p in current]
    
    if best_points is None:
        best_points = _make_grid_4x4()
    
    return best_points
# EVOLVE-BLOCK-END
