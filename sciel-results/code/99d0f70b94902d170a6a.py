"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def _count_ordinary_lines(pts):
    """Count lines containing exactly two of the points."""
    n = len(pts)
    if n < 3:
        return 0
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[j][0] - pts[i][0]
            dy = pts[j][1] - pts[i][1]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                key = (0, 0, pts[i][0])
            elif dy == 0:
                key = (0, 1, pts[i][1])
            else:
                if dy < 0:
                    dx, dy = -dx, -dy
                g = gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                c = dy * pts[i][0] - dx * pts[i][1]
                key = (1, dx, dy, c)
            lines[key] = lines.get(key, 0) + 1
    return sum(1 for c in lines.values() if c == 1)

def _evaluate(points):
    """Score: number of ordinary lines (lower is better)."""
    return _count_ordinary_lines(points)

def _not_all_collinear(points):
    n = len(points)
    if n < 3:
        return True
    for j in range(1, n):
        for k in range(j + 1, n):
            x0, y0 = points[0]
            x1, y1 = points[j]
            x2, y2 = points[k]
            if (x1 - x0) * (y2 - y0) != (y1 - y0) * (x2 - x0):
                return True
    return False

def _is_valid(points):
    """Check points are distinct, in bounds, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for p in points:
        if (p[0], p[1]) in seen:
            return False
        seen.add((p[0], p[1]))
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return False
    return _not_all_collinear(points)

def _generate_initial_candidates():
    """Generate diverse initial configurations."""
    candidates = []
    n = 12
    
    # 3x4 grid
    candidates.append([[x, y] for x in range(4) for y in range(3)])
    # 4x3 grid
    candidates.append([[x, y] for x in range(3) for y in range(4)])
    
    # Hexagonal patterns
    for s in [1, 2, 3]:
        pts = []
        for i in range(6):
            angle = i * math.pi / 3
            pts.append([round(s * 2 * math.cos(angle)), round(s * 2 * math.sin(angle))])
        pts.append([0, 0])
        for i in range(5):
            angle = i * math.pi / 3 + math.pi / 6
            pts.append([round(s * math.cos(angle)), round(s * math.sin(angle))])
        seen = set()
        unique = []
        for p in pts:
            if (p[0], p[1]) not in seen:
                seen.add((p[0], p[1]))
                unique.append(p)
        while len(unique) < 12:
            unique.append([len(unique) - 6, 0])
        if _is_valid(unique[:12]):
            candidates.append(unique[:12])
    
    # Pencil configurations
    for offset in [0, 1, 2]:
        pts = [[0, 0]]
        for i in range(11):
            angle = i * math.pi / 11 + offset * 0.1
            x = round(10 * math.cos(angle))
            y = round(10 * math.sin(angle))
            if (x, y) not in [(p[0], p[1]) for p in pts]:
                pts.append([x, y])
        while len(pts) < 12:
            pts.append([len(pts), 0])
        if _is_valid(pts[:12]):
            candidates.append(pts[:12])
    
    # Sheared grids
    for shear in [1, 2]:
        for w in [3, 4]:
            for h in [4, 3]:
                if w * h == 12:
                    pts = [[x + shear * y // 2, y] for x in range(w) for y in range(h)]
                    if _is_valid(pts):
                        candidates.append(pts)
    
    # Random configurations
    rng = random.Random(42)
    for _ in range(20):
        pts = [[rng.randint(-10, 10), rng.randint(-10, 10)] for _ in range(12)]
        if _is_valid(pts):
            candidates.append(pts)
    
    return candidates

def _snap_to_line(points, idx):
    """Generate candidate positions for point idx by snapping to lines through pairs of other points."""
    n = len(points)
    others = [i for i in range(n) if i != idx]
    candidates = set()
    occupied = {(points[i][0], points[i][1]) for i in others}
    
    for i in range(len(others)):
        for j in range(i + 1, len(others)):
            oi, oj = others[i], others[j]
            x1, y1 = points[oi]
            x2, y2 = points[oj]
            
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dxn = dx // g
            dyn = dy // g
            
            # Try positions along the line
            for t in range(-8, 9):
                nx = x1 + t * dxn
                ny = y1 + t * dyn
                if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                    if (nx, ny) not in occupied:
                        candidates.add((nx, ny))
            
            # Midpoint
            if (x1 + x2) % 2 == 0 and (y1 + y2) % 2 == 0:
                mx, my = (x1 + x2) // 2, (y1 + y2) // 2
                if (mx, my) not in occupied:
                    candidates.add((mx, my))
            
            # Thirds
            if dx % 3 == 0 and dy % 3 == 0:
                for frac in [1, 2]:
                    fx = x1 + frac * dx // 3
                    fy = y1 + frac * dy // 3
                    if (fx, fy) not in occupied:
                        candidates.add((fx, fy))
    
    return list(candidates)

def _random_near(points, idx, radius=15):
    """Generate random positions near other points."""
    others = [points[i] for i in range(len(points)) if i != idx]
    cx = sum(p[0] for p in others) // len(others)
    cy = sum(p[1] for p in others) // len(others)
    candidates = set()
    occupied = {(p[0], p[1]) for p in others}
    rng = random.Random()
    
    for _ in range(50):
        nx = cx + rng.randint(-radius, radius)
        ny = cy + rng.randint(-radius, radius)
        if abs(nx) <= 10**9 and abs(ny) <= 10**9 and (nx, ny) not in occupied:
            candidates.add((nx, ny))
    
    return list(candidates)

def construct():
    start = time.time()
    deadline = start + 130  # safety margin
    
    n = 12
    rng = random.Random(42)
    
    # Generate initial candidates
    candidates = _generate_initial_candidates()
    
    # Evaluate all and pick best
    best_points = None
    best_score = float('inf')
    
    for cand in candidates:
        if not _is_valid(cand):
            continue
        score = _evaluate(cand)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in cand]
    
    if best_points is None:
        best_points = [[i, i * i] for i in range(12)]
        best_score = _evaluate(best_points)
    
    # Simulated Annealing with Line Snapping
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    while time.time() < deadline and T > T_min:
        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.6:
            # Line snapping: move one point to a position on a line through two others
            idx = rng.randrange(n)
            snap_candidates = _snap_to_line(current_points, idx)
            if not snap_candidates:
                T *= cooling
                continue
            nx, ny = rng.choice(snap_candidates)
        elif move_type < 0.8:
            # Random near centroid
            idx = rng.randrange(n)
            near_candidates = _random_near(current_points, idx, radius=20)
            if not near_candidates:
                T *= cooling
                continue
            nx, ny = rng.choice(near_candidates)
        else:
            # Swap two points' positions (perturbation)
            i, j = rng.sample(range(n), 2)
            current_points[i], current_points[j] = current_points[j], current_points[i]
            new_score = _evaluate(current_points)
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
            else:
                current_points[i], current_points[j] = current_points[j], current_points[i]
            T *= cooling
            continue
        
        old_x, old_y = current_points[idx]
        current_points[idx] = [nx, ny]
        
        if not _is_valid(current_points):
            current_points[idx] = [old_x, old_y]
            T *= cooling
            continue
        
        new_score = _evaluate(current_points)
        delta = new_score - current_score
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
        
        T *= cooling
    
    # Final hill-climbing polish
    if time.time() < deadline - 5:
        for _ in range(200):
            if time.time() > deadline - 2:
                break
            improved = False
            for idx in range(n):
                if time.time() > deadline - 2:
                    break
                old_x, old_y = current_points[idx]
                snap_candidates = _snap_to_line(current_points, idx)
                near_candidates = _random_near(current_points, idx, radius=10)
                all_cands = snap_candidates + near_candidates
                if not all_cands:
                    continue
                
                for nx, ny in all_cands:
                    if (nx, ny) == (old_x, old_y):
                        continue
                    if abs(nx) > 10**9 or abs(ny) > 10**9:
                        continue
                    # Check duplicate
                    dup = False
                    for i, (px, py) in enumerate(current_points):
                        if i != idx and px == nx and py == ny:
                            dup = True
                            break
                    if dup:
                        continue
                    
                    current_points[idx] = [nx, ny]
                    if not _is_valid(current_points):
                        current_points[idx] = [old_x, old_y]
                        continue
                    
                    score = _evaluate(current_points)
                    if score < current_score:
                        current_score = score
                        improved = True
                        break
                    else:
                        current_points[idx] = [old_x, old_y]
                
                if improved:
                    break
            
            if current_score < best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
            
            if not improved:
                break
    
    # Final validation
    assert len(best_points) == 12
    assert all(isinstance(p, list) and len(p) == 2 for p in best_points)
    assert all(abs(p[0]) <= 10**9 and abs(p[1]) <= 10**9 for p in best_points)
    assert _not_all_collinear(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
